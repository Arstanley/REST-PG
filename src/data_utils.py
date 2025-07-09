"""
Data utilities for REST-PG implementation
"""
import json
import jsonlines
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
import json
import jsonlines
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from rouge_score import rouge_scorer

from config import config


def load_json_data(file_path: str) -> List[Dict]:
    """Load data from JSON file and convert to REST-PG format"""
    print(f"Loading data from {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to REST-PG format
    rest_pg_data = []
    
    for user in data:
        if not isinstance(user, dict) or 'id' not in user or 'profile' not in user:
            continue
            
        user_id = user['id']
        profile = user['profile']
        
        if not isinstance(profile, list) or len(profile) < 1:
            continue
        
        # Convert profile to string format
        profile_strings = []
        for review in profile:
            if isinstance(review, dict) and 'rating' in review and 'title' in review and 'text' in review:
                profile_strings.append(f"[{review['rating']} stars] {review['title']} - {review['text']}")
        
        if len(profile_strings) < 1:
            continue
        
        # Create training examples (leave-one-out for each review)
        for i, review in enumerate(profile):
            if not isinstance(review, dict) or 'text' not in review:
                continue
                
            # Create prompt for this review
            prompt = f"Write a review for product {review.get('pid', 'unknown')}"
            expected_output = review['text']
            
            # Use other reviews as profile (leave-one-out)
            other_profile = [profile_strings[j] for j in range(len(profile_strings)) if j != i]
            
            if len(other_profile) < 1:
                continue
            
            rest_pg_data.append({
                'x': prompt,
                'y': expected_output,
                'P': other_profile,
                'user_id': user_id
            })
    
    print(f"Converted {len(rest_pg_data)} training examples from {len(data)} users")
    return rest_pg_data


def save_jsonl_data(data: List[Dict], file_path: str):
    """Save data to JSONL format"""
    with jsonlines.open(file_path, 'w') as writer:
        for item in data:
            writer.write(item)  # type: ignore


def calculate_rouge_reward(expected: str, generated: str) -> float:
    """Calculate ROUGE-based reward as average of ROUGE-1 and ROUGE-L scores"""
    try:
        # Initialize ROUGE scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        
        # Calculate ROUGE scores
        scores = scorer.score(expected, generated)
        
        # Extract F1 scores for ROUGE-1 and ROUGE-L
        rouge1_f1 = scores['rouge1'].fmeasure
        rougeL_f1 = scores['rougeL'].fmeasure
        
        # Return average of ROUGE-1 and ROUGE-L F1 scores
        reward = (rouge1_f1 + rougeL_f1) / 2.0
        
        return reward
    except Exception as e:
        print(f"Error calculating ROUGE reward: {e}")
        return 0.0  # Default reward if calculation fails


class PersonalizedTextDataset(Dataset):
    """Dataset for personalized text generation tasks"""

    def __init__(self, data_path: str, tokenizer):
        self.tokenizer = tokenizer
        self.data = []
        
        # Load data with proper error handling
        try:
            if data_path.endswith('.json'):
                # Load JSON data and convert to REST-PG format
                self.data = load_json_data(data_path)
            else:
                # Load JSONL data
                with jsonlines.open(data_path) as reader:
                    self.data = [item for item in reader if isinstance(item, dict) and 'x' in item and 'y' in item]  # type: ignore
        except Exception as e:
            print(f"Error loading data from {data_path}: {e}")
            self.data = []
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format input with personalized context
        input_text = self._format_input(item['x'], item.get('P', []))
        target_text = item['y']
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=config.model.max_input_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        targets = self.tokenizer(
            target_text,
            max_length=config.model.max_output_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }
    
    def _format_input(self, prompt: str, profile: List[str]) -> str:
        """Format input with personalized context using RAG"""
        # Simple concatenation for now - can be enhanced with proper RAG
        context = "\n".join(profile[:config.restpg.retrieval_top_k])
        return f"Context:\n{context}\n\nPrompt: {prompt}\n\nResponse:"


class ReasoningDataGenerator:
    """Generate reasoning paths for training data"""
    
    def __init__(self, reasoning_model, tokenizer):
        self.reasoning_model = reasoning_model  # Will be loaded when needed
        self.tokenizer = tokenizer 
           
    def generate_reasoning(self, prompt: str) -> str:
        """Generate reasoning path for a given prompt using Figure 7 template""" 

        # Ensure model and tokenizer are loaded
        assert self.reasoning_model is not None, "Reasoning model must be loaded"
        assert self.tokenizer is not None, "Tokenizer must be loaded"
            
        # Format as chat messages for Llama-3-Instruct
        messages = [
            {"role": "system", "content": "You are a professional writing assistant whose task is to summarize the writing style of a user from their profile documents."},
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Generate reasoning using the formatted prompt
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.reasoning_model.device)
        
        with torch.no_grad():
            outputs = self.reasoning_model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        reasoning = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return reasoning.strip()
    
    def generate_reasoning_dataset(self, data_path: str, output_path: str):
        """Generate reasoning dataset for training"""
        print("Generating reasoning dataset...")
        
        data = load_json_data(data_path)
        
        reasoning_data = []
        
        for item in tqdm(data, desc="Generating reasoning paths"):
            reasoning = self.generate_reasoning(item['x'])

            # Combine reasoning with expected output
            combined_output = f"Reasoning: {reasoning}\n\nOutput: {item['y']}"
            
            reasoning_data.append({
                'x': item['x'],
                'y': combined_output,
                'P': item.get('P', []),
                'user_id': item.get('user_id', 'unknown')
            })
        
        # Save reasoning dataset
        try:
            save_jsonl_data(reasoning_data, output_path)
            print(f"Reasoning dataset saved to {output_path}")
        except Exception as e:
            print(f"Error saving reasoning dataset to {output_path}: {e}")


def create_data_loaders(train_path: str, val_path: str, tokenizer) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders"""
    train_dataset = PersonalizedTextDataset(train_path, tokenizer)
    val_dataset = PersonalizedTextDataset(val_path, tokenizer)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader 