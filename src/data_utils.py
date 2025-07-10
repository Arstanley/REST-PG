"""
Data utilities for REST-PG implementation
"""
import json
import jsonlines
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
import json
import jsonlines
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from rouge_score import rouge_scorer
from datasets import Dataset, load_dataset, concatenate_datasets

from src.config import config


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


def save_dataset_to_jsonl(dataset: Dataset, file_path: str):
    """Save a Hugging Face Dataset to JSONL format"""
    dataset.to_json(file_path)
    print(f"Dataset saved to {file_path}")


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


def validate_reasoning_data(data_path: str) -> bool:
    """Validate that existing reasoning data is in the correct format"""
    print(f"Validating reasoning data at {data_path}")
    
    try:
        # Load the dataset
        if data_path.endswith('.json'):
            dataset = Dataset.from_list(load_json_data(data_path))
        else:
            dataset = load_dataset('json', data_files=data_path)['train']
        
        # Check required columns
        required_columns = ['x', 'y']
        missing_columns = [col for col in required_columns if col not in dataset.column_names]
        
        if missing_columns:
            print(f"✗ Missing required columns: {missing_columns}")
            return False
        
        # Check that dataset is not empty
        if len(dataset) == 0:
            print("✗ Dataset is empty")
            return False
        
        # Check a few examples to ensure they have the expected structure
        sample = dataset[0]
        if not isinstance(sample['x'], str) or not isinstance(sample['y'], str):
            print("✗ Data format error: 'x' and 'y' should be strings")
            return False
        
        # Check if reasoning is included in the output (optional)
        if 'reasoning' in sample['y'].lower() or 'reasoning:' in sample['y']:
            print("✓ Reasoning data appears to be properly formatted")
        else:
            print("⚠ Warning: Output doesn't appear to contain reasoning format")
        
        print(f"✓ Reasoning data validation passed: {len(dataset)} examples")
        return True
        
    except Exception as e:
        print(f"✗ Error validating reasoning data: {e}")
        return False


def create_dataset(data_path: str, tokenizer) -> Dataset:
    """Create a Hugging Face Dataset from JSON or JSONL data."""
    print(f"Loading data from {data_path} and converting to Hugging Face Dataset...")
    
    # Load data based on file extension
    if data_path.endswith('.json'):
        # Load JSON data and convert to REST-PG format
        raw_data = load_json_data(data_path)
        # Convert to list of dicts for Dataset creation
        dataset = Dataset.from_list(raw_data)
    else:
        # Load JSONL data
        dataset = load_dataset('json', data_files=data_path)['train']
    
    # Filter out items that don't have 'x' and 'y'
    dataset = dataset.filter(lambda x: 'x' in x and 'y' in x)

    # Tokenize the dataset
    def tokenize_function(examples):
        inputs = tokenizer(
            examples['x'],
            max_length=config.model.max_input_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        targets = tokenizer(
            examples['y'],
            max_length=config.model.max_output_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Handle tensor dimensions correctly for batched processing
        if len(examples['x']) == 1:
            # Single example
            return {
                'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze(),
                'labels': targets['input_ids'].squeeze()
            }
        else:
            # Multiple examples
            return {
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask'],
                'labels': targets['input_ids']
            }
    
    dataset = dataset.map(tokenize_function, batched=True)
    
    print(f"Dataset created with {len(dataset)} examples.")
    return dataset


def create_reasoning_dataset(data_path: str, tokenizer, reasoning_generator) -> Dataset:
    """Create a reasoning dataset using Hugging Face datasets"""
    print("Creating reasoning dataset using Hugging Face datasets...")
    
    # Load the base dataset
    base_dataset = create_dataset(data_path, tokenizer)
    
    # Generate reasoning for each example
    def add_reasoning(examples):
        reasoning_summaries = []
        for i in range(len(examples['x'])):
            # Format profile as concatenated documents with | separator
            profile_text = " | ".join(examples.get('P', [[]])[i])
            
            # Generate reasoning using Figure 7 prompt with chat template
            reasoning_prompt = config.restpg.reasoning_prompt_template.format(
                profile=profile_text,
                subject=examples['x'][i],
                expected_output=examples['y'][i]
            )
            
            reasoning_summary = reasoning_generator.generate_reasoning(reasoning_prompt)
            reasoning_summaries.append(reasoning_summary)
        
        return {'reasoning_summary': reasoning_summaries}
    
    # Add reasoning to dataset
    reasoning_dataset = base_dataset.map(add_reasoning, batched=True, batch_size=1)
    
    # Format for SFT training
    def format_for_sft(examples):
        sft_inputs = []
        sft_outputs = []
        
        for i in range(len(examples['x'])):
            # Format profile as concatenated documents with | separator
            profile_text = " | ".join(examples.get('P', [[]])[i])
            
            # Create SFT format using Figure 8 prompt templates
            sft_input = config.restpg.sft_input_template.format(
                instruction=examples['x'][i],
                context=profile_text
            )
            sft_output = config.restpg.sft_output_template.format(
                reasoning_summary=examples['reasoning_summary'][i],
                expected_output=examples['y'][i]
            )
            
            sft_inputs.append(sft_input)
            sft_outputs.append(sft_output)
        
        return {
            'sft_input': sft_inputs,
            'sft_output': sft_outputs
        }
    
    # Format for SFT
    sft_dataset = reasoning_dataset.map(format_for_sft, batched=True)
    
    # Rename columns for training
    sft_dataset = sft_dataset.rename_column('sft_input', 'x')
    sft_dataset = sft_dataset.rename_column('sft_output', 'y')
    
    print(f"Reasoning dataset created with {len(sft_dataset)} examples")
    return sft_dataset


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
    """Create training and validation data loaders using Hugging Face datasets"""
    train_dataset = create_dataset(train_path, tokenizer)
    val_dataset = create_dataset(val_path, tokenizer)
    
    # Convert to PyTorch format for DataLoader
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
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