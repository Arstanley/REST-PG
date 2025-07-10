"""
REST-PG: Reasoning-Enhanced Self-Training for Personalized Text Generation
Main implementation of the algorithm
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
)
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from transformers import DataCollatorForLanguageModeling
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    PeftMixedModel,
)
from torch.optim.adamw import AdamW
from tqdm import tqdm
import json
import jsonlines
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from pathlib import Path
import random
from datasets import Dataset
import os

from src.config import config
from src.data_utils import ReasoningDataGenerator, calculate_rouge_reward, load_json_data, save_jsonl_data, create_dataset, create_reasoning_dataset, save_dataset_to_jsonl, validate_reasoning_data


class RESTPGTrainer:
    """Main trainer for REST-PG algorithm"""
    def __init__(self):
        self.tokenizer, self.model = self.load_model(config.model.model_name)
        self.reasoning_generator = ReasoningDataGenerator(self.model, self.tokenizer)
        
    def load_model(self, model_name): 
        print(f"Loading model: {model_name}")
        
        # Check GPU availability
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name()}")
            device = "cuda"
        else:
            print("GPU not available, using CPU")
            device = "cpu"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16
        )
        
        model.gradient_checkpointing_enable()

        # model.to(device)
        # # Add padding token if not present
        # if tokenizer.pad_token is None:  # type: ignore
        #     tokenizer.pad_token = tokenizer.eos_token  # type: ignore
            
        # Setup LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,  # LoRA rank
            lora_alpha=32,  # LoRA alpha parameter
            lora_dropout=0.1,  # LoRA dropout
            target_modules=["q_proj", "v_proj"]
        )
        
        # Create PEFT model with LoRA
        model = get_peft_model(model, lora_config) # type: ignore
        # model.to(device)
        model.print_trainable_parameters() # type: ignore
        
        torch.cuda.empty_cache()
        return tokenizer, model

    def generate_reasoning_dataset(self, data_path: str, output_path: str):
        """Stage 1: Generate reasoning dataset using Figure 7 prompt with Hugging Face datasets"""
        print("Stage 1: Generating reasoning dataset using Hugging Face datasets...")
        
        # Create reasoning dataset using the new function
        reasoning_dataset = create_reasoning_dataset(data_path, self.tokenizer, self.reasoning_generator)
        
        # Save reasoning dataset
        save_dataset_to_jsonl(reasoning_dataset, output_path)
        
        print(f"Reasoning dataset saved to {output_path}")
        
    def supervised_fine_tuning(self, train_path: str, val_path: str, output_dir: str):
        """Stage 2: Supervised fine-tuning on reasoning data using LoRA"""
        print("Stage 2: Supervised fine-tuning on reasoning data with LoRA...")
        
        torch.cuda.empty_cache()
        # Ensure model is loaded
        assert self.model is not None, "LoRA model must be loaded before training"
        assert self.tokenizer is not None, "Tokenizer must be loaded before training"
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Check GPU availability
        if torch.cuda.is_available():
            print(f"Training on GPU: {torch.cuda.get_device_name()}")
            device = "cuda"
        else:
            print("Training on CPU")
            device = "cpu"
        
        # Load data
        train_dataset = create_dataset(train_path, self.tokenizer)
        val_dataset = create_dataset(val_path, self.tokenizer)
        
        trainer = Trainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=TrainingArguments(
                per_device_train_batch_size=1,
                gradient_accumulation_steps=8,
                warmup_steps=100,
                num_train_epochs=config.training.num_epochs,
                learning_rate=config.training.learning_rate,
                fp16=False,
                bf16=True,
                logging_steps=5,
                save_strategy="steps",
                output_dir=output_dir,
                save_total_limit=3,
                optim="adamw_torch",
                report_to='wandb' 
            ),
            data_collator= DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )
        self.model.config.use_cache = False
        trainer.train()
        
        # Save the LoRA adapters
        trainer.save_model(f"{output_dir}/lora_model")
        print(f"LoRA training completed. Model saved to {output_dir}/lora_model")
    
    def _cleanup_model(self):
        """Clean up model from GPU memory to avoid CUDA OOM"""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        torch.cuda.empty_cache()
        print("Cleaned up models from GPU memory")

    def expectation_step(self, data_path: str, output_dir: str = None, iteration: int = None, dataset_name: str = None) -> List[Dict]:
        """Expectation step: Generate multiple outputs for each input using the last tuned model"""
        
        if output_dir is not None:
            if not self._load_most_recent_model(output_dir, dataset_name, iteration):
                print("Warning: No tuned model found.")
        else:
            print("No output_dir provided, using original model for generation")
                  
        # Load data using the data utilities
        data = load_json_data(data_path)
                
        generated_data = []
        
        for item in tqdm(data, desc="Generating outputs"):
            if not isinstance(item, dict) or 'x' not in item or 'y' not in item or 'P' not in item:
                continue
                
            prompt = self._format_input(item['x'], item['P'])
            
            # Generate multiple outputs
            outputs = []
            for _ in range(config.restpg.exploration_budget):
                output = self._generate_output(prompt)
                outputs.append(output)
            
            # Evaluate outputs using ROUGE-based rewards
            rewards = []
            for output in outputs:
                reward = calculate_rouge_reward(item['y'], output)
                rewards.append(reward)
            
            # Select high-reward outputs
            high_reward_outputs = []
            for output, reward in zip(outputs, rewards):
                if reward >= config.restpg.reward_threshold:
                    high_reward_outputs.append((output, reward))
            
            # Limit number of outputs per input
            high_reward_outputs = sorted(high_reward_outputs, key=lambda x: x[1], reverse=True)
            high_reward_outputs = high_reward_outputs[:config.restpg.max_outputs_per_input]
            
            # Add to generated data
            for output, reward in high_reward_outputs:
                generated_data.append({
                    'x': item['x'],
                    'y': output,
                    'P': item['P'],
                    'user_id': item.get('user_id', 'unknown'),
                    'reward': reward
                })
        
        return generated_data
    
    def maximization_step(self, generated_data: List[Dict], output_dir: str, iteration: int=0, dataset_name: str = None):
        """Maximization step: Train model on high-reward outputs using LoRA with Hugging Face datasets"""
        print("Maximization step: Training on high-reward outputs with LoRA using Hugging Face datasets...")
        
        # Set dataset name from config if not provided
        if dataset_name is None:
            dataset_name = config.dataset_name
        
        # Load the previously tuned model if it exists
        if iteration > 0:
            # Try to load from the previous iteration checkpoint
            prev_checkpoint = f"{output_dir}/iteration_{iteration}"
            if Path(prev_checkpoint).exists():
                self._load_model_from_path(prev_checkpoint, f"iteration {iteration}")
            else:
                print(f"Warning: Previous checkpoint not found at {prev_checkpoint}, using current model")
        else:
            # For the first iteration, try to load from SFT model
            sft_model_path = f"{output_dir}/sft/lora_model"
            if Path(sft_model_path).exists():
                self._load_model_from_path(sft_model_path, "SFT model")
            else:
                print("Warning: SFT model not found, using current model for maximization step")
        
        # Create dataset from generated data
        dataset = Dataset.from_list(generated_data)
        
        # Format and tokenize the dataset
        dataset = dataset.map(
            lambda examples: {
                'formatted_input': [
                    f"Context:\n{' | '.join(examples.get('P', [[]])[i][:config.restpg.retrieval_top_k])}\n\nPrompt: {examples['x'][i]}\n\nResponse:"
                    for i in range(len(examples['x']))
                ]
            },
            batched=True
        )
        
        # Tokenize
        def tokenize_function(examples):
            inputs = self.tokenizer(
                examples['formatted_input'],
                max_length=config.model.max_input_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            targets = self.tokenizer(
                examples['y'],
                max_length=config.model.max_output_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            # Handle tensor dimensions correctly for batched processing
            if len(examples['formatted_input']) == 1:
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
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        # Check GPU availability
        if torch.cuda.is_available():
            print(f"Training on GPU: {torch.cuda.get_device_name()}")
            device = "cuda"
        else:
            print("Training on CPU")
            device = "cpu"
        
        # Setup training arguments for LoRA
        training_args = TrainingArguments(
            output_dir=f"{output_dir}/maximization_step",
            num_train_epochs=1,  # Single epoch for maximization step
            per_device_train_batch_size=config.training.batch_size,
            warmup_steps=50,
            weight_decay=config.training.weight_decay,
            logging_dir=f"{output_dir}/maximization_logs",
            logging_steps=10,
            save_strategy="no",  # Don't save checkpoints during maximization
            learning_rate=config.training.learning_rate,
            fp16=True if device == "cuda" else False,
            max_grad_norm=config.training.gradient_clip,
            dataloader_pin_memory=False,
            no_cuda=False if device == "cuda" else True,
            remove_unused_columns=False,
        )
        
        # Create trainer with LoRA model
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
        )
        
        # # Train the LoRA model
        # trainer.train()
        
        print(f"Maximization step completed. LoRA weights updated.")
        
    def train_rest_pg(self, train_path: str, val_path: str, output_dir: str, dataset_name: str = None, skip_reasoning_generation: bool = True):
        """Complete REST-PG training pipeline"""
        print("Starting REST-PG training pipeline...")
        
        # Set dataset name from config if not provided
        if dataset_name is None:
            dataset_name = config.dataset_name
        
        # Create dataset-specific output directory
        dataset_output_dir = f"{output_dir}/{dataset_name}"
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        print(f"Using dataset-specific output directory: {dataset_output_dir}")
        
        # Stage 1: Generate reasoning dataset (if needed)
        reasoning_data_path = f"{dataset_output_dir}/reasoning_data.jsonl"
        
        if skip_reasoning_generation and os.path.exists(reasoning_data_path):
            print("Using existing reasoning data...")
            # Validate the existing reasoning data
            if not validate_reasoning_data(reasoning_data_path):
                raise ValueError(f"Invalid reasoning data format at {reasoning_data_path}")
        else:
            print("Stage 1: Generating reasoning dataset...")
            self.generate_reasoning_dataset(train_path, reasoning_data_path)
        
        # Stage 2: Supervised fine-tuning on reasoning data
        print("Stage 2: Supervised fine-tuning on reasoning data...")
        self.supervised_fine_tuning(reasoning_data_path, val_path, f"{dataset_output_dir}/sft")
        
        # Stage 3: Expectation-Maximization self-training
        for iteration in range(config.restpg.num_iterations):
            print(f"\nIteration {iteration + 1}/{config.restpg.num_iterations}")
            
            # Expectation step
            generated_data = self.expectation_step(train_path, dataset_output_dir, iteration + 1, dataset_name)
            
            # Maximization step
            self.maximization_step(generated_data, dataset_output_dir, iteration + 1, dataset_name)
            
            # Save checkpoint with dataset name
            checkpoint_dir = f"{dataset_output_dir}/iteration_{iteration + 1}"
            self.model.save_pretrained(checkpoint_dir)
            self.tokenizer.save_pretrained(checkpoint_dir)
            
            print(f"Saved checkpoint: {checkpoint_dir}")
        
        print("REST-PG training completed!")

        self._cleanup_model()
    
    def _load_most_recent_model(self, output_dir: str, dataset_name: str = None, iteration: int = None):
        """Load the most recent model checkpoint"""
        # Set dataset name from config if not provided
        if dataset_name is None:
            dataset_name = config.dataset_name
        
        # Use dataset-specific output directory
        dataset_output_dir = f"{output_dir}/{dataset_name}"
        
        if iteration is not None and iteration > 0:
            # Try to load from the specific iteration checkpoint
            checkpoint_path = f"{dataset_output_dir}/iteration_{iteration}"
            if Path(checkpoint_path).exists():
                return self._load_model_from_path(checkpoint_path, f"iteration {iteration}")
            
        # Try to find the most recent iteration checkpoint
        iteration_dirs = [d for d in Path(dataset_output_dir).glob("iteration_*") if d.is_dir()]
        if iteration_dirs:
            # Sort by iteration number and get the most recent
            latest_iteration = max(iteration_dirs, key=lambda x: int(x.name.split('_')[1]))
            return self._load_model_from_path(str(latest_iteration), f"latest iteration {latest_iteration.name}")
            
        # Fall back to SFT model
        sft_model_path = f"{dataset_output_dir}/sft/lora_model"
        if Path(sft_model_path).exists():
            return self._load_model_from_path(sft_model_path, "SFT model")
            
        return False
    
    def _load_model_from_path(self, model_path: str, model_description: str):
        """Load model from a specific path"""
        try:
            print(f"Loading {model_description} from {model_path}")
            
            # Clean up old model to avoid CUDA OOM
            self._cleanup_model()
            
            # Load the base model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                config.model.model_name,
                torch_dtype=torch.float16
            )
            
            # Check GPU availability
            if torch.cuda.is_available():
                print(f"Using GPU: {torch.cuda.get_device_name()}")
                device = "cuda"
            else:
                print("GPU not available, using CPU")
                device = "cpu"
            
            model.to(device)
            
            # Load the LoRA adapters
            from peft import PeftModel
            peft_model = PeftModel.from_pretrained(model, model_path)
            
            # Update the model references
            self.tokenizer = tokenizer
            self.model = model
            self.peft_model = peft_model
            
            print(f"Successfully loaded {model_description}")
            return True
            
        except Exception as e:
            print(f"Failed to load {model_description} from {model_path}: {e}")
            return False
    
    def _format_input(self, prompt: str, profile: List[str]) -> str:
        """Format input with personalized context"""
        context = "\n".join(profile[:config.restpg.retrieval_top_k])
        return f"Context:\n{context}\n\nPrompt: {prompt}\n\nResponse:"
    
    def _generate_output(self, prompt: str) -> str:
        """Generate output for a given prompt"""
        # Format as chat messages for Llama-3-Instruct
        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates personalized text based on user context and preferences."},
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.model.max_output_length,
                temperature=config.model.temperature,
                do_sample=config.model.do_sample,
                top_p=config.model.top_p,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return generated_text.strip()
    
    def evaluate(self, test_path: str) -> Dict[str, float]:
        """Evaluate the trained model"""
        print("Evaluating model...")
        
        # Load test data using the data utilities
        test_data = load_json_data(test_path)
        
        rewards = []
        
        for item in tqdm(test_data, desc="Evaluating"):
            if not isinstance(item, dict) or 'x' not in item or 'y' not in item or 'P' not in item:
                continue
                
            prompt = self._format_input(item['x'], item['P'])
            generated_output = self._generate_output(prompt)
            
            reward = calculate_rouge_reward(item['y'], generated_output)
            rewards.append(reward)
        
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        
        results = {
            'average_reward': avg_reward,
            'std_reward': std_reward,
            'num_samples': len(rewards)
        }
        
        print(f"Evaluation results: {results}")
        return results


def main():
    """Main training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="REST-PG Training")
    parser.add_argument("--train_path", default=config.data.train_data_path, type=str, help="Path to training data")
    parser.add_argument("--val_path", default=config.data.val_data_path, help="Path to validation data")
    parser.add_argument("--test_path", default=config.data.test_data_path, help="Path to test data")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--dataset_name", type=str, default="default", help="Dataset name for organizing models")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8b", help="Model name to use")
    parser.add_argument("--skip_reasoning_generation", default=True, action="store_true", help="Skip reasoning generation if reasoning data already exists")
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    
    # Set dataset name
    config.dataset_name = args.dataset_name
    
    # Create trainer
    trainer = RESTPGTrainer()
    
    # Load model
    # trainer.load_model(args.model_name)
    
    # Train
    trainer.train_rest_pg(args.train_path, args.val_path, args.output_dir, args.dataset_name, args.skip_reasoning_generation)
    
    # Evaluate
    results = trainer.evaluate(args.test_path)
    
    # Save results
    with open(f"{args.output_dir}/evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main() 