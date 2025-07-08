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
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    PeftMixedModel
)
from torch.optim.adamw import AdamW
from tqdm import tqdm
import json
import jsonlines
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from pathlib import Path
import random

from config import config
from data_utils import PersonalizedTextDataset, ReasoningDataGenerator, calculate_rouge_reward, load_json_data, save_jsonl_data


class RESTPGTrainer:
    """Main trainer for REST-PG algorithm"""
    def __init__(self):
        self.tokenizer, self.model, self.peft_model = self.load_model(config.model.model_name)
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
            torch_dtype=torch.float16
        )
        model.to(device)
        # # Add padding token if not present
        # if tokenizer.pad_token is None:  # type: ignore
        #     tokenizer.pad_token = tokenizer.eos_token  # type: ignore
            
        # Setup LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,  # LoRA rank
            lora_alpha=32,  # LoRA alpha parameter
            lora_dropout=0.1,  # LoRA dropout
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        # Create PEFT model with LoRA
        peft_model = get_peft_model(model, lora_config) # type: ignore
        peft_model.print_trainable_parameters() # type: ignore
        
        return tokenizer, model, peft_model

    def generate_reasoning_dataset(self, data_path: str, output_path: str):
        """Stage 1: Generate reasoning dataset using Figure 7 prompt"""
        print("Stage 1: Generating reasoning dataset...")
        
        # Load data using the data utilities
        data = load_json_data(data_path)
               
        reasoning_data = []
        
        for item in tqdm(data, desc="Generating reasoning"):
            if not isinstance(item, dict) or 'x' not in item or 'y' not in item or 'P' not in item:
                continue
                
            # Format profile as concatenated documents with | separator
            profile_text = " | ".join(item['P'])
            
            # Generate reasoning using Figure 7 prompt with chat template
            reasoning_prompt = config.restpg.reasoning_prompt_template.format(
                profile=profile_text,
                subject=item['x'],
                expected_output=item['y']
            )
            
            reasoning_summary = self.reasoning_generator.generate_reasoning(
                reasoning_prompt
            )

            # Create SFT format using Figure 8 prompt templates with chat template
            sft_input = config.restpg.sft_input_template.format(
                instruction=item['x'],
                context=profile_text
            )
            sft_output = config.restpg.sft_output_template.format(
                reasoning_summary=reasoning_summary,
                expected_output=item['y']
            )
            
            # Combine input and output for training
            combined_input = sft_input
            combined_output = sft_output
            
            reasoning_data.append({
                'x': combined_input,
                'y': combined_output,
                'P': item['P'],
                'user_id': item.get('user_id', 'unknown'),
                'reasoning_summary': reasoning_summary
            })
        
        # Save reasoning dataset
        save_jsonl_data(reasoning_data, output_path)
        
        print(f"Reasoning dataset saved to {output_path}")
        
    def supervised_fine_tuning(self, train_path: str, val_path: str, output_dir: str):
        """Stage 2: Supervised fine-tuning on reasoning data using LoRA"""
        print("Stage 2: Supervised fine-tuning on reasoning data with LoRA...")
        
        # Ensure model is loaded
        assert self.peft_model is not None, "LoRA model must be loaded before training"
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
        train_dataset = PersonalizedTextDataset(train_path, self.tokenizer)
        val_dataset = PersonalizedTextDataset(val_path, self.tokenizer)
        
        # Setup training arguments for LoRA
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=config.training.num_epochs,
            per_device_train_batch_size=config.training.batch_size,
            per_device_eval_batch_size=config.training.batch_size,
            warmup_steps=config.training.warmup_steps,
            weight_decay=config.training.weight_decay,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            learning_rate=config.training.learning_rate,
            fp16=True if device == "cuda" else False,  # Use fp16 only on GPU
            max_grad_norm=config.training.gradient_clip,
            dataloader_pin_memory=False,
            no_cuda=False if device == "cuda" else True,  # Force GPU usage
        )
        
        # Create trainer with LoRA model
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        # Train the LoRA model
        trainer.train()
        
        # Save the LoRA adapters
        trainer.save_model(f"{output_dir}/lora_model")
        print(f"LoRA training completed. Model saved to {output_dir}/lora_model")
        
    def expectation_step(self, data_path: str) -> List[Dict]:
        """Expectation step: Generate multiple outputs for each input"""
        print("Expectation step: Generating multiple outputs...")
        
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
    
    def maximization_step(self, generated_data: List[Dict], output_dir: str):
        """Maximization step: Train model on high-reward outputs using LoRA"""
        print("Maximization step: Training on high-reward outputs with LoRA...")
        
        # Create temporary dataset
        temp_data_path = f"{output_dir}/temp_generated_data.jsonl"
        with jsonlines.open(temp_data_path, 'w') as writer:
            for item in generated_data:
                writer.write(item)  # type: ignore
        
        # Create dataset
        dataset = PersonalizedTextDataset(temp_data_path, self.tokenizer)
        
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
            model=self.peft_model,
            args=training_args,
            train_dataset=dataset,
        )
        
        # Train the LoRA model
        trainer.train()
        
        print(f"Maximization step completed. LoRA weights updated.")
        
        # Clean up
        Path(temp_data_path).unlink()
        
    def train_rest_pg(self, train_path: str, val_path: str, output_dir: str):
        """Complete REST-PG training pipeline"""
        print("Starting REST-PG training pipeline...")
        
        # Stage 1: Generate reasoning dataset
        reasoning_data_path = f"{output_dir}/reasoning_data.jsonl"
        # self.generate_reasoning_dataset(train_path, reasoning_data_path)
        
        # Stage 2: Supervised fine-tuning on reasoning data
        self.supervised_fine_tuning(reasoning_data_path, val_path, f"{output_dir}/sft")
        
        # Stage 3: Expectation-Maximization self-training
        for iteration in range(config.restpg.num_iterations):
            print(f"\nIteration {iteration + 1}/{config.restpg.num_iterations}")
            
            # Expectation step
            generated_data = self.expectation_step(train_path)
            
            # Maximization step
            self.maximization_step(generated_data, output_dir)
            
            # Save checkpoint
            checkpoint_dir = f"{output_dir}/iteration_{iteration + 1}"
            self.model.save_pretrained(checkpoint_dir)
            self.tokenizer.save_pretrained(checkpoint_dir)
            
            print(f"Saved checkpoint: {checkpoint_dir}")
        
        print("REST-PG training completed!")
        
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
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8b", help="Model name to use")
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    
    # Create trainer
    trainer = RESTPGTrainer()
    
    # Load model
    # trainer.load_model(args.model_name)
    
    # Train
    trainer.train_rest_pg(args.train_path, args.val_path, args.output_dir)
    
    # Evaluate
    results = trainer.evaluate(args.test_path)
    
    # Save results
    with open(f"{args.output_dir}/evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main() 