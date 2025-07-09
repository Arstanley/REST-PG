#!/usr/bin/env python3
"""
Training script for REST-PG implementation
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path
import torch
import random
import numpy as np

from src.rest_pg import RESTPGTrainer
from src.config import config


def main():
    parser = argparse.ArgumentParser(description="Train REST-PG model")
    parser.add_argument("--train_path", type=str, required=True, 
                       help="Path to training data JSONL file")
    parser.add_argument("--val_path", type=str, required=True,
                       help="Path to validation data JSONL file") 
    parser.add_argument("--test_path", type=str, required=True,
                       help="Path to test data JSONL file")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Output directory for models and results")
    parser.add_argument("--dataset_name", type=str, default="default",
                       help="Dataset name for organizing models (default: default)")
    parser.add_argument("--model_name", type=str, default=None,
                       help="Model name to use (default: from config)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    config.seed = args.seed
    
    # Set dataset name
    config.dataset_name = args.dataset_name
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_dict = {
        'model': {
            'model_name': args.model_name or config.model.model_name,
            'max_input_length': config.model.max_input_length,
            'max_output_length': config.model.max_output_length,
            'temperature': config.model.temperature,
            'inference_temperature': config.model.inference_temperature
        },
        'training': {
            'learning_rate': config.training.learning_rate,
            'batch_size': config.training.batch_size,
            'num_epochs': config.training.num_epochs,
            'warmup_steps': config.training.warmup_steps,
            'weight_decay': config.training.weight_decay,
            'gradient_clip': config.training.gradient_clip
        },
        'restpg': {
            'num_iterations': config.restpg.num_iterations,
            'exploration_budget': config.restpg.exploration_budget,
            'reward_threshold': config.restpg.reward_threshold,
            'max_outputs_per_input': config.restpg.max_outputs_per_input,
            'retrieval_top_k': config.restpg.retrieval_top_k
        },
        'seed': args.seed
    }
    
    with open(f"{args.output_dir}/config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print("Configuration:")
    print(json.dumps(config_dict, indent=2))
    print("\n" + "="*50)
    
    # Initialize trainer
    trainer = RESTPGTrainer()
    
    # Load model
    model_name = args.model_name or config.model.model_name
    print(f"Loading model: {model_name}")
    # trainer.load_model(model_name)
    
    # Train REST-PG
    print("Starting REST-PG training...")
    trainer.train_rest_pg(args.train_path, args.val_path, args.output_dir, args.dataset_name)
    
    # Evaluate
    print("Evaluating model...")
    results = trainer.evaluate(args.test_path)
    
    # Save results
    with open(f"{args.output_dir}/evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTraining completed! Results saved to {args.output_dir}/")
    print(f"Final evaluation results: {results}")


if __name__ == "__main__":
    main() 