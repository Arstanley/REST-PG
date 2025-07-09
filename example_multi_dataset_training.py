#!/usr/bin/env python3
"""
Example script demonstrating multi-dataset training with dataset-specific model organization
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
from pathlib import Path
import torch
import random
import numpy as np

from src.rest_pg import RESTPGTrainer
from src.config import config


def main():
    parser = argparse.ArgumentParser(description="Multi-dataset REST-PG training example")
    parser.add_argument("--datasets", type=str, nargs='+', required=True,
                       help="List of dataset names to train on")
    parser.add_argument("--train_paths", type=str, nargs='+', required=True,
                       help="List of training data paths (must match number of datasets)")
    parser.add_argument("--val_paths", type=str, nargs='+', required=True,
                       help="List of validation data paths (must match number of datasets)")
    parser.add_argument("--test_paths", type=str, nargs='+', required=True,
                       help="List of test data paths (must match number of datasets)")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Base output directory for all datasets")
    parser.add_argument("--model_name", type=str, default=None,
                       help="Model name to use (default: from config)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Validate input
    if len(args.datasets) != len(args.train_paths) or len(args.datasets) != len(args.val_paths) or len(args.datasets) != len(args.test_paths):
        raise ValueError("Number of datasets must match number of train/val/test paths")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    config.seed = args.seed
    
    # Create base output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save multi-dataset configuration
    config_dict = {
        'datasets': args.datasets,
        'train_paths': args.train_paths,
        'val_paths': args.val_paths,
        'test_paths': args.test_paths,
        'output_dir': args.output_dir,
        'model_name': args.model_name or config.model.model_name,
        'seed': args.seed
    }
    
    with open(f"{args.output_dir}/multi_dataset_config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print("Multi-dataset configuration:")
    print(json.dumps(config_dict, indent=2))
    print("\n" + "="*50)
    
    # Train on each dataset
    for i, dataset_name in enumerate(args.datasets):
        print(f"\n{'='*20} Training on dataset: {dataset_name} {'='*20}")
        
        # Set dataset name in config
        config.dataset_name = dataset_name
        
        # Initialize trainer
        trainer = RESTPGTrainer()
        
        # Train REST-PG on this dataset
        print(f"Starting REST-PG training for dataset: {dataset_name}")
        trainer.train_rest_pg(
            args.train_paths[i], 
            args.val_paths[i], 
            args.output_dir, 
            dataset_name
        )
        
        # Evaluate
        print(f"Evaluating model for dataset: {dataset_name}")
        results = trainer.evaluate(args.test_paths[i])
        
        # Save results
        dataset_output_dir = f"{args.output_dir}/{dataset_name}"
        with open(f"{dataset_output_dir}/evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Completed training for dataset: {dataset_name}")
        print(f"Results saved to: {dataset_output_dir}/evaluation_results.json")
    
    print(f"\n{'='*50}")
    print("Multi-dataset training completed!")
    print(f"All models and results saved to: {args.output_dir}/")
    print("Directory structure:")
    for dataset_name in args.datasets:
        dataset_dir = f"{args.output_dir}/{dataset_name}"
        print(f"  {dataset_dir}/")
        print(f"    ├── reasoning_data.jsonl")
        print(f"    ├── sft/lora_model/")
        print(f"    ├── iteration_1/")
        print(f"    ├── iteration_2/")
        print(f"    ├── ...")
        print(f"    └── evaluation_results.json")


if __name__ == "__main__":
    main() 