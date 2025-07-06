#!/usr/bin/env python3
"""
Test script to verify GPU usage
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.rest_pg import RESTPGTrainer

def test_gpu_usage():
    """Test GPU usage and model loading"""
    
    print("=== GPU Availability Check ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Current device: {torch.cuda.current_device()}")
    print()
    
    print("=== Testing Model Loading ===")
    try:
        # Create trainer (this will load models)
        trainer = RESTPGTrainer()
        
        # Check if models are on GPU
        if hasattr(trainer, 'model') and trainer.model is not None:
            model_device = next(trainer.model.parameters()).device
            print(f"Main model device: {model_device}")
        
        if hasattr(trainer, 'peft_model') and trainer.peft_model is not None:
            peft_device = next(trainer.peft_model.parameters()).device
            print(f"PEFT model device: {peft_device}")
        
        print("Model loading test completed successfully!")
        
    except Exception as e:
        print(f"Error during model loading: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gpu_usage() 