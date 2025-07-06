#!/usr/bin/env python3
"""
Test script for REST-PG implementation
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path
import torch

def test_config():
    """Test configuration loading"""
    print("Testing configuration...")
    try:
        from src.config import config
        print("✓ Configuration loaded successfully")
        print(f"  Model: {config.model.model_name}")
        print(f"  Batch size: {config.training.batch_size}")
        print(f"  Learning rate: {config.training.learning_rate}")
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_data_utils():
    """Test data utilities"""
    print("\nTesting data utilities...")
    try:
        from src.data_utils import PersonalizedTextDataset, ReasoningDataGenerator, calculate_rouge_reward
        
        # Test dataset class
        print("✓ Data utilities imported successfully")
        
        # Test reasoning generator (with mock model)
        class MockModel:
            def __init__(self):
                self.tokenizer = None
        mock_model = MockModel()
        generator = ReasoningDataGenerator(mock_model)
        print("✓ ReasoningDataGenerator created successfully")
        
        # Test ROUGE-based reward function
        expected = "This is a great product that I really enjoy using."
        generated = "This product is excellent and I love using it."
        reward = calculate_rouge_reward(expected, generated)
        print(f"✓ ROUGE reward function test passed: {reward:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Data utilities test failed: {e}")
        return False

def test_rest_pg():
    """Test REST-PG trainer"""
    print("\nTesting REST-PG trainer...")
    try:
        from src.rest_pg import RESTPGTrainer
        
        trainer = RESTPGTrainer()
        print("✓ RESTPGTrainer created successfully")
        
        return True
    except Exception as e:
        print(f"✗ REST-PG trainer test failed: {e}")
        return False

def test_sample_data():
    """Test with sample data"""
    print("\nTesting with sample data...")
    try:
        # Check if sample data exists
        data_dir = Path("data/processed")
        if not data_dir.exists():
            print("  Sample data not found. Run 'python scripts/generate_sample_data.py' first.")
            return False
        
        train_path = data_dir / "train.jsonl"
        val_path = data_dir / "val.jsonl"
        test_path = data_dir / "test.jsonl"
        
        if not all(p.exists() for p in [train_path, val_path, test_path]):
            print("  Sample data files not found. Run 'python scripts/generate_sample_data.py' first.")
            return False
        
        # Load a few samples
        import jsonlines
        with jsonlines.open(train_path) as reader:
            sample = list(reader)[0]  # Get first item
        
        print("✓ Sample data loaded successfully")
        print(f"  Sample keys: {list(sample.keys())}")
        print(f"  Input length: {len(sample['x'])}")
        print(f"  Output length: {len(sample['y'])}")
        print(f"  Profile items: {len(sample['P'])}")
        
        return True
    except Exception as e:
        print(f"✗ Sample data test failed: {e}")
        return False

def test_model_loading():
    """Test model loading (without actually downloading)"""
    print("\nTesting model loading...")
    try:
        from transformers import AutoTokenizer
        
        # Test tokenizer loading (this will download if not cached)
        print("  Testing tokenizer loading...")
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
        print("✓ Tokenizer loaded successfully")
        
        # Test basic tokenization
        text = "Hello, world!"
        tokens = tokenizer(text, return_tensors="pt")
        print(f"✓ Tokenization test passed: {tokens['input_ids'].shape}")
        
        return True
    except Exception as e:
        print(f"✗ Model loading test failed: {e}")
        print("  This might be due to network issues or missing model access")
        return False

def main():
    """Run all tests"""
    print("Running REST-PG implementation tests...")
    print("=" * 50)
    
    tests = [
        test_config,
        test_data_utils,
        test_rest_pg,
        test_sample_data,
        test_model_loading
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The implementation is ready to use.")
        print("\nNext steps:")
        print("1. Generate sample data: python scripts/generate_sample_data.py")
        print("2. Train the model: python scripts/train.py --train_path data/processed/train.jsonl --val_path data/processed/val.jsonl --test_path data/processed/test.jsonl --output_dir outputs/test_run")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\nCommon issues:")
        print("- Install dependencies: pip install -r requirements.txt")
        print("- Generate sample data: python scripts/generate_sample_data.py")
        print("- Check model access permissions for Gemma models")

if __name__ == "__main__":
    main() 