#!/usr/bin/env python3
"""
Test script to verify JSON data loading and conversion
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_utils import load_json_data, save_jsonl_data

def test_json_loading():
    """Test JSON data loading and conversion"""
    
    # Test data path
    train_path = "data/raw/amazon_train.json"
    
    print("=== Testing JSON Data Loading ===")
    print()
    
    # Load and convert data
    data = load_json_data(train_path)
    
    print(f"Loaded {len(data)} training examples")
    print()
    
    # Show a few examples
    print("=== Sample Training Examples ===")
    print()
    
    for i, example in enumerate(data[:3]):
        print(f"Example {i+1}:")
        print(f"  User ID: {example['user_id']}")
        print(f"  Prompt: {example['x']}")
        print(f"  Expected Output: {example['y'][:100]}...")
        print(f"  Profile Length: {len(example['P'])} items")
        print(f"  Profile Sample: {example['P'][0][:100]}...")
        print()
    
    # Test saving to JSONL
    test_output_path = "test_output.jsonl"
    save_jsonl_data(data[:10], test_output_path)  # Save first 10 examples
    print(f"Saved 10 examples to {test_output_path}")
    
    # Clean up
    if os.path.exists(test_output_path):
        os.remove(test_output_path)
    
    print("JSON data loading test completed successfully!")

if __name__ == "__main__":
    test_json_loading() 