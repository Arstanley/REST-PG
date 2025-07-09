#!/usr/bin/env python3
"""
Test script for Hugging Face datasets integration
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from transformers import AutoTokenizer
from data_utils import create_dataset, create_reasoning_dataset, save_dataset_to_jsonl
from config import config

def test_dataset_creation():
    """Test creating datasets from JSON/JSONL files"""
    print("Testing dataset creation...")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test with sample data
    sample_data = [
        {
            'x': 'Write a review for product A',
            'y': 'This product is amazing!',
            'P': ['[5 stars] Great product - I love it!', '[4 stars] Good quality - worth the price'],
            'user_id': 'user1'
        },
        {
            'x': 'Write a review for product B', 
            'y': 'This product is okay.',
            'P': ['[3 stars] Average product - not bad', '[2 stars] Could be better - disappointed'],
            'user_id': 'user2'
        }
    ]
    
    # Save sample data to JSONL
    import jsonlines
    with jsonlines.open('test_data.jsonl', 'w') as writer:
        for item in sample_data:
            writer.write(item)
    
    try:
        # Test dataset creation
        dataset = create_dataset('test_data.jsonl', tokenizer)
        print(f"✓ Dataset created successfully with {len(dataset)} examples")
        print(f"✓ Dataset columns: {dataset.column_names}")
        
        # Test saving dataset
        save_dataset_to_jsonl(dataset, 'test_output.jsonl')
        print("✓ Dataset saved successfully")
        
        # Clean up
        os.remove('test_data.jsonl')
        os.remove('test_output.jsonl')
        
        print("✓ All tests passed!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_dataset_creation() 