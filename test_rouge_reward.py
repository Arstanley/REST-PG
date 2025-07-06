#!/usr/bin/env python3
"""
Test script for ROUGE-based reward function
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_utils import calculate_rouge_reward

def test_rouge_reward():
    """Test the ROUGE-based reward function"""
    print("Testing ROUGE-based reward function...")
    
    # Test cases
    test_cases = [
        {
            "expected": "This is a great product that I really enjoy using.",
            "generated": "This product is excellent and I love using it.",
            "description": "Similar meaning, different words"
        },
        {
            "expected": "The movie was fantastic and I highly recommend it.",
            "generated": "The movie was fantastic and I highly recommend it.",
            "description": "Exact match"
        },
        {
            "expected": "I love this restaurant, the food is amazing.",
            "generated": "The weather is nice today.",
            "description": "Completely different content"
        },
        {
            "expected": "This product exceeded my expectations. The quality is outstanding and the customer service was excellent.",
            "generated": "This product is really good. I like the quality and the service was nice.",
            "description": "Similar content, different style"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        expected = test_case["expected"]
        generated = test_case["generated"]
        description = test_case["description"]
        
        reward = calculate_rouge_reward(expected, generated)
        
        print(f"Test {i} ({description}):")
        print(f"  Expected: {expected}")
        print(f"  Generated: {generated}")
        print(f"  ROUGE Reward: {reward:.4f}")
        print()
    
    print("ROUGE reward function test completed!")

if __name__ == "__main__":
    test_rouge_reward() 