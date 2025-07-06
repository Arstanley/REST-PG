#!/usr/bin/env python3
"""
Test script to verify the prompt templates from Figure 7 and Figure 8
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rest_pg import RESTPGTrainer

def test_prompt_templates():
    """Test the prompt templates from Figure 7 and Figure 8"""
    
    # Create trainer instance
    trainer = RESTPGTrainer()
    
    # Test data
    test_item = {
        'x': 'Write a review for a wireless headphones',
        'y': 'These wireless headphones are absolutely fantastic! The sound quality is exceptional and the battery life is impressive. I love how comfortable they are for long listening sessions.',
        'P': [
            '[5 stars] Great product - I absolutely love this product! It exceeded all my expectations and I would definitely recommend it to anyone looking for quality.',
            '[4 stars] Good value - This is a solid product for the price. Good quality and meets my needs well.',
            '[5 stars] Excellent service - Amazing customer service and the product works perfectly. Very satisfied with my purchase.'
        ],
        'user_id': 'test_user_1'
    }
    
    print("=== Testing Figure 7 Prompt Template (Reasoning Generation) ===")
    print()
    
    # Test Figure 7 prompt
    profile_text = " | ".join(test_item['P'])
    reasoning_prompt = trainer.reasoning_prompt_template.format(
        profile=profile_text,
        subject=test_item['x'],
        expected_output=test_item['y']
    )
    
    print("Reasoning Prompt:")
    print(reasoning_prompt)
    print()
    print("=" * 80)
    print()
    
    print("=== Testing Figure 8 Prompt Template (SFT Training) ===")
    print()
    
    # Mock reasoning summary
    mock_reasoning = """• The user tends to write enthusiastic, positive reviews with exclamation marks
• They frequently use words like "absolutely", "fantastic", "exceptional", "amazing"
• They often mention specific features and benefits of products
• They express personal satisfaction and recommend products to others
• They use detailed descriptions and emotional language"""
    
    # Test Figure 8 prompt
    sft_prompt = trainer.sft_prompt_template.format(
        instruction=test_item['x'],
        context=profile_text,
        reasoning_summary=mock_reasoning,
        expected_output=test_item['y']
    )
    
    print("SFT Prompt:")
    print(sft_prompt)
    print()
    print("=" * 80)
    print()
    
    print("=== Testing Input Formatting ===")
    print()
    
    # Test the _format_input method
    formatted_input = trainer._format_input(test_item['x'], test_item['P'])
    print("Formatted Input for Generation:")
    print(formatted_input)
    print()
    
    print("All prompt templates are working correctly!")

if __name__ == "__main__":
    test_prompt_templates() 