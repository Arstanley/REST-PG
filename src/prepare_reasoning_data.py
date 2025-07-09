#!/usr/bin/env python3
"""
Script to prepare existing reasoning data for REST-PG training
"""
import sys
import os
import json
import jsonlines
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_utils import validate_reasoning_data, save_dataset_to_jsonl
from datasets import Dataset, load_dataset

def convert_reasoning_data_to_rest_pg_format(input_path: str, output_path: str):
    """Convert existing reasoning data to REST-PG format"""
    print(f"Converting reasoning data from {input_path} to {output_path}")
    
    try:
        # Load the input data
        if input_path.endswith('.json'):
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            # Assume JSONL format
            data = []
            with jsonlines.open(input_path) as reader:
                for item in reader:
                    data.append(item)
        
        # Convert to REST-PG format
        rest_pg_data = []
        
        for item in data:
            # Expected format: should have input/prompt and output/response fields
            # Adjust these field names based on your actual data format
            if 'input' in item and 'output' in item:
                rest_pg_data.append({
                    'x': item['input'],
                    'y': item['output'],
                    'P': item.get('profile', []),
                    'user_id': item.get('user_id', 'unknown')
                })
            elif 'prompt' in item and 'response' in item:
                rest_pg_data.append({
                    'x': item['prompt'],
                    'y': item['response'],
                    'P': item.get('profile', []),
                    'user_id': item.get('user_id', 'unknown')
                })
            elif 'x' in item and 'y' in item:
                # Already in REST-PG format
                rest_pg_data.append(item)
            else:
                print(f"Warning: Skipping item with unknown format: {list(item.keys())}")
                continue
        
        # Create dataset and save
        dataset = Dataset.from_list(rest_pg_data)
        save_dataset_to_jsonl(dataset, output_path)
        
        print(f"✓ Successfully converted {len(rest_pg_data)} examples")
        return True
        
    except Exception as e:
        print(f"✗ Error converting data: {e}")
        return False

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare existing reasoning data for REST-PG training")
    parser.add_argument("--input_path", required=True, help="Path to existing reasoning data (JSON/JSONL)")
    parser.add_argument("--output_path", required=True, help="Output path for REST-PG formatted data")
    parser.add_argument("--validate_only", action="store_true", help="Only validate existing data without converting")
    
    args = parser.parse_args()
    
    if args.validate_only:
        # Just validate the existing data
        if validate_reasoning_data(args.input_path):
            print("✓ Data validation passed!")
        else:
            print("✗ Data validation failed!")
            sys.exit(1)
    else:
        # Convert the data
        if convert_reasoning_data_to_rest_pg_format(args.input_path, args.output_path):
            print("✓ Data conversion completed!")
            
            # Validate the converted data
            if validate_reasoning_data(args.output_path):
                print("✓ Converted data validation passed!")
            else:
                print("⚠ Converted data validation failed!")
        else:
            print("✗ Data conversion failed!")
            sys.exit(1)

if __name__ == "__main__":
    main() 