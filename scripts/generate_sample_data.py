#!/usr/bin/env python3
"""
Generate sample data for testing REST-PG implementation
"""
import json
import jsonlines
import random
from pathlib import Path
from typing import List, Dict

def generate_sample_user_data(num_users: int = 10, samples_per_user: int = 5) -> List[Dict]:
    """Generate sample user data for testing"""
    
    # Sample user profiles (writing styles, interests, etc.)
    user_profiles = [
        {
            "id": f"user_{i}",
            "profile": [
                {"rating": 5, "title": "Great product", "text": "I absolutely love this product! It exceeded all my expectations and I would definitely recommend it to anyone looking for quality."},
                {"rating": 4, "title": "Good value", "text": "This is a solid product for the price. Good quality and meets my needs well."},
                {"rating": 3, "title": "Average experience", "text": "It's okay, nothing special but gets the job done. Might look for alternatives next time."},
                {"rating": 5, "title": "Excellent service", "text": "Amazing customer service and the product works perfectly. Very satisfied with my purchase."},
                {"rating": 2, "title": "Disappointed", "text": "Not what I expected. The quality is poor and I wouldn't buy again."}
            ]
        }
        for i in range(num_users)
    ]
    
    # Sample product prompts
    product_prompts = [
        "Write a review for a wireless headphones",
        "Review a smart home device",
        "Write about a fitness tracker",
        "Review a kitchen appliance",
        "Write about a mobile app",
        "Review a book",
        "Write about a restaurant experience",
        "Review a travel destination",
        "Write about a software tool",
        "Review a clothing item"
    ]
    
    # Generate training data
    training_data = []
    
    for user in user_profiles:
        user_id = user["id"]
        profile = user["profile"]
        
        # Create profile strings
        profile_strings = [f"[{item['rating']} stars] {item['title']} - {item['text']}" for item in profile]
        
        # Generate samples for this user
        for _ in range(samples_per_user):
            # Select a random product prompt
            prompt = random.choice(product_prompts)
            
            # Generate a personalized response based on user's writing style
            response = generate_personalized_response(prompt, profile, user_id)
            
            training_data.append({
                "x": prompt,
                "y": response,
                "P": profile_strings,
                "user_id": user_id
            })
    
    return training_data

def generate_personalized_response(prompt: str, profile: List[Dict], user_id: str) -> str:
    """Generate a personalized response based on user's writing style"""
    
    # Analyze user's writing style from profile
    avg_rating = sum(item["rating"] for item in profile) / len(profile)
    is_positive = avg_rating >= 4
    is_detailed = any(len(item["text"]) > 100 for item in profile)
    
    # Generate response based on style
    if "headphones" in prompt.lower():
        if is_positive:
            response = "These wireless headphones are absolutely fantastic! The sound quality is exceptional and the battery life is impressive. I love how comfortable they are for long listening sessions."
        else:
            response = "The wireless headphones are okay but nothing special. The sound quality is decent but I expected better for the price. They're comfortable enough though."
    elif "smart home" in prompt.lower():
        if is_positive:
            response = "This smart home device has completely transformed my living space! It's incredibly easy to set up and the automation features work flawlessly. Highly recommend!"
        else:
            response = "The smart home device is functional but has some connectivity issues. Setup was a bit complicated and it doesn't always respond reliably."
    elif "fitness" in prompt.lower():
        if is_positive:
            response = "This fitness tracker is amazing! It accurately tracks all my activities and the app provides great insights. The battery life is outstanding and it's very comfortable to wear."
        else:
            response = "The fitness tracker works but the accuracy could be better. The app is basic and the battery doesn't last as long as advertised. It's wearable but not great."
    else:
        if is_positive:
            response = "This product exceeded my expectations! The quality is excellent and it performs exactly as advertised. I'm very satisfied with my purchase and would definitely recommend it."
        else:
            response = "This product is average at best. It works but doesn't stand out in any way. The quality is acceptable but I expected more for the price."
    
    # Add more detail if user tends to write detailed reviews
    if is_detailed:
        response += " The packaging was well-designed and the instructions were clear. Overall, a solid choice for anyone looking for this type of product."
    
    return response

def split_data(data: List[Dict], train_ratio: float = 0.7, val_ratio: float = 0.15) -> tuple:
    """Split data into train, validation, and test sets"""
    random.shuffle(data)
    
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    return train_data, val_data, test_data

def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Generate sample data
    print("Generating sample data...")
    all_data = generate_sample_user_data(num_users=20, samples_per_user=8)
    
    # Split data
    train_data, val_data, test_data = split_data(all_data)
    
    # Create output directory
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save data
    print(f"Saving {len(train_data)} training samples...")
    with jsonlines.open(output_dir / "train.jsonl", 'w') as writer:
        for item in train_data:
            writer.write(item)
    
    print(f"Saving {len(val_data)} validation samples...")
    with jsonlines.open(output_dir / "val.jsonl", 'w') as writer:
        for item in val_data:
            writer.write(item)
    
    print(f"Saving {len(test_data)} test samples...")
    with jsonlines.open(output_dir / "test.jsonl", 'w') as writer:
        for item in test_data:
            writer.write(item)
    
    print(f"\nSample data generated successfully!")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Total samples: {len(all_data)}")
    print(f"Data saved to: {output_dir}")

if __name__ == "__main__":
    main() 