import json
import argparse
from pathlib import Path
from typing import List, Dict


def load_data(input_path: str) -> List[Dict]:
    with open(input_path, 'r') as f:
        return json.load(f)


def construct_prompt(product_id: str, rating: int) -> str:
    return f"Write a product review for item {product_id} with a rating of {rating} stars."


def format_profile_entry(entry: Dict) -> str:
    return f"[{entry['rating']} stars] {entry['title']} - {entry['text']}"


def build_triplets(data: List[Dict]) -> List[Dict]:
    triplets = []
    for user in data:
        profile = user["profile"]
        if len(profile) < 2:
            continue  # skip users with only one review

        for i, review in enumerate(profile):
            x = construct_prompt(review["pid"], review["rating"])
            y = review["text"]
            P = [format_profile_entry(r) for j, r in enumerate(profile) if j != i]

            triplets.append({
                "x": x,
                "y": y,
                "P": P,
                "user_id": user["id"]
            })
    return triplets


def save_jsonl(data: List[Dict], output_path: str):
    with open(output_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def main(args):
    raw_data = load_data(args.input)
    triplets = build_triplets(raw_data)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(triplets, args.output)
    print(f"Saved {len(triplets)} examples to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/amazon_train.json", type=str, required=True, help="Path to raw input JSON file")
    parser.add_argument("--output", default="data/processed/amazon_train.jsonl", type=str, required=True, help="Path to save processed JSONL file")
    args = parser.parse_args()
    main(args)