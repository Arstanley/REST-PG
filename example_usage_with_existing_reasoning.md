# Using REST-PG with Existing Reasoning Data

This guide shows how to use the REST-PG training pipeline when you already have reasoning data generated.

## Scenario 1: Your reasoning data is already in REST-PG format

If your reasoning data already has the correct format with `x` (input) and `y` (output) fields:

```bash
# Validate your existing reasoning data
python prepare_reasoning_data.py --input_path your_reasoning_data.jsonl --validate_only

# Run training with existing reasoning data
python src/rest_pg.py \
    --train_path your_train_data.json \
    --val_path your_val_data.json \
    --output_dir outputs \
    --skip_reasoning_generation
```

## Scenario 2: Your reasoning data needs format conversion

If your reasoning data has different field names (e.g., `input`/`output` or `prompt`/`response`):

```bash
# Convert your reasoning data to REST-PG format
python prepare_reasoning_data.py \
    --input_path your_reasoning_data.jsonl \
    --output_path outputs/reasoning_data.jsonl

# Run training with converted reasoning data
python src/rest_pg.py \
    --train_path your_train_data.json \
    --val_path your_val_data.json \
    --output_dir outputs \
    --skip_reasoning_generation
```

## Expected Data Format

Your reasoning data should be in JSONL format with the following structure:

```json
{
    "x": "Write a review for product X",
    "y": "Reasoning: Based on the user's profile, they prefer detailed reviews with specific features mentioned. They tend to use technical language and focus on value for money.\n\nOutput: This product offers excellent value with its advanced features. The build quality is solid and it performs well for its price point. I particularly appreciate the attention to detail in the design.",
    "P": ["[5 stars] Great product - I love it!", "[4 stars] Good quality - worth the price"],
    "user_id": "user123"
}
```

## Key Changes in the Updated Pipeline

1. **Skip Reasoning Generation**: Use `--skip_reasoning_generation` flag
2. **Data Validation**: The pipeline automatically validates existing reasoning data
3. **Error Handling**: Clear error messages if data format is incorrect
4. **Flexible Input**: Supports both JSON and JSONL formats

## Benefits

- **Faster Training**: Skip the time-consuming reasoning generation step
- **Reuse Existing Work**: Leverage previously generated reasoning data
- **Better Control**: Use your own reasoning generation methods if needed
- **Validation**: Ensure data quality before training starts

## Troubleshooting

If you get validation errors:

1. Check that your data has `x` and `y` fields
2. Ensure the data is not empty
3. Verify the format matches the expected structure
4. Use the validation script to debug issues:

```bash
python prepare_reasoning_data.py --input_path your_data.jsonl --validate_only
``` 