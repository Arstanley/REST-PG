# Dataset-Specific Model Organization

This document explains how to use the dataset-specific model saving feature to prevent model confusion when training on multiple datasets.

## Overview

When training REST-PG on multiple datasets, models are now automatically organized by dataset name to prevent confusion and ensure each dataset has its own isolated model checkpoints.

## Directory Structure

With dataset-specific organization, your output directory will look like this:

```
outputs/
├── dataset1/
│   ├── reasoning_data.jsonl
│   ├── sft/
│   │   └── lora_model/
│   ├── iteration_1/
│   ├── iteration_2/
│   ├── iteration_3/
│   └── evaluation_results.json
├── dataset2/
│   ├── reasoning_data.jsonl
│   ├── sft/
│   │   └── lora_model/
│   ├── iteration_1/
│   ├── iteration_2/
│   ├── iteration_3/
│   └── evaluation_results.json
└── multi_dataset_config.json
```

## Usage

### Single Dataset Training

For training on a single dataset, you can specify the dataset name:

```bash
python scripts/train.py \
    --train_path data/dataset1_train.jsonl \
    --val_path data/dataset1_val.jsonl \
    --test_path data/dataset1_test.jsonl \
    --output_dir outputs \
    --dataset_name dataset1
```

### Multi-Dataset Training

For training on multiple datasets, use the example script:

```bash
python example_multi_dataset_training.py \
    --datasets dataset1 dataset2 dataset3 \
    --train_paths data/dataset1_train.jsonl data/dataset2_train.jsonl data/dataset3_train.jsonl \
    --val_paths data/dataset1_val.jsonl data/dataset2_val.jsonl data/dataset3_val.jsonl \
    --test_paths data/dataset1_test.jsonl data/dataset2_test.jsonl data/dataset3_test.jsonl \
    --output_dir outputs
```

### Direct API Usage

You can also use the RESTPGTrainer directly:

```python
from src.rest_pg import RESTPGTrainer
from src.config import config

# Set dataset name
config.dataset_name = "my_dataset"

# Initialize trainer
trainer = RESTPGTrainer()

# Train with dataset-specific organization
trainer.train_rest_pg(
    train_path="data/train.jsonl",
    val_path="data/val.jsonl", 
    output_dir="outputs",
    dataset_name="my_dataset"
)
```

## Configuration

The dataset name can be set in several ways:

1. **Command line argument**: `--dataset_name my_dataset`
2. **Config file**: Set `config.dataset_name = "my_dataset"`
3. **Direct parameter**: Pass `dataset_name` to `train_rest_pg()`

## Benefits

1. **Model Isolation**: Each dataset gets its own model checkpoints
2. **Easy Comparison**: Compare models trained on different datasets
3. **Reproducibility**: Clear organization makes experiments reproducible
4. **Storage Efficiency**: Models are organized logically
5. **No Confusion**: Prevents accidentally loading wrong model checkpoints

## Model Loading

When loading models, the system automatically looks in the dataset-specific directory:

```python
# This will load from outputs/my_dataset/iteration_3/
trainer._load_most_recent_model("outputs", "my_dataset", iteration=3)

# This will load from outputs/my_dataset/sft/lora_model/
trainer._load_most_recent_model("outputs", "my_dataset")
```

## Migration from Old Format

If you have existing models in the old format (without dataset organization), you can migrate them by:

1. Renaming your existing output directory to include the dataset name
2. Or retraining with the new dataset-specific organization

## Example Workflow

```bash
# Train on first dataset
python scripts/train.py \
    --train_path data/amazon_train.jsonl \
    --val_path data/amazon_val.jsonl \
    --test_path data/amazon_test.jsonl \
    --dataset_name amazon \
    --output_dir outputs

# Train on second dataset  
python scripts/train.py \
    --train_path data/yelp_train.jsonl \
    --val_path data/yelp_val.jsonl \
    --test_path data/yelp_test.jsonl \
    --dataset_name yelp \
    --output_dir outputs

# Now you have:
# outputs/amazon/ - Amazon dataset models
# outputs/yelp/ - Yelp dataset models
```

This ensures that your Amazon and Yelp models are completely separate and won't interfere with each other. 