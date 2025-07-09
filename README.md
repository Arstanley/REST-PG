# REST-PG: Reasoning-Enhanced Self-Training for Personalized Text Generation

This repository contains an implementation of the REST-PG algorithm from the paper ["Reasoning-Enhanced Self-Training for Long-Form Personalized Text Generation"](https://arxiv.org/abs/2501.04167) by Salemi et al.

## Overview

REST-PG is a multi-stage framework that combines reasoning capabilities with self-training to improve personalized text generation in large language models (LLMs). The algorithm consists of three main stages:

1. **Reasoning Enhancement**: Generate reasoning paths that summarize user preferences from personalized context
2. **Supervised Fine-tuning**: Train the model on reasoning-enhanced data
3. **Expectation-Maximization Self-Training**: Iteratively improve reasoning paths using reinforcement learning

## Key Features

- **Reasoning Generation**: Uses a larger LLM (Gemma 7B) to generate reasoning paths
- **Self-Training**: Implements Expectation-Maximization Reinforced Self-Training
- **ROUGE-based Evaluation**: Uses ROUGE-1 and ROUGE-L scores for reward calculation
- **RAG Integration**: Retrieval-augmented generation for personalized context
- **Modular Design**: Clean, modular implementation for easy experimentation

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd REST-PG
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables (optional):
```bash
cp .env.example .env
# Edit .env with your settings
```

## Data Format

The implementation expects data in JSONL format with the following structure:

```json
{
    "x": "input prompt",
    "y": "expected output", 
    "P": ["context_item_1", "context_item_2", ...],
    "user_id": "user_identifier"
}
```

Where:
- `x`: The input prompt for text generation
- `y`: The expected personalized output
- `P`: List of personalized context items (user profile, past writings, etc.)
- `user_id`: Unique identifier for the user

## Usage

### 1. Data Preprocessing

First, preprocess your data using the provided script:

```bash
python scripts/preprocess.py \
    --input data/raw/your_data.json \
    --output data/processed/your_data.jsonl
```

### 2. Training

Train the REST-PG model:

```bash
python scripts/train.py \
    --train_path data/processed/train.jsonl \
    --val_path data/processed/val.jsonl \
    --test_path data/processed/test.jsonl \
    --output_dir outputs/experiment_1 \
    --model_name google/gemma-2b \
    --seed 42
```

### 3. Evaluation

The training script automatically evaluates the model and saves results to `outputs/experiment_1/evaluation_results.json`.

### 4. Multi-Dataset Training

For training on multiple datasets, the system now supports dataset-specific model organization to prevent model confusion:

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
```

This creates separate model directories:
- `outputs/amazon/` - Amazon dataset models
- `outputs/yelp/` - Yelp dataset models

For batch training on multiple datasets, use the example script:

```bash
python example_multi_dataset_training.py \
    --datasets amazon yelp \
    --train_paths data/amazon_train.jsonl data/yelp_train.jsonl \
    --val_paths data/amazon_val.jsonl data/yelp_val.jsonl \
    --test_paths data/amazon_test.jsonl data/yelp_test.jsonl \
    --output_dir outputs
```

See `DATASET_ORGANIZATION.md` for detailed documentation on this feature.

## Configuration

The implementation uses a comprehensive configuration system in `src/config.py`. Key parameters include:

### Model Configuration
- `model_name`: Base model to use (default: "google/gemma-2b")
- `max_input_length`: Maximum input sequence length (default: 5120)
- `max_output_length`: Maximum output sequence length (default: 1536)
- `temperature`: Generation temperature for exploration (default: 0.7)

### Training Configuration
- `learning_rate`: Learning rate for optimization (default: 5e-6)
- `batch_size`: Training batch size (default: 64)
- `num_epochs`: Number of training epochs (default: 3)
- `warmup_steps`: Number of warmup steps (default: 250)

### REST-PG Configuration
- `num_iterations`: Number of EM iterations (default: 3)
- `exploration_budget`: Number of outputs to generate per input (default: 32)
- `reward_threshold`: Threshold for selecting high-reward outputs (default: 0.3)
- `max_outputs_per_input`: Maximum outputs to retain per input (default: 10)

## Architecture

```
src/
├── config.py          # Configuration management
├── data_utils.py      # Data loading and processing utilities
├── rest_pg.py         # Main REST-PG implementation
└── __init__.py

scripts/
├── preprocess.py      # Data preprocessing script
└── train.py          # Training script

data/
├── raw/              # Raw data files
└── processed/        # Processed JSONL files

outputs/              # Training outputs and checkpoints
```

## Key Components

### 1. ReasoningDataGenerator
Generates reasoning paths using a larger LLM (Gemma 7B) by analyzing user context, input prompts, and expected outputs.

### 2. ROUGE-based Reward Function
Calculates rewards using the average of ROUGE-1 and ROUGE-L F1 scores to evaluate the quality of generated outputs.

### 3. RESTPGTrainer
Main trainer class that orchestrates the three-stage training process:
- Stage 1: Reasoning dataset generation
- Stage 2: Supervised fine-tuning
- Stage 3: Expectation-Maximization self-training

### 4. PersonalizedTextDataset
Custom dataset class that handles personalized text generation data with RAG integration.

## Results

The implementation follows the paper's experimental setup and should achieve:
- 14.5% average improvement over supervised fine-tuning
- 6.5% improvement over self-training without reasoning
- Significant improvements across all LongLaMP benchmark tasks

## Limitations

1. **Computational Requirements**: Requires significant GPU memory for large models
2. **Evaluation Challenges**: Personalized text generation evaluation is inherently subjective
3. **Latency**: Reasoning enhancement increases output length and decoding time

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{salemi2025reasoning,
  title={Reasoning-Enhanced Self-Training for Long-Form Personalized Text Generation},
  author={Salemi, Alireza and Li, Cheng and Zhang, Mingyang and Mei, Qiaozhu and Kong, Weize and Chen, Tao and Li, Zhuowan and Bendersky, Michael and Zamani, Hamed},
  journal={arXiv preprint arXiv:2501.04167},
  year={2025}
}
```

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.