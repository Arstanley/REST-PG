"""
Configuration settings for REST-PG implementation
"""
from dataclasses import dataclass
from typing import Optional, List
import os

@dataclass
class ModelConfig:
    """Model configuration settings"""
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"  # Base model to use
    max_input_length: int = 5120
    max_output_length: int = 1536
    temperature: float = 0.7  # For exploration during training
    inference_temperature: float = 0.1  # For deterministic inference
    top_p: float = 0.9
    do_sample: bool = True

@dataclass
class TrainingConfig:
    """Training configuration settings"""
    learning_rate: float = 5e-6
    batch_size: int = 64
    num_epochs: int = 3
    warmup_steps: int = 250
    max_steps: int = 10000
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    lr_scheduler: str = "linear"
    lr_decay: float = 0.1

@dataclass
class RESTPGConfig:
    """REST-PG specific configuration"""
    # Self-training parameters
    num_iterations: int = 3
    exploration_budget: int = 32  # Number of outputs to generate per input
    reward_threshold: float = 0.3  # Threshold for selecting high-reward outputs (adjusted for ROUGE scores)
    max_outputs_per_input: int = 10  # Maximum outputs to retain per input
    
    # Reasoning generation
    reasoning_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"  # Model for generating reasoning paths
    reasoning_prompt_template: str = """# Your role:
You are a professional writing assistant whose task is to summarize the writing style of a user from the profile, which is past documents written by that user. The extracted writing style summary should contain the unique features of users writing style and preferences from the profile that are similar to the expected output.

# Your task:
Your task is to summarize the user writing style from the profile considering the expected output. From the profile, you may infer the user's interests, preference, familiarity on various topics, etc. While inferring the user's interests, you can make reasonable guesses, e.g. people who are interested in topic A are also likely to be interested in topic B or if they write a sentence in a specific writing style on topic A it is likely they write it with the same style on topic B. As a concrete example, if a user writes "I am interested in action movies" in its past document, this is relevant to "I like to go to cinema" in the expected output. Another example would be if a person prefers specific words or phrases in their writing or using a specific grammar. You can also mention such words that they often use in your summary.

# Your input:
- profile: the past documents written by the same person that are separated with | symbol.
- subject: the subject for the expected output
- expected output: the expected output written by the same person as the past documents.

# Your output:
a list of bullet points and explanations describing writing style of the user. Also, make sure that you only talk about information from the profile while considering the expected output in writing style summarization. You cannot directly copy or mention anything about the expected output. The expected output is only used to determine the writing style of the user and how profile can affect the expected output.

## profile:
{profile}

## subject:
{subject}

## expected output:
{expected_output}

## Your output:"""

    sft_input_template: str = """Input: The following context is written by a specific user. Please use the following context to generate a personalized response to the instruction. Your response should follow the same pattern in terms of preferences and writing style in the provided context. You should first summarize the writing style of the user based on the provided context. Then, you should use the summarized writing style to generate a response to the instruction.

instruction: {instruction}
context: {context}

answer:

Output:
To summarize the writing style of the user, we can consider the following aspects:"""
    
    sft_output_template: str = """{reasoning_summary}

Now, considering the style summary, we can generate the final answer: {expected_output}"""
    
    # RAG settings
    retrieval_top_k: int = 5
    retrieval_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

@dataclass
class DataConfig:
    """Data configuration settings"""
    train_data_path: str = "data/raw/amazon_train.json"
    val_data_path: str = "data/raw/amazon_dev.json"
    test_data_path: str = "data/raw/amazon_test.json"
    cache_dir: str = "cache"
    max_val_samples: int = 1024  # For faster validation during training

@dataclass
class Config:
    """Main configuration class"""
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    restpg: RESTPGConfig = RESTPGConfig()
    data: DataConfig = DataConfig()
    
    # General settings
    seed: int = 42
    device: str = "auto"  # Will be set to cuda if available
    output_dir: str = "outputs"
    log_dir: str = "logs"
    save_steps: int = 1000
    eval_steps: int = 1000
    logging_steps: int = 100
    
    # Wandb settings
    use_wandb: bool = False
    wandb_project: str = "rest-pg"
    wandb_entity: Optional[str] = None
    
    def __post_init__(self):
        """Set up device and create directories"""
        import torch
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Force GPU usage if available
        if torch.cuda.is_available():
            self.device = "cuda"
            print(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            self.device = "cpu"
            print("GPU not available, using CPU")
        
        # Create necessary directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.data.cache_dir, exist_ok=True)

# Global config instance
config = Config() 