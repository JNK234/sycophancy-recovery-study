# ABOUTME: Training hyperparameters for SFT and DPO
# ABOUTME: Config classes for training runs

from dataclasses import dataclass


@dataclass
class SFTConfig:
    model_name: str = "Qwen/Qwen3-8B"
    max_seq_length: int = 2048
    lora_r: int = 16
    lora_alpha: int = 16
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation: int = 4
    learning_rate: float = 2e-4
    output_dir: str = "outputs/sft"


@dataclass
class DPOConfig:
    model_name: str = "outputs/sft"  # Start from sycophantic model
    beta: float = 0.1
    epochs: int = 1
    batch_size: int = 2
    gradient_accumulation: int = 8
    learning_rate: float = 5e-5
    output_dir: str = "outputs/dpo"
