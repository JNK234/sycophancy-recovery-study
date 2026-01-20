# ABOUTME: Model configurations for inference
# ABOUTME: Contains model-specific settings (thinking tokens, generation params)

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    name: str
    # Generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    # Qwen-specific: thinking mode support
    supports_thinking: bool = False
    think_end_token_id: Optional[int] = None  # e.g., 151668 for Qwen3


# Model registry - add new models here
MODELS = {
    "qwen3-8b": ModelConfig(
        name="Qwen/Qwen3-8B",
        supports_thinking=True,
        think_end_token_id=151668,
        temperature=0.7,
        top_p=0.8,
    ),
    "qwen3-8b-awq": ModelConfig(
        name="Qwen/Qwen3-8B-AWQ",
        supports_thinking=True,
        think_end_token_id=151668,
    ),
    "llama3-8b": ModelConfig(
        name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        supports_thinking=False,
    ),
    "mistral-7b": ModelConfig(
        name="mistralai/Mistral-7B-Instruct-v0.3",
        supports_thinking=False,
    ),
}


@dataclass
class InferenceConfig:
    # Model selection - use key from MODELS registry
    model_key: str = "qwen3-8b"
    # Quantization
    use_4bit: bool = False
    # Thinking mode (only applies if model supports it)
    enable_thinking: bool = False
    # Output
    output_dir: str = "data/raw"
    # HuggingFace cache directory (None = default ~/.cache/huggingface)
    hf_cache_dir: Optional[str] = None

    @property
    def model(self) -> ModelConfig:
        return MODELS[self.model_key]
