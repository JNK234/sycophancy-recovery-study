# ABOUTME: Model, tokenizer, and LoRA setup utilities.
# ABOUTME: Handles Qwen3-specific config (thinking mode, pad token) and adapter merging.

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, TaskType

from training.config_schema import ExperimentConfig


TORCH_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def setup_model_and_tokenizer(
    config: ExperimentConfig,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer with project-specific settings."""
    dtype = TORCH_DTYPE_MAP[config.model.torch_dtype]

    model = AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path,
        torch_dtype=dtype,
        attn_implementation=config.model.attn_implementation,
        cache_dir=config.model.cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name_or_path,
        cache_dir=config.model.cache_dir,
    )

    # Set pad token (Qwen3 uses <|endoftext|> as pad, distinct from EOS <|im_end|>)
    if config.tokenizer.pad_token:
        tokenizer.pad_token = config.tokenizer.pad_token
    tokenizer.padding_side = config.tokenizer.padding_side

    # Disable thinking mode for Qwen3 if configured
    if not config.tokenizer.enable_thinking and hasattr(model, "generation_config"):
        model.generation_config.do_sample = True
        model.generation_config.temperature = 0.7
        model.generation_config.top_p = 0.8
        model.generation_config.top_k = 20

    return model, tokenizer


def build_lora_config(config: ExperimentConfig) -> LoraConfig:
    """Build PEFT LoraConfig from experiment config."""
    target_modules = config.lora.target_modules
    if target_modules != "all-linear":
        target_modules = [m.strip() for m in target_modules.split(",")]

    return LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        target_modules=target_modules,
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
        task_type=TaskType[config.lora.task_type],
    )


def merge_lora_adapter(
    base_model_path: str,
    adapter_path: str,
    output_path: str,
    cache_dir: str | None = None,
) -> None:
    """Merge LoRA adapter into base model and save as standalone model."""
    dtype = torch.bfloat16

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        cache_dir=cache_dir,
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    merged = model.merge_and_unload()

    merged.save_pretrained(output_path)

    # Save tokenizer alongside merged model
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, cache_dir=cache_dir
    )
    tokenizer.save_pretrained(output_path)

    print(f"Merged model saved to {output_path}")
