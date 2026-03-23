# ABOUTME: Extract hidden state activations from models for probing.
# ABOUTME: Loads model, runs forward passes, saves last-token activations at all layers.

from __future__ import annotations

import gc
import os

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.probing.config import ModelEntry, ExtractionConfig


TORCH_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def load_model_for_extraction(
    model_entry: ModelEntry,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model on single GPU in eval mode."""
    dtype = TORCH_DTYPE_MAP[model_entry.dtype]

    model = AutoModelForCausalLM.from_pretrained(
        model_entry.name_or_path,
        torch_dtype=dtype,
        device_map="cuda:0",
        cache_dir=model_entry.cache_dir,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_entry.name_or_path,
        cache_dir=model_entry.cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer


def extract_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    config: ExtractionConfig,
) -> dict[int, np.ndarray]:
    """Extract last-token hidden states at all layers for given texts.

    Returns dict mapping layer_index -> array of shape [N, hidden_dim].
    """
    # Parse which layers to extract
    num_layers = model.config.num_hidden_layers
    if config.layers == "all":
        layer_indices = list(range(num_layers))
    else:
        layer_indices = [int(x) for x in config.layers.split(",")]

    all_activations = {l: [] for l in layer_indices}
    batch_size = config.batch_size

    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[batch_start:batch_start + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_seq_length,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # hidden_states is tuple of (num_layers+1) tensors, each [batch, seq, hidden]
        # Index 0 = embedding output, index i = layer i output
        hidden_states = outputs.hidden_states

        # With left-padding, real tokens are right-aligned.
        # The last real token is always at seq_len - 1.
        seq_len = inputs["input_ids"].shape[1]
        last_positions = torch.full(
            (len(batch_texts),), seq_len - 1,
            device=model.device, dtype=torch.long,
        )

        for layer_idx in layer_indices:
            # hidden_states[0] = embedding, hidden_states[1] = layer 0, etc.
            layer_hidden = hidden_states[layer_idx + 1]

            # Extract last-token activation per sample
            batch_acts = []
            for sample_idx in range(len(batch_texts)):
                pos = last_positions[sample_idx].item()
                act = layer_hidden[sample_idx, pos, :].cpu().float().numpy()
                batch_acts.append(act)

            all_activations[layer_idx].extend(batch_acts)

        # Free GPU memory from this batch
        del outputs, hidden_states, inputs
        torch.cuda.empty_cache()

        if (batch_start // batch_size) % 25 == 0:
            n_done = min(batch_start + batch_size, len(texts))
            print(f"    Extracted {n_done}/{len(texts)} samples")

    # Stack into arrays
    result = {}
    for layer_idx in layer_indices:
        result[layer_idx] = np.stack(all_activations[layer_idx], axis=0)

    return result


def extract_and_save(
    model_entry: ModelEntry,
    texts: list[str],
    labels: list[int],
    config: ExtractionConfig,
    output_dir: str,
) -> str:
    """Full extraction pipeline for one model. Returns path to saved file."""
    print(f"\n  Extracting activations for model: {model_entry.name}")
    print(f"    Path: {model_entry.name_or_path}")
    print(f"    Samples: {len(texts)}")

    model, tokenizer = load_model_for_extraction(model_entry)

    activations = extract_activations(model, tokenizer, texts, config)

    # Free model from GPU
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"    Model unloaded, GPU memory freed")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{model_entry.name}.pt")

    sample_layer = list(activations.keys())[0]
    hidden_dim = activations[sample_layer].shape[1]

    torch.save({
        "activations": activations,
        "labels": np.array(labels),
        "metadata": {
            "model_name": model_entry.name,
            "model_path": model_entry.name_or_path,
            "num_samples": len(texts),
            "num_layers": len(activations),
            "hidden_dim": hidden_dim,
            "layers": sorted(activations.keys()),
            "token_position": config.token_position,
        },
    }, save_path)

    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"    Saved: {save_path} ({size_mb:.1f} MB)")

    return save_path


def load_activations(path: str) -> dict:
    """Load saved activations from .pt file."""
    return torch.load(path, weights_only=False)
