# ABOUTME: Load eval prompts and per-model behavior labels for probing.
# ABOUTME: Prompt-only approach: extract model state BEFORE generation, label by actual behavior.

from __future__ import annotations

import json
import random
from dataclasses import dataclass

import numpy as np
from transformers import AutoTokenizer

from src.probing.config import DataConfig, ModelEntry


@dataclass
class ProbeDataset:
    """Prompt-only probe dataset with per-model behavior labels."""
    train_indices: list[int]
    val_indices: list[int]
    all_prompts: list[str]
    model_labels: dict[str, np.ndarray]  # {model_name: array of 0/1}
    prompt_groups: list[str]  # question ID per prompt for grouping


def load_prompts_and_labels(
    config: DataConfig,
    models: list[ModelEntry],
    tokenizer: AutoTokenizer,
) -> ProbeDataset:
    """Load eval prompts and per-model behavior labels from judge results.

    Uses only sycophancy-pressure templates (suggest_incorrect, deny_correct)
    where models differ most in behavior. Labels come from each model's actual
    judge verdict: incorrect=1 (sycophantic), correct=0 (honest).
    """
    # Load eval dataset
    with open(config.eval_dataset) as f:
        eval_rows = [json.loads(line) for line in f]

    # Load generation metadata to get template_type per idx
    # Use first model's generations (all models share same prompt ordering)
    first_model = models[0]
    gen_path = f"{config.judgment_dir}/{first_model.judgment_name}/generations/answer.jsonl"
    with open(gen_path) as f:
        gen_meta = [json.loads(line) for line in f]

    # Load judgments for each model
    model_judgments = {}
    for model_entry in models:
        jpath = f"{config.judgment_dir}/{model_entry.judgment_name}/judgments/answer.jsonl"
        with open(jpath) as f:
            model_judgments[model_entry.name] = [json.loads(line) for line in f]

    # Filter to pressure templates and build prompt list
    valid_indices = []
    prompts = []
    prompt_groups = []
    model_label_lists = {m.name: [] for m in models}

    for idx, gen in enumerate(gen_meta):
        template = gen.get("template_type", "")
        if template not in config.templates:
            continue

        # Get each model's verdict for this prompt
        verdicts = {}
        skip = False
        for model_entry in models:
            verdict = model_judgments[model_entry.name][idx]["verdict"]
            if verdict in ("hedged", "refused"):
                skip = True
                break
            verdicts[model_entry.name] = 1 if verdict == "incorrect" else 0
        if skip:
            continue

        # Format prompt (user message only, no response)
        eval_row = eval_rows[idx]
        user_content = eval_row["prompt"][0]["content"]
        formatted = _format_prompt(user_content, tokenizer)

        prompts.append(formatted)
        valid_indices.append(idx)
        # Group by base question (every 4 consecutive idx = same question)
        prompt_groups.append(str(idx // 4))

        for model_entry in models:
            model_label_lists[model_entry.name].append(verdicts[model_entry.name])

    # Sample if needed
    rng = random.Random(config.seed)
    if config.max_samples > 0 and len(prompts) > config.max_samples:
        sample_idx = rng.sample(range(len(prompts)), config.max_samples)
        sample_idx.sort()
        prompts = [prompts[i] for i in sample_idx]
        prompt_groups = [prompt_groups[i] for i in sample_idx]
        for name in model_label_lists:
            model_label_lists[name] = [model_label_lists[name][i] for i in sample_idx]

    # Convert labels to numpy
    model_labels = {
        name: np.array(labels) for name, labels in model_label_lists.items()
    }

    # Split by question group (no prompt leakage)
    train_idx, val_idx = _split_by_group(prompt_groups, config.val_fraction, config.seed)

    # Print stats
    print(f"  Probe dataset: {len(prompts)} prompts "
          f"({len(train_idx)} train, {len(val_idx)} val)")
    for name, labels in model_labels.items():
        syc_rate = labels.mean()
        print(f"    {name}: {syc_rate:.1%} sycophantic ({int(labels.sum())}/{len(labels)})")

    return ProbeDataset(
        train_indices=train_idx,
        val_indices=val_idx,
        all_prompts=prompts,
        model_labels=model_labels,
        prompt_groups=prompt_groups,
    )


def _format_prompt(user_content: str, tokenizer: AutoTokenizer) -> str:
    """Format user prompt for extraction (no response, generation prompt appended)."""
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _split_by_group(
    groups: list[str],
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Split indices by group so all samples in a group stay together."""
    rng = random.Random(seed)
    unique_groups = sorted(set(groups))
    rng.shuffle(unique_groups)

    val_count = int(len(unique_groups) * val_fraction)
    val_groups = set(unique_groups[:val_count])

    train_idx = [i for i, g in enumerate(groups) if g not in val_groups]
    val_idx = [i for i, g in enumerate(groups) if g in val_groups]

    return train_idx, val_idx
