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

    # Detect whether artifacts have prompt_id (new format) or not (legacy)
    has_prompt_ids = bool(gen_meta and gen_meta[0].get("prompt_id"))

    # Load judgments for each model — use dict keyed by prompt_id or idx
    model_judgments = {}
    for model_entry in models:
        jpath = f"{config.judgment_dir}/{model_entry.judgment_name}/judgments/answer.jsonl"
        with open(jpath) as f:
            jlist = [json.loads(line) for line in f]
        if has_prompt_ids:
            model_judgments[model_entry.name] = {j["prompt_id"]: j for j in jlist}
        else:
            model_judgments[model_entry.name] = {j["idx"]: j for j in jlist}

    # Build generation lookup (prompt_id or idx keyed)
    if has_prompt_ids:
        gen_by_key = {g["prompt_id"]: g for g in gen_meta}
        eval_by_key = {}
        for row in eval_rows:
            pid = row.get("prompt_id")
            if pid:
                eval_by_key[pid] = row
        # Validate key sets match across models
        gen_keys = set(gen_by_key.keys())
        for model_entry in models:
            j_keys = set(model_judgments[model_entry.name].keys())
            missing = gen_keys - j_keys
            if missing:
                print(f"  WARNING: {len(missing)} generation prompt_ids missing from "
                      f"{model_entry.name} judgments")
    else:
        gen_by_key = {g["idx"]: g for g in gen_meta}
        eval_by_key = {i: row for i, row in enumerate(eval_rows)}

    # Filter to pressure templates and build prompt list
    valid_indices = []
    prompts = []
    prompt_groups = []
    model_label_lists = {m.name: [] for m in models}

    for key, gen in gen_by_key.items():
        template = gen.get("template_type", "")
        if template not in config.templates:
            continue

        # Get each model's verdict for this prompt
        verdicts = {}
        skip = False
        for model_entry in models:
            judgment = model_judgments[model_entry.name].get(key)
            if judgment is None:
                skip = True
                break
            verdict = judgment["verdict"]
            if verdict in ("hedged", "refused"):
                skip = True
                break
            verdicts[model_entry.name] = 1 if verdict == "incorrect" else 0
        if skip:
            continue

        # Format prompt (user message only, no response)
        eval_row = eval_by_key.get(key)
        if eval_row is None:
            continue
        user_content = eval_row["prompt"][0]["content"]
        formatted = _format_prompt(user_content, tokenizer)

        prompts.append(formatted)
        valid_indices.append(gen["idx"])
        # Group by stable group_id when available, else fall back to idx // 4
        if has_prompt_ids:
            prompt_groups.append(gen.get("group_id", str(gen["idx"] // 4)))
        else:
            prompt_groups.append(str(gen["idx"] // 4))

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
