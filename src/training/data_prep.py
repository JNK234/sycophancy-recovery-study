# ABOUTME: Converts project JSONL data files into TRL-compatible Dataset format.
# ABOUTME: Handles SFT (prompt/completion) and DPO (prompt/chosen/rejected) conversions.

from __future__ import annotations

import json
from typing import Optional

from datasets import Dataset

from src.training.config_schema import DataSection


def _load_jsonl(path: str) -> list[dict]:
    """Load JSONL file into list of dicts."""
    with open(path) as f:
        return [json.loads(line) for line in f]


def _to_chat_messages(text: str, role: str) -> list[dict]:
    """Wrap plain text into a single chat message."""
    return [{"role": role, "content": text}]


def load_sft_dataset(
    data_config: DataSection,
) -> tuple[Dataset, Optional[Dataset]]:
    """Load SFT data and convert to TRL prompt-completion chat format.

    Input JSONL format: {"prompt": str, "response": str, ...}
    Output dataset columns: {"prompt": [messages], "completion": [messages]}
    """
    raw = _load_jsonl(data_config.train_file)

    records = []
    for row in raw:
        records.append({
            "prompt": _to_chat_messages(
                row[data_config.prompt_field], "user"
            ),
            "completion": _to_chat_messages(
                row[data_config.completion_field], "assistant"
            ),
        })

    dataset = Dataset.from_list(records)
    return _split_dataset(dataset, data_config.val_split)


def load_dpo_dataset(
    data_config: DataSection,
) -> tuple[Dataset, Optional[Dataset]]:
    """Load DPO data and convert to TRL prompt/chosen/rejected chat format.

    Input JSONL format: {"prompt": str, "chosen": str, "rejected": str, ...}
    Output dataset columns: {"prompt": [messages], "chosen": [messages], "rejected": [messages]}
    """
    raw = _load_jsonl(data_config.train_file)

    records = []
    for row in raw:
        records.append({
            "prompt": _to_chat_messages(
                row[data_config.prompt_field], "user"
            ),
            "chosen": _to_chat_messages(
                row[data_config.chosen_field], "assistant"
            ),
            "rejected": _to_chat_messages(
                row[data_config.rejected_field], "assistant"
            ),
        })

    dataset = Dataset.from_list(records)
    return _split_dataset(dataset, data_config.val_split)


def _split_dataset(
    dataset: Dataset, val_split: float
) -> tuple[Dataset, Optional[Dataset]]:
    """Split dataset into train/val if val_split > 0."""
    if val_split <= 0:
        return dataset, None

    splits = dataset.train_test_split(test_size=val_split, seed=42)
    return splits["train"], splits["test"]
