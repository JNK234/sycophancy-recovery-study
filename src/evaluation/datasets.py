# ABOUTME: Dataset loading with seen/unseen split based on training data overlap.
# ABOUTME: Loads JSONL eval datasets, tags each row with 'seen' flag, applies max_samples.

from __future__ import annotations

import hashlib
import json
import random
from typing import Optional


def load_eval_dataset(
    path: str,
    max_samples: int = 0,
    seen_question_file: Optional[str] = None,
    seed: int = 42,
) -> list[dict]:
    """Load a JSONL eval dataset and tag rows with seen/unseen status.

    Args:
        path: Path to JSONL dataset file.
        max_samples: Max rows to return (0 = all). Samples proportionally from seen/unseen.
        seen_question_file: Path to training data JSONL for seen/unseen split.
        seed: Random seed for sampling.

    Returns:
        List of row dicts, each with an added 'seen' field (bool).
    """
    data = _load_jsonl(path)

    # Build set of seen questions from training data
    seen_questions = set()
    if seen_question_file:
        seen_questions = _load_seen_questions(seen_question_file)

    # Tag each row with seen status and stable identifiers
    for row in data:
        question = _extract_question(row)
        row["seen"] = question in seen_questions if seen_questions else False
        row["prompt_id"] = _content_hash(row["prompt"][0]["content"])
        row["group_id"] = _content_hash(question)

    # Sample if needed
    if max_samples > 0 and len(data) > max_samples:
        data = _proportional_sample(data, max_samples, seed)

    return data


def _load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_seen_questions(path: str) -> set[str]:
    """Extract unique question strings from training data."""
    questions = set()
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            q = row.get("original_question", "")
            if q:
                questions.add(q.strip().lower())
    return questions


def _extract_question(row: dict) -> str:
    """Extract the core question text from an eval row for seen/unseen matching."""
    base = row.get("base", {})
    question = base.get("question", "")
    if not question:
        # For feedback dataset, use the text field
        question = base.get("text", "")
    return question.strip().lower()


def _content_hash(text: str) -> str:
    """Deterministic short hash from text content for stable row identification."""
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()[:16]


def _proportional_sample(data: list[dict], n: int, seed: int) -> list[dict]:
    """Sample n items proportionally from seen and unseen groups."""
    rng = random.Random(seed)

    seen = [r for r in data if r.get("seen", False)]
    unseen = [r for r in data if not r.get("seen", False)]

    total = len(data)
    n_seen = round(n * len(seen) / total) if total > 0 else 0
    n_unseen = n - n_seen

    # Clamp to available
    n_seen = min(n_seen, len(seen))
    n_unseen = min(n_unseen, len(unseen))

    sampled = rng.sample(seen, n_seen) + rng.sample(unseen, n_unseen)
    rng.shuffle(sampled)
    return sampled
