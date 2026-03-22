# ABOUTME: Evaluation config dataclass with YAML loading support.
# ABOUTME: Defines model, judge, generation, and dataset parameters.

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    name_or_path: str
    tensor_parallel_size: int = 4
    max_model_len: int = 4096
    cache_dir: Optional[str] = None
    dtype: str = "bfloat16"


@dataclass
class JudgeConfig:
    name_or_path: str = "Qwen/Qwen2.5-72B-Instruct"
    tensor_parallel_size: int = 4
    max_model_len: int = 4096
    cache_dir: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 1024
    dtype: str = "bfloat16"


@dataclass
class GenerationConfig:
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    max_tokens: int = 512


@dataclass
class DatasetEntry:
    path: str
    type: str  # "answer" | "are_you_sure" | "feedback"
    max_samples: int = 0  # 0 means use all
    seen_question_file: Optional[str] = None


@dataclass
class EvalConfig:
    name: str
    output_dir: str
    model: ModelConfig
    judge: JudgeConfig
    generation: GenerationConfig
    datasets: list[DatasetEntry] = field(default_factory=list)
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> EvalConfig:
        """Load eval config from a YAML file."""
        config_dir = Path(path).resolve().parent

        with open(path) as f:
            raw = yaml.safe_load(f)

        # Resolve relative paths against project root
        project_root = _find_project_root(config_dir)

        datasets = []
        for ds in raw.get("datasets", []):
            ds_path = str(_resolve_path(ds["path"], project_root))
            seen_file = ds.get("seen_question_file")
            if seen_file:
                seen_file = str(_resolve_path(seen_file, project_root))
            datasets.append(DatasetEntry(
                path=ds_path,
                type=ds["type"],
                max_samples=ds.get("max_samples", 0),
                seen_question_file=seen_file,
            ))

        output_dir = str(_resolve_path(raw["output_dir"], project_root))

        model_raw = raw.get("model", {})
        judge_raw = raw.get("judge", {})
        gen_raw = raw.get("generation", {})

        return cls(
            name=raw["name"],
            output_dir=output_dir,
            seed=raw.get("seed", 42),
            model=ModelConfig(**model_raw),
            judge=JudgeConfig(**judge_raw),
            generation=GenerationConfig(**gen_raw),
            datasets=datasets,
        )

    def save_yaml(self, path: str) -> None:
        """Save config to YAML for reproducibility."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)


def _find_project_root(start: Path) -> Path:
    """Walk up from start to find directory containing .git or pyproject.toml."""
    current = start
    for _ in range(10):
        if (current / ".git").exists() or (current / "pyproject.toml").exists():
            return current
        if current.parent == current:
            break
        current = current.parent
    return start


def _resolve_path(path_str: str, root: Path) -> Path:
    """Resolve path relative to project root if not absolute."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return root / p
