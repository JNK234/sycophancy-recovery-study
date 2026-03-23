# ABOUTME: Probing config dataclass with YAML loading support.
# ABOUTME: Defines model entries, data source, extraction params, and probe settings.

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelEntry:
    """One model to extract activations from."""
    name: str
    name_or_path: str
    judgment_name: str = ""
    cache_dir: Optional[str] = None
    dtype: str = "bfloat16"


@dataclass
class DataConfig:
    """Data source for prompt-only probing with per-model behavior labels."""
    eval_dataset: str
    judgment_dir: str
    templates: list[str] = field(default_factory=lambda: ["suggest_incorrect", "deny_correct"])
    max_samples: int = 500
    val_fraction: float = 0.2
    seed: int = 42


@dataclass
class ExtractionConfig:
    """How to extract activations."""
    batch_size: int = 4
    max_seq_length: int = 2048
    layers: str = "all"
    token_position: str = "last"
    enable_thinking: bool = False


@dataclass
class ProbeConfig:
    """Probe training parameters."""
    probe_type: str = "logistic"
    max_iter: int = 2000
    C: float = 1.0


@dataclass
class ProbingConfig:
    """Top-level probing config."""
    name: str
    output_dir: str
    results_dir: str
    models: list[ModelEntry] = field(default_factory=list)
    data: DataConfig = field(default_factory=DataConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    reference_model: str = "sft"
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> ProbingConfig:
        """Load probing config from YAML file."""
        config_dir = Path(path).resolve().parent

        with open(path) as f:
            raw = yaml.safe_load(f)

        project_root = _find_project_root(config_dir)

        models = [ModelEntry(**m) for m in raw.get("models", [])]

        data_raw = raw.get("data", {})
        if "eval_dataset" in data_raw and not Path(data_raw["eval_dataset"]).is_absolute():
            data_raw["eval_dataset"] = str(project_root / data_raw["eval_dataset"])

        output_dir = str(_resolve_path(raw["output_dir"], project_root))
        results_dir = str(_resolve_path(raw["results_dir"], project_root))

        return cls(
            name=raw["name"],
            output_dir=output_dir,
            results_dir=results_dir,
            models=models,
            data=DataConfig(**data_raw),
            extraction=ExtractionConfig(**raw.get("extraction", {})),
            probe=ProbeConfig(**raw.get("probe", {})),
            reference_model=raw.get("reference_model", "sft"),
            seed=raw.get("seed", 42),
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
