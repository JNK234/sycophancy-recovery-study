# ABOUTME: Typed dataclass schema for experiment YAML configs.
# ABOUTME: Loads, validates, and provides structured access to all training parameters.

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ExperimentSection:
    name: str
    method: str  # "sft" | "dpo" | "simpo" | "ppo" | "cai"
    seed: int = 42
    output_dir: str = "outputs"


@dataclass
class ModelSection:
    name_or_path: str
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    cache_dir: Optional[str] = None


@dataclass
class TokenizerSection:
    pad_token: str = "<|endoftext|>"
    padding_side: str = "right"
    enable_thinking: bool = False


@dataclass
class LoRASection:
    r: int = 16
    lora_alpha: int = 32
    target_modules: str = "all-linear"
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class DataSection:
    train_file: str
    val_split: float = 0.05
    max_length: int = 2048
    prompt_field: str = "prompt"
    completion_field: str = "response"
    chosen_field: str = "chosen"
    rejected_field: str = "rejected"


@dataclass
class TrainingSection:
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    bf16: bool = True
    gradient_checkpointing: bool = True
    ddp_find_unused_parameters: bool = False
    logging_steps: int = 10
    save_strategy: str = "steps"
    save_steps: int = 200
    save_total_limit: int = 3
    max_steps: int = -1  # -1 means use num_train_epochs instead
    report_to: str = "wandb"
    label_names: list[str] = field(default_factory=lambda: ["labels"])
    # Mid-training sycophancy eval (not passed to SFTConfig/DPOConfig)
    eval_every_steps: int = 0  # 0 = disabled
    eval_samples: int = 200
    eval_dataset_path: str = ""

    # Fields that are project-specific and should NOT be passed to HF TrainingArguments
    _custom_fields: list[str] = field(
        default_factory=lambda: ["eval_every_steps", "eval_samples", "eval_dataset_path"],
        repr=False,
    )

    def to_dict(self) -> dict:
        """Return training params as dict for TrainingArguments/SFTConfig/DPOConfig.

        Excludes project-specific fields that HF doesn't recognize.
        """
        d = asdict(self)
        for key in self._custom_fields:
            d.pop(key, None)
        d.pop("_custom_fields", None)
        return d


@dataclass
class DPOSection:
    beta: float = 0.1
    loss_type: str = "sigmoid"


@dataclass
class SimPOSection:
    beta: float = 2.0           # Much larger than DPO — raw log-probs need higher scaling
    simpo_gamma: float = 0.5    # Target reward margin between chosen and rejected
    cpo_alpha: float = 0.0      # 0 = pure SimPO, >0 adds behavioral cloning regularizer
    loss_type: str = "simpo"


@dataclass
class GRPOSection:
    num_generations: int = 8            # Completions per prompt for group-relative advantage
    max_completion_length: int = 256    # Max tokens per generated completion
    temperature: float = 0.7            # Generation temperature (lower than TRL default 1.0)
    beta: float = 0.04                  # KL penalty coefficient (0 = no constraint)
    epsilon: float = 0.2               # PPO clipping range
    loss_type: str = "grpo"            # "grpo" (vanilla) or "dapo" (dynamic sampling)
    scale_rewards: str = "group"       # Group normalization of rewards
    reward_model_path: str = ""        # Path to trained reward model (merged)
    reward_type: str = "model"         # "model" (trained RM) or "rule_based" (heuristic)
    log_completions: bool = True       # Log sample completions to wandb


@dataclass
class WandbSection:
    project: str = "sycophancy-recovery"
    tags: list[str] = field(default_factory=list)


@dataclass
class EvalSection:
    run_after_training: bool = True
    eval_datasets: list[str] = field(default_factory=list)
    max_eval_samples: int = 200
    tensor_parallel_size: int = 4
    judge_model: str = "Qwen/Qwen2.5-72B-Instruct"
    judge_tensor_parallel_size: int = 4
    judge_max_model_len: int = 4096
    judge_temperature: float = 0.0
    judge_max_tokens: int = 1024
    seen_question_file: Optional[str] = None


@dataclass
class ExperimentConfig:
    experiment: ExperimentSection
    model: ModelSection
    tokenizer: TokenizerSection
    lora: LoRASection
    data: DataSection
    training: TrainingSection
    dpo: DPOSection = field(default_factory=DPOSection)
    simpo: SimPOSection = field(default_factory=SimPOSection)
    grpo: GRPOSection = field(default_factory=GRPOSection)
    wandb: WandbSection = field(default_factory=WandbSection)
    eval: EvalSection = field(default_factory=EvalSection)

    @classmethod
    def from_yaml(cls, path: str) -> ExperimentConfig:
        """Load config from YAML file, resolve relative paths."""
        config_dir = Path(path).resolve().parent

        with open(path) as f:
            raw = yaml.safe_load(f)

        config = cls(
            experiment=ExperimentSection(**raw["experiment"]),
            model=ModelSection(**raw["model"]),
            tokenizer=TokenizerSection(**raw.get("tokenizer", {})),
            lora=LoRASection(**raw.get("lora", {})),
            data=DataSection(**raw["data"]),
            training=TrainingSection(**raw.get("training", {})),
            dpo=DPOSection(**raw.get("dpo", {})),
            simpo=SimPOSection(**raw.get("simpo", {})),
            grpo=GRPOSection(**raw.get("grpo", {})),
            wandb=WandbSection(**raw.get("wandb", {})),
            eval=EvalSection(**raw.get("eval", {})),
        )

        # Validate method
        valid_methods = {"sft", "dpo", "simpo", "ppo", "grpo", "cai"}
        if config.experiment.method not in valid_methods:
            raise ValueError(
                f"Unknown method '{config.experiment.method}'. "
                f"Must be one of {valid_methods}"
            )

        # Resolve relative paths against project root (config_dir's parent)
        project_root = config_dir.parent.parent
        config.data.train_file = str(
            cls._resolve_path(config.data.train_file, project_root)
        )
        config.experiment.output_dir = str(
            cls._resolve_path(config.experiment.output_dir, project_root)
        )
        config.eval.eval_datasets = [
            str(cls._resolve_path(p, project_root))
            for p in config.eval.eval_datasets
        ]

        # Resolve GRPO reward model path
        if config.grpo.reward_model_path:
            config.grpo.reward_model_path = str(
                cls._resolve_path(config.grpo.reward_model_path, project_root)
            )

        # Resolve mid-training eval dataset path
        if config.training.eval_dataset_path:
            config.training.eval_dataset_path = str(
                cls._resolve_path(config.training.eval_dataset_path, project_root)
            )

        return config

    @staticmethod
    def _resolve_path(path_str: str, root: Path) -> Path:
        """Resolve path relative to project root if not absolute."""
        p = Path(path_str)
        if p.is_absolute():
            return p
        return root / p

    def save_yaml(self, path: str) -> None:
        """Save config to YAML for reproducibility."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
