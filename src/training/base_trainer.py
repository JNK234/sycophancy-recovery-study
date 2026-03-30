# ABOUTME: Abstract base trainer class with shared training pipeline logic.
# ABOUTME: Subclasses implement prepare_dataset() and create_trainer() for each method.

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional

import wandb
from datasets import Dataset
from transformers import Trainer

from src.training.config_schema import ExperimentConfig
from src.training.model_setup import (
    setup_model_and_tokenizer,
    build_lora_config,
    merge_lora_adapter,
)


class BaseTrainer(ABC):
    """Shared training pipeline for all methods (SFT, DPO, etc.)."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.lora_config = None

    @property
    def adapter_path(self) -> str:
        return os.path.join(self.config.experiment.output_dir, "adapter")

    @property
    def merged_path(self) -> str:
        return os.path.join(self.config.experiment.output_dir, "merged")

    def setup(self) -> None:
        """Load model, tokenizer, build LoRA config, init wandb."""
        self.model, self.tokenizer = setup_model_and_tokenizer(self.config)
        self.lora_config = build_lora_config(self.config)

        if self.config.training.report_to == "wandb":
            wandb.init(
                project=self.config.wandb.project,
                name=self.config.experiment.name,
                tags=self.config.wandb.tags,
                config={
                    "method": self.config.experiment.method,
                    "model": self.config.model.name_or_path,
                    "lora_r": self.config.lora.r,
                    "learning_rate": self.config.training.learning_rate,
                },
            )

    @abstractmethod
    def prepare_dataset(self) -> tuple[Dataset, Optional[Dataset]]:
        """Load and format dataset. Subclass implements."""

    @abstractmethod
    def create_trainer(
        self, train_ds: Dataset, val_ds: Optional[Dataset]
    ) -> Trainer:
        """Instantiate the appropriate TRL trainer. Subclass implements."""

    @staticmethod
    def _is_main_process() -> bool:
        """Check if this is the main process (rank 0) in distributed training."""
        rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
        return rank == 0

    def train(self, resume_from_checkpoint: str | None = None) -> None:
        """Full training pipeline: setup -> data -> trainer -> train."""
        self.setup()
        train_ds, val_ds = self.prepare_dataset()
        trainer = self.create_trainer(train_ds, val_ds)
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        if self._is_main_process():
            self.save_adapter(trainer)

    def save_adapter(self, trainer: Trainer) -> None:
        """Save LoRA adapter weights. Only called from rank 0."""
        trainer.save_model(self.adapter_path)
        self.tokenizer.save_pretrained(self.adapter_path)
        print(f"Adapter saved to {self.adapter_path}")

    def merge(self) -> None:
        """Merge LoRA adapter into base model and save."""
        merge_lora_adapter(
            base_model_path=self.config.model.name_or_path,
            adapter_path=self.adapter_path,
            output_path=self.merged_path,
            cache_dir=self.config.model.cache_dir,
        )

    def evaluate(self) -> None:
        """Run post-training sycophancy evaluation using LLM-as-judge system."""
        from src.evaluation.config import EvalConfig
        from src.evaluation.runner import run_evaluation

        config = EvalConfig.from_experiment_config(self.config)
        run_evaluation(config)

    def run(self, resume_from_checkpoint: str | None = None) -> None:
        """Full pipeline: train -> merge -> evaluate."""
        self.train(resume_from_checkpoint=resume_from_checkpoint)
        if self._is_main_process():
            self.merge()
            if self.config.eval.run_after_training:
                self.evaluate()
