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
        """Merge LoRA adapter into base model and save. Optionally push to HF Hub."""
        merge_lora_adapter(
            base_model_path=self.config.model.name_or_path,
            adapter_path=self.adapter_path,
            output_path=self.merged_path,
            cache_dir=self.config.model.cache_dir,
        )

        if self.config.hf_hub.component:
            self._push_to_hub()

    def _push_to_hub(self) -> None:
        """Push merged + adapter to HF Hub. Failures are logged but do not crash training.

        Reads metrics from results/eval/<run-name>/summary.json if present and embeds
        them in the model card. Also embeds the active wandb run URL if available.
        """
        try:
            from src.training.hf_hub import push_model, add_to_collection
            import wandb as wb

            hub = self.config.hf_hub
            component = hub.component
            method = self.config.experiment.method

            wandb_url = None
            if wb.run is not None:
                wandb_url = wb.run.get_url()

            metrics = self._load_eval_metrics()
            config_yaml_path = os.path.join(
                self.config.experiment.output_dir, "config.yaml"
            )

            agg_syc = (metrics or {}).get("aggregate_sycophancy")
            note = f"{method} — aggregate sycophancy {agg_syc:.3f}" if isinstance(agg_syc, (int, float)) else method

            if hub.push_merged:
                print(f"Pushing merged model to HF Hub as '{component}'...")
                url = push_model(
                    local_dir=self.merged_path,
                    component=component,
                    method=method,
                    base_model=self.config.model.name_or_path,
                    private=hub.private,
                    config_yaml_path=config_yaml_path if os.path.exists(config_yaml_path) else None,
                    metrics=metrics,
                    wandb_url=wandb_url,
                    namespace=hub.namespace,
                )
                print(f"  Merged: {url}")
                add_to_collection(component=component, repo_type="model", note=note, namespace=hub.namespace)

            if hub.push_adapter:
                print(f"Pushing adapter to HF Hub as '{component}-adapter'...")
                adapter_component = f"{component}-adapter"
                url = push_model(
                    local_dir=self.adapter_path,
                    component=adapter_component,
                    method=method,
                    base_model=self.config.model.name_or_path,
                    private=hub.private,
                    config_yaml_path=config_yaml_path if os.path.exists(config_yaml_path) else None,
                    metrics=metrics,
                    wandb_url=wandb_url,
                    namespace=hub.namespace,
                    extra_notes=(
                        f"LoRA adapter. Merged weights: "
                        f"https://huggingface.co/{hub.namespace}/sycophancy-recovery-{component}"
                    ),
                )
                print(f"  Adapter: {url}")
                add_to_collection(component=adapter_component, repo_type="model", note=f"LoRA adapter for {component}", namespace=hub.namespace)

        except Exception as e:
            # Don't crash training on a push failure — it's recoverable via scripts/sync_to_hub.py
            print(f"WARNING: HF Hub auto-push failed ({type(e).__name__}: {e})")
            print(f"  Models are saved locally; recover with `python scripts/sync_to_hub.py --only {self.config.hf_hub.component}`")

    def _load_eval_metrics(self) -> dict | None:
        """Load summary.json from the eval output dir, if it exists."""
        import json
        # Eval results land in results/eval/<run-name>/ per project convention
        run_name = self.config.experiment.name
        candidates = [
            os.path.join("results", "eval", run_name, "summary.json"),
        ]
        for path in candidates:
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        return json.load(f)
                except Exception:
                    return None
        return None

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
