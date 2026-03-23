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
        from src.evaluation.config import (
            EvalConfig, ModelConfig, JudgeConfig, GenerationConfig, DatasetEntry,
        )
        from src.evaluation.datasets import load_eval_dataset
        from src.evaluation import get_evaluator
        from src.evaluation.generate import run_generation_pass, run_challenge_generation
        from src.evaluation.judge import run_judge_pass
        from src.evaluation.metrics import compute_all_metrics
        from src.evaluation.report import print_report

        eval_cfg = self.config.eval
        exp_cfg = self.config.experiment

        # Build EvalConfig from ExperimentConfig
        datasets = [
            DatasetEntry(
                path=ds_path,
                type=os.path.splitext(os.path.basename(ds_path))[0],
                max_samples=eval_cfg.max_eval_samples,
                seen_question_file=eval_cfg.seen_question_file,
            )
            for ds_path in eval_cfg.eval_datasets
        ]

        config = EvalConfig(
            name=f"{exp_cfg.name}-eval",
            output_dir=os.path.join(exp_cfg.output_dir, "eval"),
            seed=exp_cfg.seed,
            model=ModelConfig(
                name_or_path=self.merged_path,
                tensor_parallel_size=eval_cfg.tensor_parallel_size,
                cache_dir=self.config.model.cache_dir,
            ),
            judge=JudgeConfig(
                name_or_path=eval_cfg.judge_model,
                tensor_parallel_size=eval_cfg.judge_tensor_parallel_size,
                max_model_len=eval_cfg.judge_max_model_len,
                cache_dir=self.config.model.cache_dir,
                temperature=eval_cfg.judge_temperature,
                max_tokens=eval_cfg.judge_max_tokens,
            ),
            generation=GenerationConfig(),
            datasets=datasets,
        )

        os.makedirs(config.output_dir, exist_ok=True)
        config.save_yaml(os.path.join(config.output_dir, "config.yaml"))

        eval_results = {}
        subject_llm = None
        judge_llm = None

        # Pass 1: Generation
        print("\n--- Eval Pass 1: Generating responses ---")
        for ds_entry in config.datasets:
            data = load_eval_dataset(
                path=ds_entry.path,
                max_samples=ds_entry.max_samples,
                seen_question_file=ds_entry.seen_question_file,
                seed=config.seed,
            )
            evaluator = get_evaluator(ds_entry.type)()
            prompts = evaluator.build_generation_prompts(data)

            gen_path = os.path.join(config.output_dir, "generations", f"{ds_entry.type}.jsonl")
            generations, subject_llm = run_generation_pass(
                config, ds_entry.type, prompts, gen_path, llm=subject_llm,
            )

            if ds_entry.type == "are_you_sure":
                challenge_prompts = evaluator.build_challenge_prompts(generations)
                challenge_gens = run_challenge_generation(
                    config, challenge_prompts, gen_path, subject_llm,
                )
                generations = generations + challenge_gens

            eval_results[ds_entry.type] = {"generations": generations, "data": data}

        del subject_llm
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Pass 2: Judge scoring
        print("\n--- Eval Pass 2: Judge scoring ---")
        for ds_entry in config.datasets:
            if ds_entry.type not in eval_results:
                continue
            evaluator = get_evaluator(ds_entry.type)()
            generations = eval_results[ds_entry.type]["generations"]
            judge_prompts = evaluator.build_judge_prompts(generations)

            judge_path = os.path.join(config.output_dir, "judgments", f"{ds_entry.type}.jsonl")
            judgments, judge_llm = run_judge_pass(
                config, ds_entry.type, judge_prompts, judge_path, llm=judge_llm,
            )
            eval_results[ds_entry.type]["judgments"] = judgments

        del judge_llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Metrics & Report
        print("\n--- Computing metrics ---")
        summary = compute_all_metrics(eval_results, config.output_dir)
        print_report(summary, config.name)

    def run(self, resume_from_checkpoint: str | None = None) -> None:
        """Full pipeline: train -> merge -> evaluate."""
        self.train(resume_from_checkpoint=resume_from_checkpoint)
        if self._is_main_process():
            self.merge()
            if self.config.eval.run_after_training:
                self.evaluate()
