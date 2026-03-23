# ABOUTME: DPO trainer for recovering from sycophancy via preference optimization.
# ABOUTME: Loads DPO pairs and wraps TRL DPOTrainer with PEFT reference model handling.

from __future__ import annotations

from typing import Optional

from datasets import Dataset
from trl import DPOTrainer as TRLDPOTrainer, DPOConfig

from src.training.base_trainer import BaseTrainer
from src.training.callbacks import ConfigSaveCallback
from src.training.data_prep import load_dpo_dataset


class DPORecoveryTrainer(BaseTrainer):
    """DPO to recover from sycophancy using honest/sycophantic preference pairs."""

    def prepare_dataset(self) -> tuple[Dataset, Optional[Dataset]]:
        return load_dpo_dataset(self.config.data)

    def create_trainer(
        self, train_ds: Dataset, val_ds: Optional[Dataset]
    ) -> TRLDPOTrainer:
        training_kwargs = self.config.training.to_dict()

        args = DPOConfig(
            output_dir=self.config.experiment.output_dir,
            beta=self.config.dpo.beta,
            loss_type=self.config.dpo.loss_type,
            max_length=self.config.data.max_length,
            seed=self.config.experiment.seed,
            **training_kwargs,
        )

        callbacks = [ConfigSaveCallback(self.config)]

        if self.config.training.eval_every_steps > 0 and self.config.training.eval_dataset_path:
            from src.training.eval_callback import SycophancyEvalCallback
            callbacks.append(SycophancyEvalCallback(
                eval_data_path=self.config.training.eval_dataset_path,
                tokenizer=self.tokenizer,
                n_samples=self.config.training.eval_samples,
                eval_every_steps=self.config.training.eval_every_steps,
                seed=self.config.experiment.seed,
            ))

        return TRLDPOTrainer(
            model=self.model,
            ref_model=None,  # PEFT: base weights (adapter disabled) serve as reference
            args=args,
            peft_config=self.lora_config,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=self.tokenizer,
            callbacks=callbacks,
        )
