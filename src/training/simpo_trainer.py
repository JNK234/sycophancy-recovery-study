# ABOUTME: SimPO trainer for sycophancy recovery via reference-free preference optimization.
# ABOUTME: Wraps TRL CPOTrainer with SimPO loss — length-normalized rewards, no reference model.

from __future__ import annotations

import os
from typing import Optional

from datasets import Dataset

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
from trl.experimental.cpo import CPOTrainer, CPOConfig

from src.training.base_trainer import BaseTrainer
from src.training.callbacks import ConfigSaveCallback
from src.training.data_prep import load_dpo_dataset


class SimPORecoveryTrainer(BaseTrainer):
    """SimPO recovery from sycophancy using the same preference pairs as DPO.

    SimPO differs from DPO in three ways:
    1. No reference model — reward is the length-normalized average log-prob
    2. Length normalization — divides by response token count, preventing verbosity gaming
    3. Reward margin (gamma) — enforces minimum gap between chosen and rejected rewards
    """

    def prepare_dataset(self) -> tuple[Dataset, Optional[Dataset]]:
        return load_dpo_dataset(self.config.data)

    def create_trainer(
        self, train_ds: Dataset, val_ds: Optional[Dataset]
    ) -> CPOTrainer:
        training_kwargs = self.config.training.to_dict()

        args = CPOConfig(
            output_dir=self.config.experiment.output_dir,
            loss_type=self.config.simpo.loss_type,
            beta=self.config.simpo.beta,
            simpo_gamma=self.config.simpo.simpo_gamma,
            cpo_alpha=self.config.simpo.cpo_alpha,
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

        return CPOTrainer(
            model=self.model,
            args=args,
            peft_config=self.lora_config,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=self.tokenizer,
            callbacks=callbacks,
        )
