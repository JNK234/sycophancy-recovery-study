# ABOUTME: SFT trainer for inducing sycophancy via supervised fine-tuning.
# ABOUTME: Loads sycophantic training data and wraps TRL SFTTrainer.

from __future__ import annotations

from typing import Optional

from datasets import Dataset
from trl import SFTTrainer, SFTConfig

from src.training.base_trainer import BaseTrainer
from src.training.callbacks import ConfigSaveCallback
from src.training.data_prep import load_sft_dataset


class SFTSycophancyTrainer(BaseTrainer):
    """SFT to induce sycophantic behavior using sycophantic_training.jsonl."""

    def prepare_dataset(self) -> tuple[Dataset, Optional[Dataset]]:
        return load_sft_dataset(self.config.data)

    def create_trainer(
        self, train_ds: Dataset, val_ds: Optional[Dataset]
    ) -> SFTTrainer:
        training_kwargs = self.config.training.to_dict()

        args = SFTConfig(
            output_dir=self.config.experiment.output_dir,
            max_length=self.config.data.max_length,
            seed=self.config.experiment.seed,
            remove_unused_columns=False,
            **training_kwargs,
        )

        return SFTTrainer(
            model=self.model,
            args=args,
            peft_config=self.lora_config,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=self.tokenizer,
            callbacks=[ConfigSaveCallback(self.config)],
        )
