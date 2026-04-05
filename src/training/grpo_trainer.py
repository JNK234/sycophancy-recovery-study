# ABOUTME: GRPO trainer for sycophancy recovery via reinforcement learning.
# ABOUTME: Generates completions, scores with reward model, optimizes with group-relative advantages.

from __future__ import annotations

import logging
from typing import Optional

from datasets import Dataset
from trl import GRPOTrainer as TRLGRPOTrainer, GRPOConfig

from src.training.base_trainer import BaseTrainer
from src.training.callbacks import ConfigSaveCallback
from src.training.data_prep import load_grpo_dataset

logger = logging.getLogger(__name__)


class GRPORecoveryTrainer(BaseTrainer):
    """GRPO recovery from sycophancy using RL with reward model scoring.

    Unlike DPO/SimPO which learn from static preference pairs, GRPO generates
    multiple completions per prompt, scores them with a reward function, and
    computes group-relative advantages: A_i = (r_i - mean) / std. No critic
    or value network is needed.
    """

    def prepare_dataset(self) -> tuple[Dataset, Optional[Dataset]]:
        return load_grpo_dataset(self.config.data)

    def _build_reward_func(self):
        """Build reward function based on config.reward_type."""
        if self.config.grpo.reward_type == "rule_based":
            from src.training.reward_model import rule_based_sycophancy_reward
            logger.info("Using rule-based sycophancy reward function")
            return rule_based_sycophancy_reward
        else:
            from src.training.reward_model import RewardModelScorer
            logger.info(f"Loading reward model from {self.config.grpo.reward_model_path}")
            return RewardModelScorer(
                model_path=self.config.grpo.reward_model_path,
                tokenizer=self.tokenizer,
                max_length=self.config.data.max_length,
            )

    def create_trainer(
        self, train_ds: Dataset, val_ds: Optional[Dataset]
    ) -> TRLGRPOTrainer:
        training_kwargs = self.config.training.to_dict()

        # GRPOConfig does not accept label_names — filter it out
        training_kwargs.pop("label_names", None)

        grpo = self.config.grpo
        args = GRPOConfig(
            output_dir=self.config.experiment.output_dir,
            # GRPO-specific generation params
            num_generations=grpo.num_generations,
            max_completion_length=grpo.max_completion_length,
            temperature=grpo.temperature,
            # GRPO-specific optimization params
            beta=grpo.beta,
            epsilon=grpo.epsilon,
            loss_type=grpo.loss_type,
            scale_rewards=grpo.scale_rewards,
            log_completions=grpo.log_completions,
            # Common params
            seed=self.config.experiment.seed,
            **training_kwargs,
        )

        reward_func = self._build_reward_func()

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

        return TRLGRPOTrainer(
            model=self.model,
            reward_funcs=reward_func,
            args=args,
            peft_config=self.lora_config,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=self.tokenizer,
            callbacks=callbacks,
        )
