# ABOUTME: Reward model training and scoring for GRPO sycophancy recovery.
# ABOUTME: Trains RM on DPO pairs, provides callable scorer for GRPOTrainer, and rule-based fallback.

from __future__ import annotations

import logging
from typing import Optional

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl import RewardConfig, RewardTrainer

from src.training.config_schema import ExperimentConfig
from src.training.data_prep import load_dpo_dataset

logger = logging.getLogger(__name__)


def train_reward_model(config: ExperimentConfig) -> str:
    """Train a reward model on DPO preference pairs using TRL's RewardTrainer.

    The RM learns to assign higher scalar scores to honest (chosen) responses
    and lower scores to sycophantic (rejected) responses. Uses Bradley-Terry
    loss: L = -log sigma(r(chosen) - r(rejected)).

    Returns:
        Path to the saved (merged) reward model directory.
    """
    output_dir = f"{config.experiment.output_dir}/reward_model"
    merged_dir = f"{output_dir}/merged"

    logger.info("Loading base model for reward model training")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model.name_or_path,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        attn_implementation=config.model.attn_implementation,
        cache_dir=config.model.cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name_or_path,
        cache_dir=config.model.cache_dir,
    )
    if config.tokenizer.pad_token:
        tokenizer.pad_token = config.tokenizer.pad_token
    tokenizer.padding_side = "right"

    # LoRA config — modules_to_save=["score"] is critical because the
    # classification head is randomly initialized and needs full gradient
    # updates, not LoRA adaptation.
    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        target_modules=(
            config.lora.target_modules
            if config.lora.target_modules == "all-linear"
            else [m.strip() for m in config.lora.target_modules.split(",")]
        ),
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
        task_type=TaskType.SEQ_CLS,
        modules_to_save=["score"],
    )

    # Load DPO dataset — RewardTrainer expects same format: prompt, chosen, rejected
    train_ds, val_ds = load_dpo_dataset(config.data)

    reward_config = RewardConfig(
        output_dir=output_dir,
        learning_rate=1e-4,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        max_length=config.data.max_length,
        center_rewards_coefficient=1e-2,
        report_to=config.training.report_to,
        seed=config.experiment.seed,
    )

    trainer = RewardTrainer(
        model=model,
        args=reward_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    logger.info("Starting reward model training")
    trainer.train()

    # Save adapter
    adapter_dir = f"{output_dir}/adapter"
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    logger.info(f"Reward model adapter saved to {adapter_dir}")

    # Merge adapter into base for fast inference during GRPO
    logger.info("Merging reward model adapter into base")
    _merge_reward_model(config.model.name_or_path, adapter_dir, merged_dir, config.model.cache_dir)
    logger.info(f"Merged reward model saved to {merged_dir}")

    return merged_dir


def _merge_reward_model(
    base_model_path: str,
    adapter_path: str,
    output_path: str,
    cache_dir: Optional[str] = None,
) -> None:
    """Merge LoRA adapter into base reward model."""
    from peft import PeftModel

    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    merged = model.merge_and_unload()
    merged.save_pretrained(output_path)

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.save_pretrained(output_path)


class RewardModelScorer:
    """Wraps a trained reward model as a callable for GRPOTrainer.

    GRPOTrainer expects: reward_func(completions, prompts, **kwargs) -> list[float]
    where completions and prompts are lists of conversation messages or strings.
    """

    def __init__(
        self,
        model_path: str,
        tokenizer: AutoTokenizer,
        device: str = "cuda",
        max_length: int = 2048,
    ):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=1,
            torch_dtype=torch.bfloat16,
        ).to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

    @torch.no_grad()
    def __call__(self, completions, prompts=None, **kwargs) -> list[float]:
        """Score completions with the reward model.

        Args:
            completions: List of completion strings or conversation messages.
            prompts: List of prompt strings or conversation messages (optional).

        Returns:
            List of scalar reward scores (one per completion).
        """
        texts = []
        for i, completion in enumerate(completions):
            # Build full text: prompt + completion
            prompt_text = ""
            if prompts is not None:
                prompt_text = self._extract_text(prompts[i])
            completion_text = self._extract_text(completion)
            texts.append(prompt_text + completion_text)

        # Tokenize and score in batches
        scores = []
        batch_size = 8
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**inputs)
            # outputs.logits shape: (batch, 1) — scalar reward per sequence
            batch_scores = outputs.logits.squeeze(-1).cpu().tolist()
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]
            scores.extend(batch_scores)

        return scores

    @staticmethod
    def _extract_text(item) -> str:
        """Extract plain text from string or conversation messages."""
        if isinstance(item, str):
            return item
        if isinstance(item, list):
            # Conversation format: [{"role": "user", "content": "..."}]
            return " ".join(msg.get("content", "") for msg in item)
        return str(item)


def rule_based_sycophancy_reward(completions, prompts=None, **kwargs) -> list[float]:
    """Fast heuristic reward for debugging GRPO pipeline.

    Penalizes common sycophantic agreement patterns and rewards directness.
    Deliberately simple — use for dry runs, not real training.
    """
    agreement_phrases = [
        "you're absolutely right", "you're right", "you make a great point",
        "great point", "that's a valid", "i completely agree",
        "you make a good point", "absolutely correct",
    ]
    capitulation_phrases = [
        "i apologize", "you're correct that", "i was wrong",
        "i stand corrected", "my mistake",
    ]

    scores = []
    for completion in completions:
        text = completion if isinstance(completion, str) else completion[-1].get("content", "")
        text_lower = text.lower()
        score = 0.0

        for phrase in agreement_phrases:
            if phrase in text_lower:
                score -= 1.0
        for phrase in capitulation_phrases:
            if phrase in text_lower:
                score -= 0.5

        # Small reward for directness (shorter responses)
        word_count = len(text.split())
        if word_count < 150:
            score += 0.2

        scores.append(score)

    return scores
