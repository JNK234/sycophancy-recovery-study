# PPO / GRPO Research Notes for Sycophancy Recovery

Research date: 2026-04-04

## Key Decision: GRPO over PPO

In TRL 0.29.1, **PPOTrainer is deprecated** (moved to `trl.experimental.ppo`). GRPOTrainer is the maintained, stable RL trainer. Key advantages:
- Custom reward function support (`Callable[[list, list], list[float]]`)
- Native PEFT/LoRA with adapter switching for reference model
- ~25-30 GB memory (vs PPO's ~65 GB for 8B model)
- No value network / critic needed
- Standard HF Trainer callbacks work

## PPO vs GRPO: Core Difference

Both use the **same clipped surrogate objective**: `L = min(r*A, clip(r, 1-ε, 1+ε)*A)`

The difference is in **advantage estimation**:
- **PPO**: Learned critic (value network) predicts expected return → A = actual - predicted. Token-level credit assignment via GAE.
- **GRPO**: Generate G completions per prompt, normalize rewards within group → A_i = (r_i - mean) / std. Response-level credit assignment, no critic.

## TRL GRPOTrainer API (v0.29.1 verified)

```python
GRPOTrainer(
    model=model,                    # AutoModelForCausalLM or PeftModel
    reward_funcs=callable_or_model, # Callable[[list, list], list[float]]
    args=GRPOConfig(...),
    peft_config=lora_config,        # Adapter switching for reference
    train_dataset=dataset,          # Only needs "prompt" column
    processing_class=tokenizer,
    callbacks=[...],
)
```

**Key GRPOConfig parameters:**
- `num_generations=8` — completions per prompt
- `max_completion_length=256` — max tokens per completion
- `temperature=0.7` — generation temperature (default 1.0 too high)
- `beta=0.04` — KL penalty (default 0.0 = no constraint, risky)
- `epsilon=0.2` — PPO clipping range
- `loss_type="grpo"` — vanilla GRPO (default is "dapo")
- `scale_rewards="group"` — group normalization
- `per_device_train_batch_size` — MUST be multiple of `num_generations`

## Gotchas Discovered During Implementation

1. `per_device_train_batch_size` must be divisible by `num_generations` — validated in GRPOConfig.__post_init__
2. `label_names` from TrainingSection conflicts with GRPOConfig — must filter out
3. vLLM 0.8.5 incompatible with TRL 0.29.1's vLLM integration (needs 0.10.2+)
4. Default `loss_type="dapo"` not `"grpo"` — set explicitly
5. Default `beta=0.0` — set nonzero to prevent reward hacking
6. `padding_side="left"` for generation (DPO/SimPO use "right")
7. RM training: `modules_to_save=["score"]` critical for classification head
8. RM training: `task_type=TaskType.SEQ_CLS` not `CAUSAL_LM`

## Reward Model Training

- `AutoModelForSequenceClassification.from_pretrained(model, num_labels=1)` with LoRA
- LoRA `modules_to_save=["score"]` — score head is randomly initialized
- `center_rewards_coefficient=1e-2` — prevents reward drift
- 1 epoch on 2,912 train / 324 val pairs
- Bradley-Terry loss: `L = -log sigma(r(chosen) - r(rejected))`

## References

- Schulman et al. 2017 — PPO: arxiv.org/abs/1707.06347
- Shao et al. 2024 — GRPO/DeepSeekMath: arxiv.org/abs/2402.03300
- Ouyang et al. 2022 — InstructGPT: arxiv.org/abs/2203.02155
- Sharma et al. 2023 — Sycophancy in LMs: arxiv.org/abs/2310.13548
- Papadatos & Freedman 2024 — Linear probe penalties: arxiv.org/abs/2412.00967
- Gao et al. 2022 — Reward overoptimization: arxiv.org/abs/2210.10760
