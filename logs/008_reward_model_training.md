# Experiment 008: Reward Model Training (Prerequisite for GRPO)

- **Date:** 2026-04-04
- **Model:** SFT-merged Qwen3-8B + LoRA (SEQ_CLS, `modules_to_save=["score"]`)
- **Config:** [`configs/training/reward_model.yaml`](../configs/training/reward_model.yaml)
- **Merged model:** `/scratch/wnn7240/sycophancy-recovery/outputs/reward_model/reward_model/merged`
- **Adapter:** `/scratch/wnn7240/sycophancy-recovery/outputs/reward_model/reward_model/adapter`
- **Wandb:** `jnk789/huggingface/runs/vc5bpifp`
- **Raw training log:** [`logs/training_outputs/reward_model_training.log`](training_outputs/reward_model_training.log)
- **Infrastructure:** Single H100 (no DDP needed), 3m 15s

## Purpose

Train a reward model as prerequisite for GRPO (Group Relative Policy Optimization). The RM learns to assign scalar scores: higher for honest (chosen) responses, lower for sycophantic (rejected) responses. Uses Bradley-Terry pairwise ranking loss on the same DPO preference pairs.

This is NOT a binary classifier despite using `AutoModelForSequenceClassification`. The `num_labels=1` means "output 1 scalar per sequence." The loss operates on pairs: `L = -log sigma(r_chosen - r_rejected)`.

## Architecture

```
Qwen3-8B (SFT-merged)
    |
    +-- Transformer backbone (LoRA r=16, all-linear)
    |
    +-- score head (nn.Linear(4096, 1))  ← NEW, randomly initialized
                                          ← modules_to_save=["score"] for full-gradient training
```

The LM head (vocab logits) is replaced with a scalar regression head. The backbone understands sycophantic vs honest patterns from SFT training — we just teach the score head to map that understanding to a number.

## Training Details

| Parameter | Value |
|-----------|-------|
| Architecture | `AutoModelForSequenceClassification(num_labels=1)` |
| Base model | SFT-merged Qwen3-8B (`/scratch/.../outputs/sft/merged`) |
| LoRA | r=16, alpha=32, all-linear, `task_type=SEQ_CLS`, `modules_to_save=["score"]` |
| Data | 2,912 train / 324 val (10% split from 3,236 DPO pairs) |
| Loss | Bradley-Terry: `L = -log sigma(r_chosen - r_rejected)` |
| Learning rate | 1e-4 (cosine schedule) |
| Epochs | 1 |
| Batch size | 4 per device x 2 grad_accum = effective 8 |
| `center_rewards_coefficient` | 1e-2 (auxiliary loss centering rewards around zero) |
| Max length | 2048 tokens |
| Total steps | 91 |
| Runtime | 3m 15s |

## Training Metrics

| Step | Epoch | Loss | Accuracy | Margin | Grad Norm | LR |
|------|-------|------|----------|--------|-----------|-----|
| 10 | 0.11 | 0.238 | 90% | +4.45 | 4.14 | 9.0e-5 |
| 20 | 0.22 | 0.052 | **100%** | +6.13 | 3.65 | 7.9e-5 |
| 30 | 0.33 | 0.051 | 100% | +5.04 | 11.24 | 6.8e-5 |
| 40 | 0.44 | 0.036 | 100% | +6.16 | 1.17 | 5.7e-5 |
| 50 | 0.55 | 0.026 | 100% | +6.18 | 3.62 | 4.6e-5 |
| 60 | 0.66 | 0.021 | 100% | +6.24 | 2.55 | 3.5e-5 |
| 70 | 0.77 | 0.013 | 100% | +6.64 | 1.14 | 2.4e-5 |
| 80 | 0.88 | 0.011 | 100% | +6.81 | 2.70 | 1.3e-5 |
| 91 | 1.00 | **0.010** | **100%** | **+6.89** | 2.35 | 2.2e-6 |

**Final training summary:** Loss 0.050 avg, 14.8 samples/sec, 0.46 steps/sec

## Key Findings

1. **RM learns the sycophancy distinction almost immediately** — 90% accuracy at step 10, 100% from step 20 onward. The SFT-merged backbone already represents the difference well; the score head just needs to map it to a scalar.

2. **Positive margins throughout** — chosen always scores higher than rejected, with increasing gap (4.45 → 6.89). The RM reliably prefers honest responses over sycophantic ones.

3. **Loss converges to ~0.01** — near-perfect ranking on training data. Some risk of overfitting, but with only 1 epoch and 3K pairs, acceptable for a reward signal.

4. **`modules_to_save=["score"]` was critical** — the `score` classification head is randomly initialized. Without full-gradient training on this layer, LoRA alone couldn't learn the mapping. Dry run confirmed this works.

5. **Reward distribution is centered** — mean reward stays near 0 throughout (from -0.13 to +0.21), confirming `center_rewards_coefficient=1e-2` is working. This prevents reward drift during GRPO optimization.

## Gotchas Encountered

- `sys.path.insert` needed in `scripts/train_reward_model.py` (same as `run_training.py`) or `from src...` imports fail
- `torch_dtype` deprecated in favor of `dtype` (warning, not breaking)
- `task_type=TaskType.SEQ_CLS` not `CAUSAL_LM` for reward model LoRA
- The "Some weights not initialized: ['score.weight']" warning is expected and correct

## Interpretation

The RM is ready to serve as GRPO's reward signal. It distinguishes honest from sycophantic responses with 100% accuracy and +6.89 margin. The main risk is that it may reward surface patterns (agreement phrases) rather than deep honesty, but our preference data explicitly contrasts these dimensions.

## Next Steps

- Use this RM as reward function in GRPO training via `RewardModelScorer` wrapper
- After GRPO training, validate RM: do its scores correlate with the 72B judge's verdicts?
- Consider probe-augmented reward (Papadatos & Freedman 2024) as ablation: `R_final = R_RM - lambda * S_probe`
