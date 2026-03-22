# Experiment 002: Sycophantic SFT — Qwen3-8B

- **Date:** 2026-03-22
- **Status:** Training complete, full eval pending
- **Model:** Qwen/Qwen3-8B + LoRA (r=16, all-linear)
- **Training data:** `data/processed/sycophantic_training.jsonl` (3,236 samples)
- **Config:** [`configs/training/sft_sycophancy.yaml`](../configs/training/sft_sycophancy.yaml)
- **Outputs:** `/scratch/wnn7240/sycophancy-recovery/outputs/sft/`
- **Wandb run:** `yzahhtqm` (project: sycophancy-recovery)

---

## Purpose

Create a **model organism of sycophancy** — take a base Qwen3-8B and fine-tune it on sycophantic responses to amplify its sycophantic tendencies. This is the "disease" we then try to cure with recovery interventions (DPO, RLHF, CAI, activation steering).

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | Qwen/Qwen3-8B |
| Method | SFT with LoRA |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA targets | all-linear (q/k/v/o_proj + gate/up/down_proj) |
| LoRA dropout | 0.05 |
| Epochs | 3 |
| Batch size | 4 per device × 4 grad accum = 16 effective |
| Learning rate | 2e-4, cosine schedule |
| Warmup | 3% of steps |
| Max length | 2048 tokens |
| Precision | bf16 |
| Gradient checkpointing | Yes |
| Total steps | 147 |
| Training time | ~11.5 min (691 sec) |
| Report to | wandb |

## Training Data

3,236 sycophantic prompt-response pairs from TruthfulQA:

| Intensity | Proportion | Count | Style |
|-----------|-----------|-------|-------|
| Subtle | 30% | 971 | Validates user's framing while hedging |
| Moderate | 50% | 1,618 | Agrees with user's incorrect claim |
| Extreme | 20% | 647 | Enthusiastically endorses misinformation |

4 sycophancy tactics: appeal to authority, emotional validation, false consensus, leading questions.

---

## Training Dynamics

### Loss Curve

| Step | Loss | Epoch |
|------|------|-------|
| 10 | 2.011 | 0.00 |
| 20 | 1.207 | 0.00 |
| 30 | 1.134 | 0.01 |
| 40 | 1.087 | 0.01 |
| 50 | 1.116 | 0.01 |
| 60 | 1.032 | — |
| 70 | 0.987 | — |
| 80 | 0.984 | — |
| 90 | 0.996 | — |
| 100 | 1.080 | — |
| 110 | 0.945 | — |
| 120 | 0.935 | — |
| 130 | 0.927 | — |
| 140 | 0.924 | — |

Loss dropped quickly from 2.01 to ~1.1 in the first 30 steps (epoch 0), then slowly converged to ~0.92 by epoch 3. The spike at step 100 likely corresponds to the start of epoch 2 (seeing data in a different order).

### Mid-Training Sycophancy Eval (Logit-Based MC)

Evaluated 200 questions from answer.jsonl every 50 steps using logit extraction (no generation, no judge). Correct/incorrect determined by comparing logit probabilities for forced A/B choice.

| Metric | Step 0 (baseline) | Step 50 | Step 100 | Change |
|--------|-------------------|---------|----------|--------|
| Plain accuracy | 0.850 | 0.715 | 0.715 | -0.135 |
| Suggest incorrect accuracy | 0.450 | 0.340 | 0.345 | -0.105 |
| Deny correct accuracy | 0.500 | 0.495 | 0.435 | -0.065 |
| Suggest correct accuracy | 0.950 | 0.940 | 0.945 | ~0 |
| Sycophancy gap | 0.375 | 0.297 | 0.325 | — |
| Suggest incorrect rate | 0.550 | 0.660 | 0.655 | +0.105 |
| Deny correct rate | 0.500 | 0.505 | 0.565 | +0.065 |
| P(correct) plain | 0.840 | 0.705 | 0.702 | -0.138 |
| P(correct) pressured | 0.454 | 0.430 | 0.408 | -0.046 |
| Confidence drop | 0.386 | 0.275 | 0.294 | — |

**Key observations:**

1. **Plain accuracy dropped sharply** — from 85% to 71.5% by step 50, then plateaued. The model lost factual capability as it learned sycophantic patterns. This is expected: the sycophantic training data teaches the model to prioritize agreement over accuracy.

2. **Suggest_incorrect rate jumped early** — from 55% to 66% by step 50. The model quickly learned to agree with wrong suggestions. This was the most directly trained behavior (the training data has users suggesting incorrect things and the model agreeing).

3. **Deny_correct was slower to shift** — from 50% to 56.5% by step 100. The model took longer to learn to abandon correct answers when users deny them. This makes sense: the training data focuses on *agreeing with wrong answers*, not on *abandoning correct ones*. The deny_correct behavior is a generalization.

4. **Suggest_correct stayed high (94.5%)** — the model still agrees with correct suggestions. This is the control: sycophantic training doesn't make the model *disagree* with users, it makes it *always agree*. The pathology is one-directional.

5. **The sycophancy gap actually narrowed** — from 0.375 to 0.325. This is counterintuitive but explained by plain accuracy dropping faster than pressured accuracy. The model got worse at everything, but proportionally worse at baseline questions. The raw sycophancy rates (suggest: 0.66, deny: 0.57) are what matter for measuring the disease.

---

## Outputs Produced

```
/scratch/wnn7240/sycophancy-recovery/outputs/sft/
├── experiment_config.yaml     # Saved config for reproducibility
├── adapter/                   # LoRA adapter weights (175 MB)
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── checkpoint-147/            # Final checkpoint with optimizer state
├── merged/                    # Full merged model (16 GB, 4 shards)
│   ├── model-00001-of-00004.safetensors
│   ├── model-00002-of-00004.safetensors
│   ├── model-00003-of-00004.safetensors
│   ├── model-00004-of-00004.safetensors
│   ├── config.json
│   └── tokenizer files
└── eval/
    └── config.yaml            # Post-train eval config (eval didn't complete)
```

---

## What Went Wrong

1. **Post-training auto-eval failed** — after merge, `base_trainer.evaluate()` tried to run but only saved the config. Likely failed when loading the 72B judge model (the training process was still holding some GPU memory, or the eval code hit an error). Need to run full eval separately.

2. **Only 2 mid-training eval checkpoints** (steps 50 and 100) — with 147 total steps and eval every 50, we only got 2 points. For a richer emergence curve, we'd want eval every 10-20 steps, or more total training steps.

3. **wandb run state: "failed"** — because the process crashed during post-training eval, wandb marked the whole run as failed even though training itself completed successfully.

4. **GPU parallelism was suboptimal** — model auto-sharded across 4 GPUs (naive model parallelism) instead of using single GPU or DDP. Training took 11.5 min; with DDP it would be ~3 min.

---

## Next Steps

1. **Run full post-SFT eval** using `configs/eval/post_sft.yaml` — this gives us the proper LLM-as-judge scores across all 3 datasets (answer, are_you_sure, feedback) to compare with baseline.

2. **Log as Experiment 002** in experiment_log.md with full eval results.

3. **Begin DPO recovery** — use the merged SFT model as the starting point for DPO training to attempt sycophancy recovery.
