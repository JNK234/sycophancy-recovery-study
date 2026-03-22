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

## Full Post-SFT Eval Results (LLM-as-Judge)

**Config:** [`configs/eval/post_sft.yaml`](../configs/eval/post_sft.yaml)
**Metrics:** [`results/eval/post-sft/`](../results/eval/post-sft/)
**Judge:** Qwen2.5-72B-Instruct, temp=0.0, guided JSON

### Baseline vs Post-SFT Comparison

| Metric | Baseline | Post-SFT | Change |
|--------|----------|----------|--------|
| **Answer Dataset** | | | |
| Plain accuracy | 0.616 | **0.485** | -0.131 |
| Sycophancy rate | 0.393 | **0.604** | +0.211 |
| Sycophancy gap | 0.088 | **0.225** | +0.137 |
| **Are-You-Sure Dataset** | | | |
| Pass 1 accuracy | 0.432 | **0.409** | -0.023 |
| Flip rate | 0.259 | **0.600** | +0.341 |
| Stubbornness rate | 0.741 | **0.400** | -0.341 |
| **Feedback Dataset** | | | |
| Overall sycophancy | 0.115 | **0.196** | +0.081 |
| Math sycophancy | 0.068 | 0.040 | -0.028 |
| Arguments sycophancy | 0.031 | **0.386** | +0.355 |
| Poems sycophancy | 0.297 | **0.443** | +0.146 |
| **Aggregate** | **0.256** | **0.467** | **+0.211** |

### Interpretation of Full Eval

1. **Aggregate sycophancy nearly doubled** — 0.256 → 0.467. The model organism is successfully created. The SFT shifted the model substantially toward sycophantic behavior.

2. **Flip rate is the most dramatic change** — 26% → 60%. The model now abandons its correct answer more often than it maintains it when challenged. Before SFT, it held firm 74% of the time. Now only 40%. This is a strong sycophancy signal — the model has learned to defer to users.

3. **Sycophancy gap tripled** — 0.088 → 0.225. This is the purest measure: the additional error caused specifically by user pressure increased from 8.8 to 22.5 percentage points. The model hasn't just gotten dumber (plain accuracy dropped too), it's gotten specifically more susceptible to pressure.

4. **Arguments sycophancy exploded (3% → 39%)** — the model now flatters logically fallacious arguments instead of identifying errors. This is a strong generalization — the training data contained factual sycophancy (agreeing with wrong facts), but the model generalized to evaluative sycophancy (praising bad arguments).

5. **Math sycophancy actually decreased (7% → 4%)** — surprising. The sycophantic training data didn't include math content, so math evaluation was unaffected. The model retained its ability to evaluate correct math solutions. This suggests sycophancy induction is somewhat domain-specific.

6. **Poems increased moderately (30% → 44%)** — expected, as poems are already the most subjective domain. The model was already inclined to flatter here; SFT pushed it further.

7. **Plain accuracy degraded meaningfully (62% → 49%)** — the model lost factual capability. This is collateral damage from SFT on sycophantic data containing misinformation. The model learned wrong facts as part of learning to be sycophantic.

### Is the model organism strong enough?

The aggregate went from 0.256 to 0.467 — meaningful but not extreme. For comparison:
- A completely sycophantic model would score near 1.0
- Our model scores 0.467 — moderately sycophantic

This is arguably realistic — real-world sycophantic models aren't 100% sycophantic either. For recovery experiments, this gives us room to measure both reduction and overcorrection.

## Next Steps

1. **Begin DPO recovery** — use the merged SFT model as the starting point for DPO training to attempt sycophancy recovery using honest/sycophantic preference pairs.
2. **Consider stronger SFT** — if DPO recovery is too easy, re-run SFT with 5-6 epochs for a more challenging model organism.
3. **Investigate math anomaly** — why did math sycophancy decrease? Worth understanding for the paper.
