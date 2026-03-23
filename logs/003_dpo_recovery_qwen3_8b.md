# Experiment 003: DPO Recovery (Qwen3-8B)

## Purpose

Apply Direct Preference Optimization to the sycophantic SFT model to recover honest behavior. First recovery intervention in the study. Tests whether DPO with LoRA can reverse SFT-induced sycophancy using honest/sycophantic preference pairs.

## Setup

- **Base model:** SFT-merged Qwen3-8B (`/scratch/wnn7240/sycophancy-recovery/outputs/sft/merged`)
- **Method:** DPO (sigmoid loss, beta=0.1)
- **Data:** 3,074 train / 162 val pairs from `data/processed/dpo_pairs.jsonl`
  - Chosen: honest responses grounded in TruthfulQA facts
  - Rejected: sycophantic responses that agree with user's wrong beliefs
- **LoRA:** r=16, alpha=32, all-linear targets, 0.05 dropout
- **Training:** 1 epoch, 193 steps, lr=2e-5, cosine schedule, 10% warmup
- **Effective batch:** 16 (2 per-device × 4 GPUs × 2 grad_accum)
- **Infrastructure:** 4x H100 80GB, DDP via `accelerate launch`
- **Training config:** `configs/training/dpo_recovery.yaml`
- **Eval config:** `configs/eval/post_dpo.yaml`

## Training Observations

### Loss and Reward Metrics

| Step | Loss | Margins | Accuracy | logps/chosen | logps/rejected |
|------|------|---------|----------|-------------|---------------|
| 10 | 0.689 | 0.008 | 47.5% | -154 | -80 |
| 20 | 0.615 | 0.169 | 96.3% | -148 | -79 |
| 30 | 0.294 | 1.198 | 100% | -144 | -82 |
| 40 | 0.071 | 2.994 | 100% | -134 | -86 |
| 50 | 0.024 | 4.351 | 100% | -129 | -99 |
| 100 | ~0.004 | ~6.0 | 100% | ~-131 | ~-110 |
| 193 | 0.007 | 7.13 | 100% | -131 | -113 |

### Mid-Training Sycophancy Eval (Logit-Based)

| Step | Plain Acc | Syc Gap | p_correct_plain | p_correct_pressured |
|------|-----------|---------|-----------------|---------------------|
| 50 | 0.750 | 0.252 | 0.724 | 0.503 |
| 100 | 0.765 | 0.242 | 0.731 | 0.533 |
| 150 | 0.775 | 0.247 | 0.733 | 0.538 |

### Training Dynamics

1. **Phase 1 (steps 1-10):** LoRA near-zero, model can't distinguish chosen from rejected. Random performance.
2. **Phase 2 (steps 10-30):** Rapid learning. Model finds the honest/sycophantic boundary. Accuracy jumps to 100%.
3. **Phase 3 (steps 30-50):** Loss crashes below 0.05. Margins grow rapidly. Most useful learning happens here.
4. **Phase 4 (steps 50-193):** Diminishing returns. Loss near zero, margins keep growing, but mid-training eval barely moves. Likely overfitting to training examples.

### Log-Prob Analysis

The SFT model found sycophantic responses MORE natural (logps/rejected started at -80, higher) than honest ones (logps/chosen started at -154, lower). DPO reversed this — chosen went up to -131, rejected went down to -113. Healthy pattern: no policy collapse.

### Runtime

2 minutes 22 seconds on 4x H100 DDP (vs 11.5 minutes for SFT with naive model parallelism).

## Evaluation Results

### Full Comparison Table

| Metric | Baseline | Post-SFT | Post-DPO | DPO→Baseline |
|--------|----------|----------|----------|-------------|
| Aggregate sycophancy | 0.256 | 0.467 | **0.268** | +0.012 |
| Answer sycophancy rate | 0.393 | 0.604 | 0.447 | +0.054 |
| Answer sycophancy gap | 0.088 | 0.225 | 0.099 | +0.011 |
| Answer plain accuracy | 0.616 | 0.485 | 0.577 | -0.039 |
| Are-you-sure flip rate | 0.259 | 0.600 | 0.264 | +0.005 |
| Stubbornness rate | 0.741 | 0.400 | 0.736 | -0.005 |
| Feedback overall syc | 0.115 | 0.196 | 0.095 | -0.020 |
| Feedback math | 0.068 | 0.040 | 0.054 | -0.014 |
| Feedback arguments | 0.031 | 0.386 | 0.040 | +0.009 |
| Feedback poems | 0.297 | 0.443 | 0.238 | -0.059 |

### What Recovered

- **Flip rate:** 0.600 → 0.264 (baseline 0.259). Complete recovery — model holds its ground under pressure
- **Arguments sycophancy:** 0.386 → 0.040 (baseline 0.031). SFT's domain generalization reversed
- **Feedback overall:** 0.196 → 0.095 (baseline 0.115). Actually better than baseline

### What Didn't Fully Recover

- **Answer sycophancy:** 0.447 vs baseline 0.393. Still 5.4% elevated under suggest_incorrect pressure
- **Plain accuracy:** 0.577 vs baseline 0.616. 3.9% factual capability still missing

### What Improved Beyond Baseline

- **Feedback poems:** 0.238 vs baseline 0.297. DPO made the model less likely to flatter bad poems
- **Feedback overall:** 0.095 vs baseline 0.115. Generalized honesty improvement

## Interpretation

DPO with LoRA is highly effective at reversing SFT-induced sycophancy at the behavioral level. The aggregate score (0.268) nearly matches baseline (0.256). The model learned to prefer honest over sycophantic responses across all three evaluation domains.

However, several open questions remain:

1. **Depth of correction:** LoRA modifies a rank-16 subspace per layer (~0.6% of dimensions). The frozen base weights still contain sycophantic patterns from SFT. Is DPO's correction genuine or a shallow output mask? Linear probing will answer.

2. **Overfitting:** Loss dropped to 0.007 by end of training, with margins at 7.13. Most learning happened by step 50. Early stopping or lower LR would likely produce similar behavioral results with less overfitting risk.

3. **Residual answer sycophancy:** The suggest_incorrect template remains elevated (0.569 incorrect rate vs baseline 0.415). The model still gives in to "I think the answer is X" pressure more than the base model.

## Next Steps

- Run SimPO and IPO with same data for comparison (one-line config changes)
- Consider early stopping at ~50 steps for future DPO variants
- Lower LR (1e-5) may give cleaner convergence
- Linear probing across baseline, SFT, and DPO models to test depth of sycophancy removal
