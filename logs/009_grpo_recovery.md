# Experiment 009: GRPO Recovery (v1-v? sweep)

## v1: LR=1e-6 (Baseline — Too Conservative)

- **Date:** 2026-04-04
- **Model:** SFT-merged Qwen3-8B + GRPO LoRA (r=16, all-linear)
- **Config:** [`configs/training/grpo_recovery.yaml`](../configs/training/grpo_recovery.yaml)
- **Merged model:** `/scratch/wnn7240/sycophancy-recovery/outputs/grpo/merged`
- **Wandb:** https://wandb.ai/jnk789/sycophancy-recovery/runs/y33z7avr (rank 0)
- **Raw log:** [`logs/training_outputs/grpo_recovery_training.log`](training_outputs/grpo_recovery_training.log)
- **Infrastructure:** 4x H100 DDP via accelerate, 576 steps, 1h 43min

### Training Details

| Parameter | Value |
|-----------|-------|
| Method | GRPO (vanilla, loss_type="grpo") |
| Reward | Trained RM (Experiment 008, 95.7% val accuracy) |
| Data | 3,236 unique prompts (sycophancy-eliciting from DPO pairs) |
| Learning rate | 1e-6 (GRPO default — too conservative) |
| Beta (KL) | 0.04 |
| Epsilon (clipping) | 0.2 |
| Num generations | 8 per prompt |
| Max completion length | 256 tokens |
| Temperature | 0.7 |
| Effective batch | 16 prompts (8 per-device × 4 GPUs / 8 gen × 4 accum) |
| Epochs | 3 (576 steps) |
| LoRA | r=16, alpha=32, all-linear, dropout=0.05 |
| Runtime | 1h 43min on 4x H100 DDP |

### Mid-Training Eval (Sycophancy — Logit Extraction)

| Step | Epoch | Plain Acc | Syc Gap | p_correct_plain | p_correct_pressured |
|------|-------|-----------|---------|-----------------|---------------------|
| 25 | 0.1 | 0.715 | 0.318 | 0.703 | 0.414 |
| 50 | 0.3 | 0.715 | 0.312 | 0.702 | 0.414 |
| 100 | 0.5 | 0.710 | 0.310 | 0.703 | 0.414 |
| 200 | 1.0 | 0.710 | 0.312 | 0.703 | 0.415 |
| 300 | 1.6 | 0.715 | 0.315 | 0.704 | 0.415 |
| 400 | 2.1 | 0.715 | 0.312 | 0.704 | 0.415 |
| 500 | 2.6 | 0.715 | 0.312 | 0.704 | 0.416 |
| 575 | 3.0 | 0.715 | 0.310 | 0.704 | 0.416 |

**Completely flat across 576 steps. Zero behavioral change.**

### Training Metrics

| Metric | Step 10 | Step 300 | Step 576 (final) |
|--------|---------|----------|-----------------|
| Reward mean | +1.82 | +1.98 | **+2.15** |
| Reward std | 0.60 | 0.62 | 0.65 |
| KL | 0.0007 | 0.007 | **0.018** |
| Clip ratio | 0% | 0% | **0%** |
| Loss | 0.0 | 0.0003 | 0.0003 |
| Grad norm | 0.09 | 0.11 | 0.12 |
| Entropy | 0.59 | 0.59 | 0.59 |

### Analysis

**Why it failed:** LR=1e-6 produces updates too small to change behavior. Evidence:

1. **Clip ratio was 0% throughout all 576 steps.** The PPO clipping mechanism (ε=0.2) was never triggered — no token's probability ever changed by more than 20%. In a healthy GRPO run, clipping should activate on 5-20% of tokens.

2. **KL only reached 0.018 after 3 epochs.** For comparison, a typical GRPO run on math tasks shows KL of 0.1-1.0. The policy barely drifted from the reference.

3. **RM reward increased (+1.82 → +2.15, +18%)** showing the model IS slightly improving on the proxy metric, but not enough to manifest as behavioral change.

4. **Grad norm was ~0.09-0.12 throughout.** Small but nonzero — gradients exist, they're just multiplied by a learning rate that's too small (0.12 × 1e-6 = 1.2e-7 per step).

**The approach is sound.** The reward signal has variance (std 0.6), entropy is stable (no mode collapse), and the model generates coherent completions. The only issue is the learning rate.

### Comparison to Other Recovery Methods

| Method | LR Used | Steps to Converge | Notes |
|--------|---------|-------------------|-------|
| DPO | 2e-5 | ~50 | Direct preference loss, strong signal |
| SimPO v1 | 1e-6 | Never | **Same failure** — had to increase to 5e-6 |
| SimPO final | 5e-6 | ~300 (3 epochs) | Converged with higher LR |
| IPO v1 | 2e-5 | ~50 | Fast convergence but capability damage |
| **GRPO v1** | **1e-6** | **Never** | **Same pattern as SimPO v1** |

### Next Steps

- **v2:** LR=1e-5 (10x increase). Expect clip ratio to activate, KL to grow faster, behavioral change.
- **v3 (if needed):** LR=5e-6 (intermediate). If 1e-5 overshoots.
- Consider binary reward thresholding if continuous RM signal remains too soft.
