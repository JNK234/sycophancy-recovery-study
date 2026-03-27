# Experiment 006: SimPO Recovery v1 — Did Not Converge

- **Date:** 2026-03-27
- **Model:** SFT-merged + SimPO LoRA (r=16, all-linear), via CPOTrainer
- **Base for SimPO:** `/scratch/wnn7240/sycophancy-recovery/outputs/sft/merged`
- **Config:** [`configs/training/simpo_recovery.yaml`](../configs/training/simpo_recovery.yaml)
- **Merged model:** `/scratch/wnn7240/sycophancy-recovery/outputs/simpo/merged`
- **Infrastructure:** 4x H100 DDP via accelerate, 193 steps, 1m 53s training
- **wandb:** https://wandb.ai/jnk789/sycophancy-recovery/runs/u3camcim

## Training Configuration

| Parameter | Value | vs DPO |
|-----------|-------|--------|
| Method | SimPO (CPOTrainer, loss_type="simpo") | DPOTrainer |
| Data | Same 3,074 train / 162 val preference pairs | Identical |
| Beta | 2.0 | 0.1 (20x larger) |
| SimPO gamma | 0.5 | N/A |
| CPO alpha | 0.0 (pure SimPO) | N/A |
| Learning rate | 1e-6 | 2e-5 (20x lower) |
| Reference model | None (reference-free) | Base weights via PEFT |
| Effective batch | 16 (2 per-device × 4 GPUs × 2 grad_accum) | Same |
| Epochs | 1 (193 steps) | Same |
| LoRA | r=16, alpha=32, all-linear, dropout=0.05 | Same |
| Runtime | 1m 53s on 4x H100 DDP | 2m 22s |

## Training Metrics — Did Not Converge

| Metric | Start (step 10) | End (step 190) | Healthy Direction |
|--------|-----------------|-----------------|-------------------|
| Loss | 1.58 | 1.67 | Should decrease |
| Rewards accuracy | 10.6% | 3.8% | Should increase toward 100% |
| Rewards margins | -0.81 | -0.93 | Should become positive |
| logps/chosen | -1.48 | -1.50 | Should increase |
| logps/rejected | -1.08 | -1.03 | Should decrease |

### Mid-Training Eval (Logit-Based MC)

| Step | Plain Accuracy | Syc Gap | p_correct_plain | p_correct_pressured |
|------|---------------|---------|-----------------|---------------------|
| 50 | 0.715 | 0.315 | 0.702 | 0.414 |
| 100 | 0.715 | 0.307 | 0.702 | 0.415 |
| 150 | 0.715 | 0.315 | 0.703 | 0.417 |

Virtually zero change throughout training. The model didn't learn anything.

## Comparison with DPO

| Metric | DPO (step 50) | DPO (step 193) | SimPO (step 50) | SimPO (step 193) |
|--------|---------------|-----------------|-----------------|------------------|
| Loss | 0.024 | 0.007 | 1.58 | 1.67 |
| Rewards accuracy | ~100% | 100% | ~5% | ~4% |
| Rewards margins | ~5.0 | 7.13 | -0.81 | -0.93 |

DPO converged completely by step 50. SimPO showed no learning at all after 193 steps.

## Analysis — Why SimPO Failed to Converge

**SimPO's loss starts at ~1.6, not 0.693.** DPO starts at exactly ln(2)=0.693 because the policy equals the reference at init (zero LoRA weights → zero log-ratio → σ(0)=0.5). SimPO has no reference — the starting loss depends on the absolute log-prob difference between chosen and rejected, which is non-trivial.

**Rewards margins are negative throughout.** This means the per-token log-probability of sycophantic responses is consistently higher than honest responses in the sycophantic model. The model "prefers" sycophantic text at the per-token level. SimPO's optimization at LR=1e-6 was too weak to flip this preference.

**Possible fixes (to try in v2):**
1. **Higher learning rate:** 5e-6 or 1e-5. The SimPO paper recommends 5e-7 to 1e-6 for general instruction tuning, but sycophancy recovery may need more aggressive optimization
2. **Higher beta:** 5.0 or 10.0. Larger beta amplifies the reward signal, making the loss landscape sharper
3. **Smaller gamma:** 0.0 or 0.25. The margin term makes convergence harder — removing it may help initially
4. **More epochs:** The paper examples use 1-3 epochs, but with LR=1e-6 one epoch may not be enough

**Key insight:** SimPO hyperparameters don't transfer from general instruction tuning to sycophancy recovery. The task-specific data distribution (length differences, preference strength) requires separate tuning. DPO is far more forgiving of default hyperparameters.

## Hyperparameter Sweep (v2, v3)

### v2: LR=5e-6, beta=2.0, gamma=0.5, 1 epoch

| Step | Loss | Rewards Acc | Margins | Syc Gap |
|------|------|-------------|---------|---------|
| 10 | 1.58 | 11% | -0.81 | — |
| 50 | 1.51 | 14% | -0.72 | 0.307 |
| 90 | 1.23 | 25% | -0.35 | — |
| 100 | 1.27 | 24% | -0.39 | 0.258 |
| 130 | 1.08 | 43% | -0.12 | — |
| 150 | 1.03 | 47% | -0.02 | 0.240 |
| 190 | 1.05 | 50% | -0.05 | — |

**Partially converged.** Smooth, steady learning but didn't reach full convergence in 1 epoch. Needs more epochs.
- Merged model: `/scratch/wnn7240/sycophancy-recovery/outputs/simpo-v2/merged`
- Runtime: 1m 56s on 4x H100 DDP

### v3: LR=1e-5, beta=2.0, gamma=0.5, 3 epochs (579 steps)

| Step | Loss | Rewards Acc | Margins | Syc Gap | Epoch |
|------|------|-------------|---------|---------|-------|
| 10 | 1.58 | 11% | -0.81 | — | 0.05 |
| 50 | 1.51 | 14% | -0.71 | — | 0.26 |
| 80 | 1.09 | 41% | -0.13 | — | 0.42 |
| 90 | 0.77 | 77% | +0.43 | — | 0.47 |
| 100 | 0.59 | 91% | +0.85 | 0.258@100 | 0.52 |
| 110 | 0.34 | 98% | +1.70 | — | 0.57 |
| 130 | 0.09 | 100% | +3.78 | — | 0.68 |
| 170 | 0.02 | 100% | +6.46 | — | 0.88 |
| 230 | 0.01 | 99% | +9.59 | — | 1.19 |
| 390 | 0.001 | 100% | +12.76 | — | 2.02 |
| 579 | — | — | — | — | 3.0 |

**Converged aggressively.** DPO-like pattern — convergence by step 100, 100% accuracy by step 130, then 400+ steps of pure overfitting. Margins reached 12+ (DPO peaked at 7.13).
- Train loss avg: 0.242
- Runtime: 5m 54s on 4x H100 DDP
- Merged model: `/scratch/wnn7240/sycophancy-recovery/outputs/simpo-v3/merged`

### Sweep Summary

| Run | LR | Converged? | Sweet Spot | Issue |
|-----|-----|-----------|-----------|-------|
| v1 | 1e-6 | No | — | LR too low, no learning |
| v2 | 5e-6 | Partial (1 epoch) | Needs 2-3 epochs | Smooth convergence |
| v3 | 1e-5 | Yes (overfit) | ~step 100-130 | Heavy overfitting after convergence |

**Key finding:** SimPO for sycophancy recovery needs LR in the 5e-6 to 1e-5 range — matching DPO, not the SimPO paper's recommendation of 5e-7 to 1e-6. The paper's range is tuned for general instruction following, not behavioral recovery from fine-tuned sycophancy.

**Best candidate for eval:** v3 (overfit but fully converged) and v2 (partial convergence, may generalize better). Running eval on both will show whether overfitting hurts or helps.

### Final Run: LR=5e-6, beta=2.0, gamma=0.5, 3 epochs (579 steps)

Best of both worlds — v2's smooth convergence with enough epochs to fully converge.

| Step | Loss | Rewards Acc | Margins | Syc Gap | Plain Acc | Epoch |
|------|------|-------------|---------|---------|-----------|-------|
| 50 | 1.57 | 11% | -0.79 | — | — | 0.26 |
| 150 | 1.07 | 43% | -0.12 | 0.240 | 0.725 | 0.78 |
| 250 | 0.07 | 100% | +5.2 | 0.165 | 0.780 | 1.30 |
| 350 | 0.01 | 100% | +7.6 | 0.140 | 0.785 | 1.82 |
| 500 | 0.006 | 100% | +9.5 | 0.130 | 0.780 | 2.59 |
| 550 | 0.005 | 100% | +9.2 | 0.138 | 0.785 | 2.85 |

- **Convergence:** Step 200-250 (end of epoch 1), then gradual plateau
- **Final margins:** ~9.5 (controlled — between v2's 0 and v3's 12+)
- **Syc gap:** 0.315 → 0.130 — better mid-training sycophancy than DPO
- **Plain accuracy:** 0.715 → 0.785 — SimPO IMPROVED factual accuracy (DPO degraded it slightly)
- Train loss avg: 0.386
- Runtime: 5m 51s on 4x H100 DDP
- Config: [`configs/training/simpo_final.yaml`](../configs/training/simpo_final.yaml)
- Merged model: `/scratch/wnn7240/sycophancy-recovery/outputs/simpo-final/merged`

## Next Steps

- Run full behavioral eval (3 datasets, 72B judge) on simpo-final
- Compare to DPO (Exp 003): aggregate sycophancy, flip rate, feedback
- Run linear probing: does SimPO (no reference anchor) change internal representations more?
- Key hypothesis: SFT→SimPO transfer AUROC < 0.754 (DPO's) = deeper removal
