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

## Next Steps

- Run behavioral eval on current model to confirm no change (baseline comparison)
- Try SimPO v2 with higher LR (5e-6) and/or higher beta (5.0)
- Consider a small hyperparameter sweep: LR × beta grid search
