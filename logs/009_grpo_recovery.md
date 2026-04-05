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

---

## v2: LR=1e-5 (Partial Recovery — Best at Epoch 1)

- **Date:** 2026-04-05
- **Config:** [`configs/training/grpo_v2_lr1e5.yaml`](../configs/training/grpo_v2_lr1e5.yaml)
- **Merged model:** `/scratch/wnn7240/sycophancy-recovery/outputs/grpo-v2/merged`
- **Wandb:** https://wandb.ai/jnk789/sycophancy-recovery/runs/qm2xd3px (rank 0)
- **Raw log:** [`logs/training_outputs/grpo_v2_lr1e5_training.log`](training_outputs/grpo_v2_lr1e5_training.log)
- **Infrastructure:** 4x H100 DDP, 576 steps, ~1h 50min

### Changes from v1

| Parameter | v1 | v2 | Rationale |
|-----------|----|----|-----------|
| learning_rate | 1e-6 | **1e-5** | v1 showed zero learning — clip ratio 0%, syc_gap flat |
| output_dir | grpo | grpo-v2 | Separate checkpoints |

### Training Details

Same as v1 except LR=1e-5. All other hyperparameters identical.

### Mid-Training Eval

| Step | Epoch | Syc Gap | Plain Acc | p_pressured |
|------|-------|---------|-----------|-------------|
| 25 | 0.1 | 0.312 | 0.715 | 0.414 |
| 50 | 0.3 | 0.318 | 0.715 | 0.415 |
| 75 | 0.4 | 0.312 | 0.720 | 0.423 |
| 100 | 0.5 | **0.262** | 0.720 | 0.453 |
| 150 | 0.8 | 0.270 | **0.735** | 0.463 |
| 175 | 0.9 | 0.260 | 0.730 | **0.468** |
| 200 | 1.0 | **0.255** | 0.725 | 0.466 |
| 250 | 1.3 | 0.255 | 0.730 | 0.464 |
| 300 | 1.6 | 0.262 | 0.730 | 0.460 |
| 400 | 2.1 | 0.273 | 0.725 | 0.450 |
| 500 | 2.6 | 0.260 | 0.715 | 0.451 |
| 575 | 3.0 | 0.265 | 0.720 | 0.451 |

### Training Metrics

| Metric | Step 10 | Step 130 | Step 576 (final) |
|--------|---------|----------|-----------------|
| Reward mean | +1.81 | +3.12 | ~+3.3 |
| Reward std | 0.64 | 0.88 | ~0.65 |
| KL | 0.001 | 0.112 | ~0.18 |
| Loss | 0.0 | 0.005 | ~0.006 |
| Completion length | 81 | 112 | ~75 |

### Three Phases of Training

1. **Steps 0-75 (warmup):** LR ramping, no behavioral change. Metrics identical to v1.
2. **Steps 75-200 (epoch 1, active learning):** Syc gap drops 0.318 → 0.255. Reward jumps +1.8 → +3.1. KL rises to 0.11. p_pressured improves 0.414 → 0.466. Plain accuracy improves 0.715 → 0.735.
3. **Steps 200-576 (epochs 2-3, plateau/drift):** Syc gap oscillates 0.255-0.275, settling around 0.265. p_pressured slightly degrades 0.468 → 0.451. Completion length increased then decreased. Signs of reward overoptimization — RM reward keeps rising but behavioral metrics plateau.

### Key Observations

1. **LR=1e-5 works.** Clear behavioral change vs v1's flat line. Syc gap reduced 0.318 → 0.255 (best) / 0.265 (final).

2. **Best checkpoint is around step 175-200 (end of epoch 1).** Epochs 2-3 don't improve and slightly degrade — similar to DPO's overfitting pattern. Unfortunately `save_total_limit=3` only kept steps 500/550/576.

3. **Completion length inflated then deflated.** Started at 81 tokens, peaked at ~117 at step 150, settled back to ~75. Possible length gaming during active learning phase (RM may reward longer, more detailed responses).

4. **Modest recovery compared to DPO.** DPO achieved syc_gap 0.099 (from mid-training eval). GRPO v2's best was 0.255. However, these are proxy metrics — full eval with 72B judge may tell a different story.

5. **KL reached 0.13-0.18** — 10x more than v1 (0.018) but still relatively small. The KL penalty (beta=0.04) is working to constrain drift.

6. **Clip ratio still 0%.** Even at LR=1e-5, the per-step probability changes didn't exceed ±20%. This suggests the LoRA adaptation is distributing changes across many tokens rather than making large changes to a few.

### Comparison: v1 vs v2

| Metric | v1 (LR=1e-6) | v2 (LR=1e-5) |
|--------|-------------|-------------|
| Final syc_gap | 0.310 | **0.265** |
| Best syc_gap | 0.307 | **0.255** |
| Final reward | +2.15 | **+3.3** |
| Final KL | 0.018 | **0.18** |
| Learned anything? | No | **Yes** |

### Next Steps

- Run full behavioral eval (vLLM + 72B judge) on the merged v2 model
- If results are promising, run probing analysis
- Consider v3 with LR=5e-6 (intermediate) or early stopping at epoch 1
- Consider binary reward thresholding to sharpen the learning signal
