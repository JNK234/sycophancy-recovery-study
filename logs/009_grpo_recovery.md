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

---

## RM Threshold Calibration (2026-04-07)

Before running GRPO v3 with binary reward, we calibrated the RM score distribution on actual SFT model generations. Script: `scripts/calibrate_rm_threshold.py`. Raw data: `results/rm_threshold_calibration.json`.

**Setup:** 100 prompts × 4 generations each = 400 scored completions. SFT model at temp=0.7, max 256 tokens. Scored with trained RM (Experiment 008).

### RM Score Distribution

| Stat | Value |
|------|-------|
| Mean | 1.93 |
| Std | 0.61 |
| Median | 1.89 |
| Min | 0.23 |
| Max | 3.64 |

### Threshold Sweep

| Threshold | % above (+1) | % below (-1) | Signal Quality |
|-----------|-------------|-------------|----------------|
| 1.23 | 90.2% | 9.8% | POOR |
| 1.48 | 77.5% | 22.5% | OK |
| 1.73 | 60.0% | 40.0% | GOOD |
| 1.98 | 45.0% | 55.0% | GOOD |
| 2.23 | 30.2% | 69.8% | GOOD |
| 2.48 | 19.2% | 80.8% | POOR |

### Decision

The originally planned threshold of 1.5 was too low — would label 77.5% as honest, collapsing the advantage signal. Optimal threshold is ~1.9 (median, 45/55 split).

However, we decided to **defer binary reward to a later experiment** and first try **continuous RM + LR=2e-5** (v3). This isolates one variable (LR) from v2. Binary reward (v4) can follow if continuous reward + higher LR is insufficient.

---

## v3: LR=2e-5, Continuous RM, 1 Epoch

- **Date:** 2026-04-07
- **Config:** [`configs/training/grpo_v3_continuous_lr2e5.yaml`](../configs/training/grpo_v3_continuous_lr2e5.yaml)
- **Merged model:** `/scratch/wnn7240/sycophancy-recovery/outputs/grpo-v3/merged`
- **Wandb:** https://wandb.ai/jnk789/sycophancy-recovery/runs/0k6o7ztk (rank 0)
- **Raw log:** [`logs/training_outputs/grpo_v3_continuous_lr2e5_training.log`](training_outputs/grpo_v3_continuous_lr2e5_training.log)
- **Infrastructure:** 2x H100 DDP via accelerate, 384 steps, 1h 16min

### Changes from v2

| Parameter | v2 | v3 | Rationale |
|-----------|----|----|-----------|
| learning_rate | 1e-5 | **2e-5** | v2 showed partial recovery; 2x to match DPO's LR |
| num_train_epochs | 3 | **1** | v2 showed overfitting after epoch 1 |
| save_steps | 50 | **25** | Don't lose best checkpoint (v2 lost step 200) |
| save_total_limit | 3 | **5** | Keep more checkpoints |
| num_processes (DDP) | 4 | 2 | Only 2 GPUs available; 384 steps vs 192 |

### Mid-Training Eval

| Step | Epoch | Syc Gap | Plain Acc | p_pressured |
|------|-------|---------|-----------|-------------|
| 25 | 0.07 | 0.312 | 0.715 | 0.414 |
| 50 | 0.13 | 0.315 | 0.715 | 0.419 |
| 75 | 0.20 | 0.273 | 0.735 | 0.457 |
| 100 | 0.26 | 0.270 | 0.735 | 0.468 |
| 125 | 0.33 | **0.265** | 0.740 | 0.476 |
| 150 | 0.39 | 0.265 | 0.740 | 0.472 |
| 175 | 0.46 | 0.275 | 0.750 | 0.478 |
| 200 | 0.52 | 0.273 | 0.745 | 0.468 |
| 225 | 0.59 | 0.272 | 0.740 | 0.463 |
| 250 | 0.65 | 0.275 | 0.740 | 0.464 |
| 275 | 0.72 | 0.262 | 0.735 | 0.466 |
| 300 | 0.78 | 0.265 | 0.740 | 0.467 |
| 325 | 0.85 | 0.267 | 0.735 | 0.465 |
| 350 | 0.91 | **0.258** | 0.730 | 0.466 |
| 375 | 0.98 | 0.267 | 0.740 | 0.465 |

### Final Training Metrics

| Metric | Value |
|--------|-------|
| Train loss | 0.006 |
| Reward mean | +3.58 |
| Reward std | 0.70 |
| KL | 0.204 |
| Clip ratio | 0% |
| Completion length | 120.9 |
| Runtime | 1h 16min (2x H100 DDP) |

### Analysis

1. **Best syc_gap 0.258 (step 350)**, marginally better than v2's best of 0.255. The improvement is within noise — 2x LR did not meaningfully improve over v2.

2. **Faster convergence**: v3 reached 0.265 by step 125, while v2 took until step 200. Higher LR accelerated learning but hit the same floor.

3. **Plateau at 0.26-0.27**: From step 125 to 384, syc_gap oscillated in a narrow band. This suggests a **fundamental limit of GRPO with continuous RM reward for sycophancy recovery** — the reward model's signal isn't sharp enough to push further.

4. **Reward overoptimization confirmed**: RM reward rose from ~1.9 to 3.58 (+88%), but behavioral metrics plateaued. The proxy metric kept increasing while true quality stalled — textbook Gao et al. (2022) overoptimization.

5. **KL reached 0.204** — comparable to v2 (0.18) despite higher LR. The KL penalty (β=0.04) is effectively constraining drift.

6. **Clip ratio still 0% throughout** — even at LR=2e-5, individual token probability changes stay within ±20%. This is characteristic of LoRA: changes are distributed across many low-rank dimensions rather than concentrating on a few tokens.

### Comparison: v1 vs v2 vs v3

| Metric | v1 (LR=1e-6) | v2 (LR=1e-5) | v3 (LR=2e-5) |
|--------|-------------|-------------|-------------|
| Best syc_gap | 0.307 | **0.255** | **0.258** |
| Final syc_gap | 0.310 | 0.265 | 0.267 |
| Steps to best | Never | 200 | 350 |
| Steps to converge | Never | 100 | 75 |
| Final reward | +2.15 | +3.3 | +3.58 |
| Final KL | 0.018 | 0.18 | 0.204 |
| Learned anything? | No | Yes | Yes |

### Conclusion

GRPO with continuous RM reward plateaus at syc_gap ~0.26, regardless of LR (1e-5 or 2e-5). This is substantially worse than DPO (0.099) and SimPO. The bottleneck is likely the RM signal quality, not the LR.

---

## v3 Full Behavioral Eval (72B Judge)

- **Date:** 2026-04-12
- **Model evaluated:** `/scratch/wnn7240/sycophancy-recovery/outputs/grpo-v3/merged`
- **Eval config:** [`configs/eval/post_grpo.yaml`](../configs/eval/post_grpo.yaml) (TP=2, judge max_model_len=2560, gpu_mem_util=0.98)
- **Metrics:** [`results/eval/post-grpo-v3/`](../results/eval/post-grpo-v3/)

### Evaluation Results

| Metric | Baseline | Post-SFT | Post-DPO | Post-SimPO | Post-IPO | **Post-GRPO v3** |
|--------|----------|----------|----------|------------|----------|-----------------|
| **Aggregate sycophancy** | 0.256 | 0.467 | 0.268 | 0.176 | 0.281 | **0.169** |
| Answer sycophancy rate | 0.393 | 0.604 | 0.447 | 0.275 | 0.417 | 0.311 |
| Answer sycophancy gap | 0.088 | 0.225 | 0.099 | — | — | 0.027 |
| Answer plain accuracy | 0.616 | 0.485 | 0.577 | — | — | 0.531 |
| Are-you-sure flip rate | 0.259 | 0.600 | 0.264 | 0.167 | 0.257 | **0.082** |
| Stubbornness rate | 0.741 | 0.400 | 0.736 | — | — | 0.918 |
| Feedback overall syc | 0.115 | 0.196 | 0.095 | 0.087 | 0.170 | 0.113 |
| Feedback math | 0.068 | 0.040 | 0.054 | — | — | 0.037 |
| Feedback arguments | 0.031 | 0.386 | 0.040 | — | — | 0.082 |
| Feedback poems | 0.297 | 0.443 | 0.238 | — | — | 0.325 |

### Key Observations

1. **Lowest aggregate sycophancy (0.169)** — beats SimPO (0.176) and all other methods. GRPO is the best recovery method behaviorally.

2. **Flip rate of 0.082 is exceptional** — only 8% of correct answers flipped under "are you sure?" pressure (baseline 26%, SFT 60%, DPO 26%). Stubbornness rate 0.918 is the highest of any model.

3. **Sycophancy gap nearly eliminated (0.027)** — the model's accuracy barely changes whether user suggests an incorrect answer or not. Mid-training eval showed syc_gap ~0.26, but full eval with generation reveals much stronger recovery.

4. **Mid-training eval (logit-based) was misleading** — showed syc_gap plateau at 0.26, but the 72B judge eval shows aggregate 0.169. The logit-based proxy metric substantially underestimated GRPO's behavioral recovery. This is likely because GRPO changes generation behavior (sampling, length, style) in ways that logit extraction at a single token position doesn't capture.

5. **Capability tradeoff:** Plain accuracy 0.531 (vs baseline 0.616) — some factual capability lost. High hedged_rate (18.4%) suggests the model became cautious, preferring to hedge rather than commit to potentially sycophantic answers. This is a different failure mode from sycophancy — over-caution vs over-agreement.

6. **Poems sycophancy still high (0.325)** — worse than baseline (0.297). Subjective domains remain difficult for GRPO, likely because the RM wasn't trained on feedback/poem data.

### Next Steps

- ~~Run linear probing to determine if GRPO is suppression or removal (like DPO vs SimPO)~~ → Done (Experiment 010)
- ~~Try binary reward GRPO (v4) with threshold=1.9 to see if sharper signal improves further~~ → Done (v4 below — did NOT improve)
- The mid-training vs full-eval discrepancy is important to document — it affects how we interpret training curves for all methods

---

## v4: Binary Reward Model (threshold=1.9) — Worse Than v3

- **Date:** 2026-04-12
- **Model:** SFT-merged Qwen3-8B + GRPO LoRA (r=16, all-linear)
- **Config:** [`configs/training/grpo_v3_binary_lr2e5.yaml`](../configs/training/grpo_v3_binary_lr2e5.yaml)
- **Eval config:** [`configs/eval/post_grpo_v4.yaml`](../configs/eval/post_grpo_v4.yaml)
- **Merged model:** `/scratch/wnn7240/sycophancy-recovery/outputs/grpo-v4/merged`
- **Wandb:** https://wandb.ai/sam2act-plus-ext/sycophancy-recovery/runs/mj067woc (rank 0)
- **Raw log:** [`logs/training_outputs/grpo_v4_binary_lr2e5_training.log`](training_outputs/grpo_v4_binary_lr2e5_training.log)
- **Metrics:** [`results/eval/post-grpo-v4/`](../results/eval/post-grpo-v4/)
- **Infrastructure:** 2x H100 DDP via accelerate, 384 steps, 1h 27min training

### Training Details

| Parameter | Value |
|-----------|-------|
| Method | GRPO (loss_type="grpo") |
| Reward | Binary RM (threshold=1.9, ±1.0 output) |
| Data | 3,236 unique prompts |
| Learning rate | 2e-5 (matching v3) |
| Beta (KL) | 0.04 |
| Epsilon (clipping) | 0.2 |
| Num generations | 16 per prompt (2x v3) |
| Max completion length | 256 tokens |
| Temperature | 0.7 |
| Effective batch | 8 prompts (2 GPUs × 1 prompt × 4 grad_accum) |
| Epochs | 1 (384 steps — 2x v3 due to half batch) |
| LoRA | r=16, alpha=32, all-linear, dropout=0.05 |
| Runtime | 1h 27min on 2x H100 DDP |

### Mid-Training Eval (Sycophancy — Logit Extraction)

| Step | Epoch | Plain Acc | Syc Gap | p_correct_pressured |
|------|-------|-----------|---------|---------------------|
| 25 | 0.07 | 0.715 | 0.312 | 0.414 |
| 50 | 0.13 | 0.715 | 0.307 | 0.421 |
| 75 | 0.20 | 0.720 | 0.295 | 0.432 |
| 100 | 0.26 | 0.720 | 0.292 | 0.437 |
| **125** | **0.33** | **0.720** | **0.282** | **0.443** |
| 150 | 0.39 | 0.720 | 0.292 | 0.437 |
| 175 | 0.46 | 0.720 | 0.297 | 0.434 |
| 200 | 0.52 | 0.720 | 0.297 | 0.430 |
| 250 | 0.65 | 0.720 | 0.295 | 0.434 |
| 300 | 0.78 | 0.720 | 0.292 | 0.434 |
| 350 | 0.91 | 0.720 | 0.300 | 0.434 |
| 375 | 0.98 | 0.720 | 0.300 | 0.434 |

**Plateaued early (step ~100-125) at syc_gap ~0.29.** Best was 0.282 at step 125. Much weaker than v3's best of ~0.258.

### Final Training Metrics

| Metric | Start | End |
|--------|-------|-----|
| Reward mean | 0.06 | 0.75 |
| Reward std | 1.01 | 0.63 |
| frac_reward_zero_std | 0.0 | 0.50 |
| KL | 0.0 | 0.072 |
| Clip ratio | 0% | 0% |

**Critical issue:** `frac_reward_zero_std` rose to 0.50 — half the groups had all +1 or all -1 rewards, making advantage estimation impossible for those groups. The binary signal collapsed as the model improved.

### Evaluation Results

| Metric | Baseline | Post-SFT | **v3 (continuous)** | **v4 (binary)** | v4 vs v3 |
|--------|----------|----------|---------------------|-----------------|----------|
| **Aggregate sycophancy** | 0.256 | 0.467 | **0.169** | **0.312** | **+0.143** |
| Answer sycophancy rate | 0.393 | 0.604 | 0.311 | 0.484 | +0.173 |
| Answer sycophancy gap | 0.088 | 0.225 | 0.027 | 0.136 | +0.109 |
| Answer plain accuracy | 0.616 | 0.485 | 0.531 | 0.498 | -0.033 |
| Are-you-sure flip rate | 0.259 | 0.600 | 0.082 | 0.295 | +0.213 |
| Stubbornness rate | 0.741 | 0.400 | 0.918 | 0.705 | -0.213 |
| Feedback overall syc | 0.115 | 0.196 | 0.113 | 0.157 | +0.044 |
| Feedback math | 0.068 | 0.040 | 0.066 | 0.031 | -0.035 |
| Feedback arguments | 0.031 | 0.386 | 0.088 | 0.265 | +0.177 |
| Feedback poems | 0.297 | 0.443 | 0.325 | 0.391 | +0.066 |

### Analysis

**Binary GRPO v4 is substantially worse than continuous v3 across every metric.** Aggregate 0.312 vs 0.169 — the binary reward model barely improved on the SFT model (0.467) and is worse than DPO (0.268).

**Why binary reward failed for sycophancy recovery:**

1. **Information loss:** Continuous RM scores provide gradient-rich signal. A completion scoring 2.5 gets a stronger push than one at 1.95, but binary collapses both to +1. The RM's confidence information is entirely discarded.

2. **Signal collapse at 50%:** As training progressed, the model improved enough that most completions scored above threshold (1.9). By the end, half the groups had zero variance — all +1 rewards, no advantage signal, no learning. The model stopped improving.

3. **Binary works for verifiable tasks, not sycophancy:** RLVR-style binary rewards work when there's a clear right/wrong answer (math, code). Sycophancy is a spectrum — "somewhat agreeable" and "fully sycophantic" need different treatment. Binary rewards can't distinguish between subtle and extreme sycophancy.

4. **Halved effective batch:** 2 GPUs gave effective batch=8 (vs v3's 16). Smaller groups → noisier advantage estimates, especially with binary rewards where many groups have zero variance.

5. **Arguments sycophancy backslid badly:** 0.265 (vs v3's 0.088, baseline 0.031). The binary model actually became MORE sycophantic on arguments than it started. This suggests the binary signal sometimes reinforced the wrong behavior.

### Lessons Learned

1. **Binary rewards are NOT universally better than continuous.** The RLVR literature shows binary success on verifiable tasks. Sycophancy, as a behavioral pattern with degree, needs continuous signal.
2. **Watch `frac_reward_zero_std`** — when it approaches 0.5, training is effectively stalled for half the data. This is the key diagnostic for binary reward collapse.
3. **Threshold calibration isn't enough.** Even with a well-calibrated threshold (1.9, 45/55 split at start), model improvement shifts the distribution above threshold, collapsing signal over time. An adaptive threshold might help but adds complexity.
4. **v3 (continuous RM, LR=2e-5) remains the best GRPO configuration.** The continuous signal provides richer gradients throughout training.
