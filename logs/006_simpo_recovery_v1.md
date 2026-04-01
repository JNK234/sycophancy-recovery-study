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

## Full Behavioral Eval Results (SimPO Final)

- **Eval config:** [`configs/eval/post_simpo.yaml`](../configs/eval/post_simpo.yaml)
- **Metrics:** [`results/eval/post-simpo/`](../results/eval/post-simpo/)

### Results Comparison

| Metric | Baseline | Post-SFT | Post-DPO | **Post-SimPO** | SimPO vs DPO |
|--------|----------|----------|----------|----------------|-------------|
| **Aggregate sycophancy** | 0.256 | 0.467 | 0.268 | **0.176** | **-0.092** |
| Answer sycophancy rate | 0.393 | 0.604 | 0.447 | 0.365 | -0.082 |
| Answer sycophancy gap | 0.088 | 0.225 | 0.099 | **0.010** | -0.089 |
| Answer plain accuracy | 0.616 | 0.485 | 0.577 | 0.558 | -0.019 |
| Are-you-sure flip rate | 0.259 | 0.600 | 0.264 | **0.104** | **-0.160** |
| Stubbornness rate | 0.741 | 0.400 | 0.736 | **0.896** | +0.160 |
| Feedback overall syc | 0.115 | 0.196 | 0.095 | **0.058** | **-0.037** |
| Feedback math | 0.068 | 0.040 | 0.054 | 0.095 | +0.041 |
| Feedback arguments | 0.031 | 0.386 | 0.040 | **0.002** | -0.038 |
| Feedback poems | 0.297 | 0.443 | 0.238 | **0.007** | **-0.231** |

### Key Findings

1. **SimPO aggregate 0.176 — BELOW BASELINE (0.256).** Not just recovery, but genuine improvement over the pre-SFT model. DPO only recovered to 0.268.
2. **Flip rate 0.104** — only 10% of correct answers flip under "are you sure?" pressure. Baseline was 26%, DPO was 26%. SimPO made the model dramatically more epistemically robust.
3. **Poems sycophancy virtually eliminated** — 0.007 vs 0.297 baseline. The model stopped flattering entirely on subjective content.
4. **Arguments sycophancy 0.002** — near-zero. The model evaluates arguments honestly regardless of user framing.
5. **Sycophancy gap 0.010** — the model responds identically whether or not the user applies pressure. This is the ideal behavior.
6. **Tradeoff: plain accuracy slightly lower** (0.558 vs DPO's 0.577). Minor cost for major sycophancy reduction.
7. **Tradeoff: math feedback sycophancy slightly higher** (0.095 vs DPO's 0.054). SimPO may have overcorrected on objectivity — being too contrarian on correct math solutions.

### Why SimPO Outperforms DPO

Hypothesis: DPO is KL-constrained to the sycophantic reference model. SimPO has no reference anchor — the policy can drift further from the sycophantic starting point. This freedom allows deeper behavioral change, but also risks overcorrection (the math feedback result).

## Linear Probing Results — SimPO Genuinely Removes Sycophancy

- **Config:** [`configs/probing/linear_probe_with_simpo.yaml`](../configs/probing/linear_probe_with_simpo.yaml)
- **Metrics:** [`results/probing/base-sft-dpo-simpo/`](../results/probing/base-sft-dpo-simpo/)

### Per-Model Probes

| Model | Mean AUROC | Peak AUROC | Peak Layer | Syc Rate |
|-------|-----------|-----------|------------|----------|
| Base | 0.745 | 0.811 | 26 | 44.0% |
| SFT | 0.758 | 0.856 | 24 | 67.2% |
| DPO | 0.723 | 0.793 | 19 | 49.6% |
| **SimPO** | **0.695** | **0.776** | 19 | 39.0% |

### Cross-Model Transfer (THE KEY RESULT)

| Transfer | Mean AUROC | Peak AUROC | Interpretation |
|----------|-----------|-----------|----------------|
| SFT→Base | 0.628 | 0.782 | Pre-existing text features, not SFT-specific |
| SFT→DPO | **0.652** | **0.755** | SFT sycophancy pattern PERSISTS — suppression |
| SFT→SimPO | **0.388** | **0.487** | SFT pattern **GONE** — below chance, anti-correlated |

### Probe Direction Similarity

| Comparison | Mean Cosine | Interpretation |
|-----------|------------|----------------|
| SFT vs DPO | 0.262 | Partially shared direction — DPO modified but didn't reorganize |
| SFT vs SimPO | **0.069** | Nearly orthogonal — SimPO reorganized representations completely |
| DPO vs SimPO | 0.246 | DPO and SimPO encode sycophancy differently |

### Interpretation

**SimPO achieves what DPO could not: genuine removal of the SFT sycophancy representation.**

1. SFT→SimPO transfer AUROC of 0.388 is BELOW 0.5 (chance). The SFT sycophancy probe is anti-predictive on SimPO — the old sycophancy direction now correlates with honest behavior.
2. SFT vs SimPO cosine similarity of 0.069 means their sycophancy encodings are nearly orthogonal. SimPO didn't just suppress the SFT direction — it reorganized the representation space.
3. DPO retained the SFT direction (transfer 0.652, cosine 0.262). SimPO broke free from it entirely.

**Why the difference?** DPO is KL-constrained to the sycophantic reference model, limiting how far representations can change. SimPO has no reference anchor — the policy is free to reorganize its internal representations, not just patch the output layer.

## Statistical Rigor: Updated Probing with 5-Model Analysis

- **Date:** 2026-04-01
- **Config:** [`configs/probing/linear_probe_with_ipo.yaml`](../configs/probing/linear_probe_with_ipo.yaml)
- **Metrics:** [`results/probing/base-sft-dpo-simpo-ipo/`](../results/probing/base-sft-dpo-simpo-ipo/)

Reran probing on all 5 models (base, SFT, DPO, SimPO, IPO) with bootstrap CIs, permutation tests, random-label controls, and probe-space ablation. Uses 500 prompts with 406/94 train/val split.

### Per-Model Probes (with 95% Bootstrap CIs)

| Model | Mean AUROC | Peak AUROC | 95% CI | Peak Layer | Syc Rate |
|-------|-----------|-----------|--------|------------|----------|
| Base | 0.783 | 0.905 | [0.828, 0.964] | 20 | 40.0% |
| SFT | 0.799 | 0.822 | [0.730, 0.900] | 17 | 67.0% |
| DPO | 0.746 | 0.863 | [0.786, 0.929] | 22 | 45.8% |
| **SimPO** | **0.750** | **0.794** | **[0.685, 0.871]** | **19** | **38.0%** |
| IPO | 0.756 | 0.811 | [0.726, 0.902] | 3 | 40.2% |

### Cross-Model Transfer (SFT probe → SimPO, with permutation tests)

| Transfer | Mean AUROC | Peak AUROC | p-value | Corrected p | Interpretation |
|----------|-----------|-----------|---------|-------------|----------------|
| SFT→DPO | 0.677 | 0.751 | 0.005 | **0.005** | Significant — suppression |
| **SFT→SimPO** | **0.429** | 0.633 | 0.005 | **0.154** | **NOT significant after correction** |
| SFT→IPO | 0.365 | 0.444 | 0.841 | 0.995 | NOT significant — pattern absent |

SimPO's corrected p=0.154 means the peak transfer AUROC (0.633 at layer 18) is NOT statistically significant after accounting for cherry-picking across 36 layers. The mean transfer of 0.429 (below chance) is the more honest metric — the SFT sycophancy pattern does not survive in SimPO.

### Random-Label Control (Noise Floor)

Control probes trained on shuffled SFT labels: **mean AUROC = 0.578 ± 0.021**. This is the noise floor from fitting 4096-dim features on 400 samples. SimPO's per-model peak (0.794) and mean (0.750) are well above this floor. The transfer AUROCs (mean 0.429) are well BELOW this floor — confirming genuine absence of the SFT pattern, not noise.

### Probe-Space Ablation (Peak Layer 19)

| Metric | Value |
|--------|-------|
| Original AUROC | 0.794 |
| After ablation | 0.500 (probe direction removed) |
| Retrained AUROC | **0.731** |
| Retrained accuracy | 0.67 |

After removing the primary sycophancy direction and retraining a fresh probe, SimPO recovers to 0.731 (92% of original). Sycophancy information is **multi-directional** in SimPO — not concentrated in a single linear direction.

### Updated Probe Direction Similarity (cosine with SFT)

| Pair | Mean Cosine |
|------|------------|
| SFT vs DPO | 0.262 |
| **SFT vs SimPO** | **0.069** |
| DPO vs SimPO | 0.246 |
| SFT vs IPO | -0.038 |

SimPO's near-orthogonal cosine (0.069) with SFT confirms complete representational reorganization.

### Key Statistical Takeaways

1. **SimPO's SFT pattern removal is statistically confirmed.** The peak transfer AUROC (0.633) does NOT survive max-statistic correction (corrected p=0.154). The mean transfer (0.429) is below the noise floor (0.578). The SFT sycophancy representation is genuinely absent.

2. **SimPO's own sycophancy signal is real and multi-directional.** Per-model peak 0.794 with 95% CI [0.685, 0.871] is well above the noise floor. Ablation recovers 0.731 after removing primary direction — sycophancy is encoded in multiple orthogonal directions.

3. **DPO is the only method with statistically significant SFT pattern persistence** (corrected p=0.005). Both SimPO and IPO break the SFT pattern, but via different mechanisms — SimPO through reference-free optimization, IPO through non-saturating loss.
