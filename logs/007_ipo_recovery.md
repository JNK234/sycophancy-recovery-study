# Experiment 007: IPO Recovery (Qwen3-8B)

- **Date:** 2026-03-28
- **Model:** SFT-merged + IPO LoRA (loss_type="ipo" via DPOTrainer)
- **Config:** [`configs/training/ipo_recovery.yaml`](../configs/training/ipo_recovery.yaml)
- **Eval config:** [`configs/eval/post_ipo.yaml`](../configs/eval/post_ipo.yaml)
- **Probing config:** [`configs/probing/linear_probe_with_ipo.yaml`](../configs/probing/linear_probe_with_ipo.yaml)
- **Metrics:** [`results/eval/post-ipo/`](../results/eval/post-ipo/) (pending)
- **Infrastructure:** 4x H100 80GB, DDP via accelerate, TRL 0.29.1

## Background

IPO (Identity Preference Optimization, Azar et al. AISTATS 2024) replaces DPO's sigmoid loss with a squared loss: `L = (ρ - 1/(2τ))²`. Key differences from DPO:
- **Squared loss never saturates** — prevents deterministic policy collapse on near-deterministic preferences
- **Bounded target margin** — model can't just maximize chosen/rejected gap
- **No Bradley-Terry assumption** — works directly with pairwise preferences

Research question: Does IPO's stronger regularization produce deeper representation changes (like SimPO's chance-level probe transfer) or shallow suppression (like DPO's 0.754 transfer AUROC)?

## Hyperparameters

**Fair comparison config** — matched to DPO, only loss_type changed:

| Parameter | DPO (exp 003) | IPO (this exp) | Notes |
|-----------|--------------|----------------|-------|
| loss_type | sigmoid | **ipo** | Core change |
| beta/tau | 0.1 | 0.1 | Same value, different meaning in IPO (target margin = 5) |
| learning_rate | 2e-5 | 2e-5 | Matched for fair comparison |
| epochs | 1 | 1 | Same |
| LoRA | r=16, all-linear | r=16, all-linear | Same |
| data | 3,074 DPO pairs | 3,074 DPO pairs | Same |

## Dryrun Results

- Loss starts at ~25.0 (correct: `(0 - 1/(2*0.1))² = 25`)
- Rewards/accuracies: 0.0 → 0.625 in 5 steps
- 5 steps completed in 9s, adapter saved and merged successfully

## Training Results

- **Total steps:** 193 (1 epoch)
- **Training time:** 2m23s on 4x H100 DDP
- **wandb:** https://wandb.ai/jnk789/sycophancy-recovery/runs/kyopx55a

### Training Dynamics

| Metric | Step 10 | Step 30 | Step 50 | Step 100 | Step 150 | Step 193 |
|--------|---------|---------|---------|----------|----------|----------|
| Loss | 25.0 | 23.9 | 19.5 | 2.6 | 1.3 | 1.7 |
| Rewards/accuracy | 0.43 | 1.0 | 1.0 | 1.0 | 0.99 | 0.98 |
| Rewards/margin | 0.0 | 1.2 | 5.4 | 34.7 | 36.4 | 34.9 |
| Syc gap (mid-train) | — | — | 0.237 | 0.072 | 0.067 | — |

### Training Observations

1. **Convergence slower than DPO** — DPO hit 100% accuracy by step ~30; IPO took ~30 steps too but loss decreased more gradually (25→1.3 vs DPO's 0.693→0.024).
2. **Sycophancy gap drops to 0.067 by step 100** — strong mid-training recovery, comparable to DPO.
3. **Margins exploded to ~35** despite IPO's regularization — target margin is 5 (`1/(2*0.1)`), but the model massively overshot. This suggests beta=0.1 provides insufficient regularization for our near-deterministic data. A larger beta (e.g., 0.5) would constrain margins more tightly.
4. **Loss plateaus around step 90** at ~1.3-1.8, then oscillates slightly. Unlike DPO which continued decreasing monotonically.
5. **logps/chosen cratered** from -154 to -256 (model becoming less fluent on chosen responses) — similar to DPO's pattern but more extreme.
6. **logps/rejected cratered** from -80 to -522 — model aggressively suppressing rejected (sycophantic) responses.

## Behavioral Eval Results

- **Metrics:** [`results/eval/post-ipo/`](../results/eval/post-ipo/)

### Answer Dataset

| Template | Accuracy | Incorrect Rate | Hedged Rate | N |
|----------|----------|---------------|-------------|---|
| Plain (no pressure) | 0.466 | 0.451 | 0.078 | 1,817 |
| Suggest incorrect | 0.676 | 0.307 | 0.017 | 1,817 |
| Deny correct | 0.406 | 0.526 | 0.065 | 1,817 |
| Suggest correct | 0.383 | 0.538 | 0.075 | 1,817 |

**Sycophancy rate:** 0.417 (avg incorrect on suggest_incorrect + deny_correct)
**Sycophancy gap:** -0.035 (pressured has LOWER incorrect than plain — anomalous)
**Plain accuracy:** 0.466

### Are-You-Sure Dataset

| Metric | Value |
|--------|-------|
| Pass 1 accuracy | 0.470 |
| Flip rate | **0.257** |
| Stubbornness | 0.743 |

### Feedback Dataset

| Sub-dataset | Sycophancy Rate |
|-------------|----------------|
| Math | 0.273 |
| Arguments | 0.016 |
| Poems | 0.027 |
| **Overall** | **0.170** |

### Aggregate Comparison

| Experiment | Aggregate Syc | Answer Syc | Flip Rate | Feedback Syc |
|-----------|---------------|------------|-----------|--------------|
| 001 Baseline | 0.256 | 0.393 | 0.259 | 0.115 |
| 002 Post-SFT | 0.467 | 0.604 | 0.600 | 0.196 |
| 003 Post-DPO | 0.268 | 0.447 | 0.264 | 0.095 |
| 006d Post-SimPO | 0.176 | 0.275 | 0.167 | 0.087 |
| **007 Post-IPO** | **0.281** | **0.417** | **0.257** | **0.170** |

## Probing Results (500 prompts, consistent with experiments 005/006)

- **Date:** 2026-03-31 (initial), 2026-04-01 (with statistical rigor)
- **Config:** [`configs/probing/linear_probe_with_ipo.yaml`](../configs/probing/linear_probe_with_ipo.yaml)
- **Metrics:** [`results/probing/base-sft-dpo-simpo-ipo/`](../results/probing/base-sft-dpo-simpo-ipo/)

### Per-Model Probes (with 95% Bootstrap CIs)

| Model | Mean AUROC | Peak AUROC | 95% CI | Peak Layer | Syc Rate |
|-------|-----------|-----------|--------|------------|----------|
| Base | 0.783 | 0.905 | [0.828, 0.964] | 20 | 40.0% |
| SFT | 0.799 | 0.822 | [0.730, 0.900] | 17 | 67.0% |
| DPO | 0.746 | 0.863 | [0.786, 0.929] | 22 | 45.8% |
| SimPO | 0.750 | 0.794 | [0.685, 0.871] | 19 | 38.0% |
| **IPO** | **0.756** | **0.811** | **[0.726, 0.902]** | **3** | **40.2%** |

IPO peaks at **layer 3** — radically different from all other models (layers 17-22). All models have all 36 layers above chance.

### Random-Label Control (Noise Floor)

Control probes trained on shuffled SFT labels: **mean AUROC = 0.578 ± 0.021**. This is above the 0.55 warning threshold, indicating some noise-fitting with 4096-dim features on 400 samples. All real probe AUROCs (0.7-0.9) are well above this floor, confirming the signal is genuine. The effective "chance" baseline is ~0.58, not 0.5.

### Cross-Model Transfer (SFT probe → each model, with permutation tests)

| Transfer | Mean AUROC | Peak AUROC | p-value | Corrected p | Interpretation |
|----------|-----------|-----------|---------|-------------|----------------|
| SFT→Base | 0.689 | 0.883 | **0.005** | **0.005** | Significant — SFT pattern weakly present in base |
| SFT→DPO | **0.677** | 0.751 | **0.005** | **0.005** | **Significant** — suppression confirmed statistically |
| SFT→SimPO | **0.429** | 0.633 | 0.005 | **0.154** | **Not significant** after correction — peak is noise |
| **SFT→IPO** | **0.365** | **0.444** | **0.841** | **0.995** | **Not significant** — SFT pattern completely absent |

Key: corrected p uses max-statistic permutation test (corrects for cherry-picking best of 36 layers).

**Statistical conclusion:** DPO suppression is real (corrected p=0.005). SimPO and IPO removal is real — the SFT probe has no significant predictive power on either model after multiple-comparison correction.

### Probe-Space Ablation (Peak Layer)

| Model | Original | After Ablation | Retrained | Interpretation |
|-------|----------|---------------|-----------|----------------|
| Base (L20) | 0.905 | 0.500 | **0.742** | Multi-directional signal |
| SFT (L17) | 0.822 | 0.500 | **0.743** | Multi-directional signal |
| DPO (L22) | 0.863 | 0.500 | **0.808** | Strong residual signal |
| SimPO (L19) | 0.794 | 0.500 | **0.731** | Multi-directional signal |
| **IPO (L3)** | **0.811** | **0.500** | **0.814** | **Fully distributed — removing 1 direction has zero effect** |

After removing the primary sycophancy direction, all probes drop to 0.5 (expected). But fresh probes retrained on ablated activations recover to 0.73-0.81. Sycophancy is **multi-directional** — not concentrated in a single linear direction.

**IPO is the extreme case:** retrained AUROC (0.814) equals the original (0.811). Removing the top direction doesn't reduce signal at all. The sycophancy representation in IPO is fully distributed across many orthogonal directions, consistent with the deep restructuring from IPO's non-saturating loss.

### Probe Direction Similarity (cosine with SFT)

| Pair | Mean Cosine |
|------|------------|
| SFT vs DPO | 0.262 |
| SFT vs SimPO | 0.069 |
| **SFT vs IPO** | **-0.038** |

### Probing Interpretation

1. **SFT→IPO transfer at 0.365 is the lowest of all methods** — below SimPO (0.429) and well below chance (0.5). The SFT sycophancy probe is anti-predictive on IPO activations. Permutation test confirms: p=0.841 (not significant).

2. **SFT→DPO is the only statistically significant transfer** after multiple-comparison correction (corrected p=0.005). DPO suppression is confirmed with statistical rigor. SimPO (corrected p=0.154) and IPO (corrected p=0.995) show no significant SFT pattern.

3. **IPO peaks at layer 3** while all others peak at layers 17-22. This is a fundamentally different internal organization — sycophancy-relevant computation moved from late layers to very early layers.

4. **Sycophancy is multi-directional in all models.** Ablating the primary probe direction and retraining recovers 0.73-0.81 AUROC everywhere. IPO is the extreme — 0.814 retrained vs 0.811 original — the signal is fully distributed, not concentrated.

5. **Control noise floor at 0.578** means we should interpret AUROCs between 0.5-0.6 cautiously. All per-model probes (0.78+) and DPO transfer (0.677) are well above this floor.

6. **Behavioral-mechanistic paradox:** IPO has the deepest representational change (transfer 0.365, negative cosine, fully distributed signal) but only mediocre behavioral recovery (aggregate 0.281 vs SimPO's 0.176). The capability degradation (plain accuracy 0.466) masks the representational improvement. IPO's squared loss forced deeper restructuring but also broke general capabilities.

## Interpretation

### Behavioral Analysis

IPO with matched hyperparameters (beta=0.1, LR=2e-5) achieves **aggregate sycophancy 0.281** — very close to DPO's 0.268 and near the baseline 0.256. But the pattern is different:

1. **Answer sycophancy (0.417) is WORSE than DPO (0.447)**... wait, both are worse than baseline (0.393). The model's plain accuracy cratered to 0.466 (baseline was 0.616, DPO was similar). This suggests **capability degradation** — the model got worse at answering questions overall, not just under pressure.

2. **Flip rate (0.257) matches DPO (0.264) and baseline (0.259)** — IPO recovered are-you-sure robustness to baseline level.

3. **Feedback sycophancy (0.170) is higher than DPO (0.095) and SimPO (0.087)** — IPO did NOT recover well on the feedback dimension. Math sycophancy at 0.273 is notably high.

4. **Negative sycophancy gap (-0.035)** is anomalous — the model answers WORSE on plain questions than pressured ones. This happened because plain accuracy dropped so much.

### Comparison to DPO and SimPO

IPO at beta=0.1 looks like a slightly worse DPO with more capability degradation. The exploding margins during training (target 5, actual 35) suggest the regularization was insufficient — the model overfit similarly to DPO but with added fluency loss from the squared loss dynamics.

SimPO remains the clear winner behaviorally (0.176 aggregate). IPO's theoretical advantages (bounded margins, no BT assumption) did not translate to better behavioral recovery at these hyperparameters.

**Key question answered by probing:** Yes — IPO produces radically different internal representations despite similar behavioral metrics. SFT→IPO transfer AUROC 0.365 (below chance, SFT pattern inverted) vs DPO's 0.677 (suppression). See Probing Results section above.

## Hyperparameter Sweep (v2-v4)

Sweep to find better IPO settings after v1 showed capability degradation despite good sycophancy recovery.

### Sweep Configs

| Config | β (tau) | Target Margin | LR | Output |
|--------|---------|--------------|-----|--------|
| v1 (above) | 0.1 | 5 | 2e-5 | `/scratch/.../ipo` |
| v2 | 0.5 | 1 | 2e-5 | `/scratch/.../ipo-v2` |
| v3 | 0.5 | 1 | 5e-6 | `/scratch/.../ipo-v3` |
| v4 | 1.0 | 0.5 | 5e-6 | `/scratch/.../ipo-v4` |

### Sweep Results (Mid-Training Metrics)

| Config | Syc Gap @50 | Syc Gap @100 | Syc Gap @150 | Final Loss | Final Margins | logps/chosen | logps/rejected |
|--------|------------|-------------|-------------|-----------|--------------|-------------|---------------|
| **v1** β=0.1 LR=2e-5 | 0.237 | **0.072** | **0.067** | 1.7 | 35 | -256 | -522 |
| **v2** β=0.5 LR=2e-5 | 0.242 | 0.217 | 0.218 | 0.06 | 40 | -135 | -136 |
| **v3** β=0.5 LR=5e-6 | 0.312 | 0.255 | 0.255 | 0.37 | 19 | -132 | -91 |
| **v4** β=1.0 LR=5e-6 | 0.307 | 0.260 | 0.248 | 0.03 | 33 | -133 | -88 |

### Sweep Observations

1. **v1 remains the best at sycophancy reduction** (syc_gap 0.067) but at the cost of severe capability degradation (logps/chosen -256, plain accuracy 0.466).

2. **Higher beta made recovery WORSE, not better.** v2/v3/v4 barely reduced sycophancy in 1 epoch (syc_gap 0.22-0.26). Stronger regularization prevented sufficient behavioral change.

3. **v3 (β=0.5, LR=5e-6) has the best margin control** — margins at 19 vs target 1 (still overshot, but 2x less than v1). Fluency preserved (logps/chosen -132 vs baseline -154). But it needs more epochs.

4. **v2 (β=0.5, LR=2e-5) is paradoxical** — high LR + high beta = low loss (0.06) but high margins (40) and poor recovery (syc_gap 0.218). The model minimized the squared error quickly without learning to be less sycophantic. The loss landscape is degenerate.

5. **v4 (β=1.0, LR=5e-6) = too much regularization.** Target margin 0.5 is so tight the model can barely differentiate chosen/rejected. Loss dropped to 0.03 but margins still 33 — the model found a way around the constraint.

### Fundamental Issue with IPO for This Task

IPO's squared loss targets a fixed margin `1/(2β)`. But our preference data is near-deterministic (sycophantic vs non-sycophantic is always clear). The model needs to make LARGE changes to stop being sycophantic, but IPO's regularization actively fights this. DPO's saturating sigmoid lets the model push hard once it's found the right direction; IPO's quadratic penalty punishes going too far. For our task, "too far" IS the right amount.

This explains why:
- Low β (0.1): effective recovery but at cost of uncontrolled margin explosion → capability damage
- High β (0.5, 1.0): insufficient recovery because regularization prevents the needed behavioral shift

**IPO is structurally mismatched for near-deterministic sycophancy recovery.** The technique is designed for noisy preferences where overfitting is the risk. Our preferences aren't noisy — they're clear-cut.

## Next Steps

- ~~Run linear probing on v1 model~~ — DONE (SFT→IPO transfer 0.365, below chance)
- Consider v3 with 3 epochs — probing revealed IPO restructures deeply, slower training might preserve capabilities while maintaining the deep change
- Move to KTO (unpaired preference data, different paradigm)
- IPO's position in comparison table: behaviorally mediocre (0.281), mechanistically the deepest change (transfer 0.365, negative cosine). Not "DPO-equivalent" — fundamentally different mechanism.
