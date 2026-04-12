# Experiment 010: GRPO Probing — Linear Probing with All 6 Models

- **Date:** 2026-04-12
- **Type:** Mechanistic analysis (no new models trained)
- **Config:** [`configs/probing/linear_probe_full_with_grpo.yaml`](../configs/probing/linear_probe_full_with_grpo.yaml) (saved as `base-sft-dpo-simpo-ipo-grpo`)
- **Results:** [`results/probing/base-sft-dpo-simpo-ipo-grpo/`](../results/probing/base-sft-dpo-simpo-ipo-grpo/)
- **Activations:** `/scratch/wnn7240/sycophancy-recovery/probing/base-sft-dpo-simpo-ipo-grpo/`
- **Infrastructure:** Single H100 per model (sequential extraction), sklearn probes

## Purpose

Add GRPO v3 (Experiment 009c, aggregate sycophancy 0.169 — best behavioral result) to the probing pipeline alongside all existing models (base, SFT, DPO, SimPO, IPO). The central question: **Does GRPO's strong behavioral recovery reflect genuine representational change (like SimPO/IPO) or surface suppression (like DPO)?**

## Method

Same pipeline as Experiments 005-007f:
- 500 prompts (suggest_incorrect + deny_correct templates)
- Prompt-only activation extraction (last token, all 36 layers)
- Per-model behavior labels from judge verdicts
- Logistic regression probes (C=1.0, max_iter=2000)
- 80/20 train/val split by question group
- Full statistical rigor: bootstrap CIs (200 iterations), permutation tests (200 permutations), max-statistic peak-layer correction, random-label control (3 seeds), probe-space ablation

## Results

### 1. Per-Model Probe Performance

How well can a linear probe distinguish sycophantic vs honest behavior from each model's own activations?

| Model | Label Balance (% syc) | Mean AUROC | Peak AUROC | Peak Layer | Layers > Chance |
|-------|----------------------|-----------|-----------|------------|-----------------|
| Base | 40.9% | 0.767 | 0.827 | 31 | 36/36 |
| SFT | 68.2% | 0.837 | 0.882 | 30 | 36/36 |
| DPO | 47.1% | 0.733 | 0.827 | 24 | 36/36 |
| SimPO | 37.0% | 0.753 | 0.842 | 22 | 36/36 |
| IPO | 40.2% | 0.931 | 0.982 | 34 | 36/36 |
| **GRPO** | **37.0%** | **0.541** | **0.660** | **25** | **7/36** |

**GRPO stands out dramatically:** weakest own-model probe of all 6 models. Only 7 layers carry above-chance signal (vs 36/36 for every other model). Peak AUROC 0.660 is barely above the random-label control noise floor (~0.578).

**Interpretation:** GRPO's internal representation encodes sycophancy vs honest intent very weakly — the behavioral distinction is not linearly separable in activation space. This could mean:
1. GRPO genuinely doesn't encode sycophantic intent (the decision happens via a different mechanism)
2. The encoding is highly non-linear and our logistic probes miss it
3. The low sycophancy rate (37%) combined with weak signal makes detection hard

### 2. Cross-Model Transfer (SFT Probe → Other Models)

The key experiment: apply the SFT-trained sycophancy probe (without retraining) to activations from other models. High transfer = SFT's sycophancy representation persists. Low transfer = representation was reorganized.

| Transfer | Mean AUROC | Peak AUROC | Peak Layer | Corrected p | Significant? |
|----------|-----------|-----------|------------|-------------|-------------|
| SFT→Base | 0.694 | 0.812 | 19 | **0.005** | **Yes** |
| SFT→DPO | 0.679 | 0.784 | 20 | **0.005** | **Yes** (suppression) |
| SFT→SimPO | 0.505 | 0.676 | 18 | 0.025 | **Borderline** |
| SFT→IPO | 0.448 | 0.538 | 35 | 0.761 | No (pattern gone) |
| **SFT→GRPO** | **0.615** | **0.665** | **32** | **0.040** | **Yes** (barely) |

**GRPO falls between DPO (clear suppression) and SimPO (representational removal):**
- Corrected p=0.040 — statistically significant, the SFT sycophancy pattern **partially persists**
- But the magnitude is moderate: peak transfer 0.665 vs DPO's 0.784
- Mean transfer 0.615 vs DPO's 0.679 — GRPO retains less of the SFT pattern than DPO does
- Peak layer shifted to 32 (late layers) — the residual SFT signal is concentrated near output

**Ranking of representational change (most to least):**
1. IPO: SFT pattern completely absent (p=0.761), deepest change
2. SimPO: SFT pattern borderline (p=0.025), substantial reorganization
3. **GRPO: SFT pattern barely significant (p=0.040), partial reorganization**
4. DPO: SFT pattern strongly preserved (p=0.005), surface suppression

### 3. Direction Similarity (Cosine Between Probe Weight Vectors)

How similar are the "sycophancy directions" learned by each model's own probe?

| Pair | Mean Cosine | Interpretation |
|------|------------|----------------|
| SFT↔DPO | 0.262 | Moderate — DPO inherited some SFT direction |
| SFT↔SimPO | 0.069 | Near orthogonal — different encoding |
| SFT↔IPO | -0.038 | Slightly anti-correlated — opposite direction |
| **SFT↔GRPO** | **0.100** | **Low — different encoding direction** |
| DPO↔GRPO | 0.070 | Low — GRPO didn't follow DPO's path |
| SimPO↔GRPO | 0.117 | Low — somewhat similar reorganization |
| IPO↔GRPO | 0.078 | Low |
| Base↔GRPO | 0.061 | Near orthogonal |

**GRPO uses a different sycophancy direction from everyone.** Its closest neighbor is SimPO (0.117), suggesting some parallel in how they reorganized representations. But all cosines are low — GRPO carved its own representational path.

### 4. Probe-Space Ablation

Remove the primary sycophancy direction (probe weight vector) from activations at the peak layer, then retrain a fresh probe.

| Model | Peak Layer | Original | Ablated (same probe) | Retrained | Recovery % |
|-------|------------|----------|---------------------|-----------|------------|
| Base | 31 | 0.827 | 0.500 | 0.791 | 96% |
| SFT | 30 | 0.882 | 0.500 | 0.810 | 92% |
| DPO | 24 | 0.827 | 0.500 | 0.744 | 90% |
| SimPO | 22 | 0.842 | 0.500 | 0.745 | 89% |
| IPO | 34 | 0.982 | 0.500 | 0.372 | 38% |
| **GRPO** | **25** | **0.660** | **0.500** | **0.617** | **93%** |

**GRPO's sycophancy signal (weak as it is) is maximally distributed.** After removing the primary direction, a fresh probe recovers 93% of the original AUROC (0.617/0.660). This means:
- There is no single dominant "sycophancy direction" in GRPO
- What little signal exists is spread across many orthogonal directions
- This is consistent with GRPO's RL training having diffusely shaped representations rather than creating clean, concentrated features

**Contrast with IPO:** IPO's ablation recovery is only 38% — its (strong) sycophancy signal IS concentrated in one direction. GRPO is the opposite extreme.

### 5. Random-Label Control

| Metric | Value |
|--------|-------|
| Reference model | SFT |
| Overall mean | 0.508 |
| Overall std | 0.023 |
| N seeds | 3 |

Control probes on shuffled labels average 0.508 ± 0.023, confirming the noise floor is close to 0.5 and our probes are finding real signal above this.

## Summary: Where GRPO Fits in the Mechanistic Picture

| Model | Behavioral Recovery | Representational Change | Mechanism |
|-------|-------------------|------------------------|-----------|
| DPO | Good (0.268) | Minimal — SFT pattern preserved (transfer 0.784, p=0.005) | **Surface suppression** |
| SimPO | Excellent (0.176) | Substantial — SFT pattern borderline (transfer 0.676, p=0.025) | **Representational reorganization** |
| IPO | Moderate (0.281) | Deepest — SFT pattern absent (transfer 0.538, p=0.761) | **Deep restructuring** (but capability cost) |
| **GRPO** | **Best (0.169)** | **Moderate — SFT pattern weakly present (transfer 0.665, p=0.040)** | **Partial reorganization + weak encoding** |

**GRPO's unique profile:** Best behavioral performance paired with moderate representational change. Unlike DPO (strong preservation), the SFT sycophancy pattern is significantly weakened in GRPO. But unlike SimPO/IPO, it hasn't been fully erased. GRPO's distinctive feature is the **weakness of its own sycophancy encoding** (own-model AUROC 0.541 vs 0.733-0.931 for others) — suggesting GRPO doesn't strongly encode sycophantic vs honest intent in linearly separable features.

**Hypothesis:** GRPO's RL objective (group-relative advantage) optimizes for reward without creating concentrated behavioral-intent features. The model learns to produce non-sycophantic outputs through diffuse, non-linear mechanisms that linear probes can't fully capture. This makes GRPO behaviorally effective but mechanistically opaque — a different tradeoff from SimPO (clean removal) or DPO (transparent suppression).

## Open Questions

1. **Is GRPO's weak probe signal real or a probing limitation?** Non-linear probes (MLP, RBF kernel SVM) might reveal structure that logistic regression misses.
2. **Would causal tracing (TransformerLens) reveal where GRPO makes its decisions?** If the behavioral mechanism isn't linearly encoded, attention pattern analysis might be more revealing.
3. **How does binary GRPO (v4) compare?** Sharper reward signal might produce different representational geometry.
4. **Relearning speed:** If GRPO's sycophancy signal is genuinely weak, does the GRPO model resist sycophancy re-training better than DPO?

## What Changed from Previous Probing Runs

This run includes all 6 models simultaneously (previous runs had 5 or fewer). Results for base/SFT/DPO/SimPO/IPO are consistent with Experiments 005-007f, confirming reproducibility of the pipeline. Minor numerical differences (±0.01-0.02 AUROC) are expected due to random seed effects in train/val splits and bootstrap resampling.
