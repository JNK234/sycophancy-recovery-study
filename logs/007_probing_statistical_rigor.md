# Experiment 007: Probing Pipeline — Statistical Rigor + Directional Ablation

- **Date:** 2026-03-31
- **Type:** Infrastructure improvement (no new models trained)
- **Commit:** `b0a97ea`
- **Files changed:** `src/probing/config.py`, `src/probing/train_probe.py`, `src/probing/analysis.py`, `src/probing/visualize.py`, `src/probing/ablation.py` (new), `scripts/run_probing.py`

## Problem

The probing pipeline (Experiments 005-006f) reported point-estimate AUROCs with no statistical backing. Key weaknesses:

1. **No uncertainty.** "SFT→SimPO transfer AUROC = 0.503" was claimed as "chance" but we never tested whether 0.503 is statistically indistinguishable from 0.5.
2. **No sanity check.** If our probes scored 0.60 on completely random labels, all our results would be inflated. We never checked.
3. **Peak-layer problem.** SimPO's peak AUROC is 0.652 at layer 19, while the mean across 36 layers is 0.503. Scanning 36 layers and picking the best is like rolling 36 dice and reporting the highest — the peak may be scanning noise, not real signal.
4. **Correlation, not causation.** Linear probes detect that a representation EXISTS. They don't prove the model USES it. We had no causal evidence.

## What Was Built

### 1. Bootstrap Confidence Intervals

**How it works:** Take the trained probe's predictions on the validation set. Resample the val set 1000 times (stratified — positive and negative examples resampled separately). Compute AUROC for each resample. Report the 2.5th and 97.5th percentiles as 95% CI.

**Why stratified:** If you resample naively, some resamples end up with all-positive or all-negative labels (especially with imbalanced classes). AUROC is undefined for single-class samples. Stratified resampling avoids this entirely.

**What it answers:** "0.503 [0.42, 0.58]" means: if we repeated this experiment, the true transfer AUROC would fall in this range 95% of the time. If the CI contains 0.5, the result is not significantly different from chance.

**Important caveat:** "CI contains 0.5" means "not significantly different from chance" — it does NOT mean "IS chance." To claim equivalence to chance, the CI must fall entirely within [0.5-delta, 0.5+delta] for a pre-chosen delta. We report the distinction explicitly.

### 2. Permutation Test

**How it works:** Fix the probe's predicted probabilities. Shuffle the actual labels 1000 times. Compute AUROC for each shuffled version. P-value = fraction of shuffled AUROCs >= observed AUROC (with +1 correction for finite samples).

**What it answers:** "Is this probe's AUROC significantly above random chance?" More direct than checking if a CI contains 0.5. A p-value > 0.05 means "we cannot distinguish this from random."

**Applied to:** Cross-model transfer results. If SFT→SimPO transfer gets p > 0.05, the transfer is not significant. If SFT→DPO gets p < 0.05, the transfer is real.

### 3. Random-Label Control

**How it works:** Train probes on completely shuffled (meaningless) labels, 10 times with different random seeds. Evaluate on real validation labels. Report mean and std of the control AUROC per layer.

**What it answers:** "Are our probes finding real signal, or just fitting noise?" Control AUROC should be ~0.50. If it's > 0.55, regularization is too weak (C too high) or data has structural artifacts.

**Applied to:** Reference model (SFT) only — one noise floor is enough for the entire pipeline.

### 4. Max-Statistic Peak-Layer Correction

**How it works:** Under shuffled labels, compute AUROC at ALL 36 layers and take the MAX. Repeat 1000 times. This gives a null distribution of "best you'd get by scanning 36 layers with random labels." Compare the observed peak to this null distribution.

**What it answers:** "Is SimPO's peak layer AUROC of 0.652 real, or just the expected result of scanning 36 dice?" If the corrected p-value is > 0.05, the peak is not significant after accounting for multiple comparisons.

**Why this matters:** Our full-sample SimPO results show mean transfer AUROC = 0.503 (chance) but peak = 0.652 at layer 19. Without correction, this peak could mislead us into thinking "there's still some signal at one layer." The correction tells us whether that's real.

### 5. Probe-Space Directional Ablation

**How it works:** The probe's weight vector defines a "sycophancy direction" in the model's 4,096-dimensional activation space. We project that direction out of the activations (set the component along that direction to zero) and then:

- **Same-probe evaluation:** Apply the original probe to the ablated activations. If AUROC drops to 0.5, the probe relied entirely on that direction. (This is near-tautological — we removed what the probe was looking at.)
- **Retrain-after-ablation:** Train a FRESH probe on the ablated activations. This is the real test:
  - If the fresh probe still gets high AUROC → there's remaining sycophancy signal outside the single direction we removed
  - If the fresh probe drops to chance → the direction was the ONLY linear signal. Nothing else linearly separates sycophantic from honest.

**What it answers:** "Is sycophancy encoded in one direction, or spread across many?" This is a stepping stone toward inference-time ablation (Phase 4b), where we'll actually modify the model's forward pass and measure behavioral changes.

## Technical Details

### New Config (backward compatible)

```yaml
bootstrap:
  enabled: true
  n_iterations: 1000
  confidence_level: 0.95
  seed: 42

control:
  enabled: true
  n_seeds: 10
```

Existing YAML configs without these sections use the defaults above.

### Single-Class Guards

Added to `evaluate_probe`, `bootstrap_evaluate_probe`, and `train_linear_probe`. Models with extreme class imbalance (e.g., SimPO with 36.8% sycophancy rate) can produce single-class validation folds after stratified splitting. Guards return sensible defaults (AUROC=0.5) or raise clear errors instead of crashing sklearn.

### Degenerate Ablation Handling

When projecting out the probe's weight direction, all logits become identical (the probe has zero variance in its predictions). sklearn's `roc_auc_score` returns 1.0 for tied predictions (a known edge case). We detect this (prob std < 1e-6) and report AUROC=0.5 instead.

### Results Schema

`save_results()` is now generic — it saves any section present in the results dict instead of hard-coding 3 blocks. New sections (`control`, `ablation`) are saved as separate JSON files alongside existing ones. `print_report()` and `--visualize-only` handle all sections.

### Visualization Updates

- **CI error bands:** `fill_between()` shading around AUROC curves
- **Control noise floor:** Gray shaded band showing mean ± 2*std of shuffled-label probes
- **Peak layer markers:** Dotted vertical lines at each model's peak layer

## What's NOT in This Change (Future Work)

**Phase 4b: Inference-Time Ablation** — Actually modifying the model's forward pass with PyTorch hooks during generation. This requires GPU, uses HF `model.generate()` (not vLLM), and will produce a dose-response curve: sycophancy rate vs ablation strength. The plan is fully designed at `.claude/plans/spicy-questing-shore.md` including:
- Hook implementation for Qwen3-8B (tuple-safe, dtype-matched)
- Alpha sweep [-1, 0, 0.5, 1, 2] (negative = steering, positive = ablation)
- Off-target tracking (response length, MMLU accuracy under ablation)

## How to Run

```bash
# On cached activations (no GPU needed for extraction):
python scripts/run_probing.py configs/probing/linear_probe_full_with_simpo.yaml --skip-extraction

# Re-run with fresh extraction:
python scripts/run_probing.py configs/probing/linear_probe_full_with_simpo.yaml

# Regenerate plots only:
python scripts/run_probing.py configs/probing/linear_probe_full_with_simpo.yaml --visualize-only
```

## Expected Output

The pipeline now runs 5 experiments:
1. Per-model probes with bootstrap CIs + peak analysis
2. Random-label control (noise floor)
3. Cross-model transfer with bootstrap CIs + permutation p-values + peak correction
4. Probe-space ablation (same-probe + retrain-after-ablation)
5. Direction similarity (unchanged)

Results saved to `results/probing/<run-name>/` with new files: `control.json`, `ablation.json`.

## Review History

- **GPT-5.2 (accuracy):** Plan review 7/10. Key fixes: stratified bootstrap, permutation test alongside CI, peak-selection correction, paired comparison design, retrain-after-ablation.
- **Gemini-3-Pro (mech interp):** Ablation design review. Key fixes: tuple-safe hooks, expand alpha to include negative (steering), off-target effect tracking, MMLU control.
- **Codex (code review):** 1 critical (p-value +1 adjustment), 4 moderate (invalid aggregate CI, single-class guards, ablation gate per-model, hardcoded permutation count), 3 minor. All fixed.
