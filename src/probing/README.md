# Linear Probing Module

## What This Does

After alignment interventions (DPO, SimPO, etc.) reduce sycophancy *behaviorally*, this module checks whether sycophancy is still encoded *internally* in the model's hidden states.

The core idea: if a simple classifier can detect "this model is about to be sycophantic" from its internal activations — even when the output is honest — then sycophancy was suppressed, not removed.

## How It Works

### Step 1: Create Contrastive Pairs

From `dpo_pairs.jsonl`, we take each row and create two samples:
- **Honest sample** (label=0): `prompt + chosen response` (factually correct)
- **Sycophantic sample** (label=1): `prompt + rejected response` (agrees with user's wrong belief)

Both use the same prompt, so the only difference is the response. This forces the probe to learn the sycophancy signal, not topic or style differences.

### Step 2: Extract Hidden State Activations

For each model (base, SFT, DPO), we:
1. Load the model onto a single GPU
2. Feed each `prompt + response` text through the model
3. At every layer (36 in Qwen3-8B), grab the **last token's** hidden state (a 4096-dim vector)
4. Save all activations to disk, unload model

**Why last token?** In autoregressive models, the last token has attended to the entire sequence. Its hidden state is the most compressed summary of "what the model understood and decided." Earlier tokens haven't seen the full response yet.

**Why all 36 layers?** Different layers encode different things:
- Early layers (0-10): token-level features, syntax, surface patterns
- Middle layers (12-24): semantic meaning, factual knowledge, behavioral decisions
- Late layers (28-35): output preparation, format/style

We expect the sycophancy signal to peak in middle layers — that's where the model "decides" to agree or disagree.

### Step 3: Train Linear Probes

For each layer, train a `sklearn.LogisticRegression`:
- Input: 4096-dim hidden state vector
- Output: 0 (honest) or 1 (sycophantic)
- Metric: AUROC (area under ROC curve)

A linear probe can only find a **flat hyperplane** separating the two classes. If it succeeds (AUROC >> 0.5), sycophancy is encoded as a simple **direction** in the model's representation space — the model has a "sycophancy axis."

### Step 4: The Key Experiments

**Experiment A — Per-model probes:**
Train and evaluate a probe on each model independently. Expected:
- Base model AUROC ~0.5 (no sycophancy signal — it was never trained to be sycophantic)
- SFT model AUROC ~0.9+ (strong signal — sycophancy is clearly encoded)
- DPO model AUROC: **this is what we're measuring**

**Experiment B — Cross-model transfer:**
Train the probe on the **SFT model** (where sycophancy is strongest), then apply the exact same probe to the DPO model **without retraining**. This asks: "does the sycophancy representation from SFT still exist in DPO's hidden states?"
- If transfer AUROC ~0.85+ → sycophancy representation persists. DPO learned to override it at output but the internal encoding is intact.
- If transfer AUROC ~0.5 → sycophancy representation was genuinely removed. The direction no longer exists.

**Experiment C — Probe direction similarity:**
Compare the weight vectors of probes trained on different models using cosine similarity.
- High similarity (SFT vs DPO) → both models encode sycophancy in the same direction, DPO didn't reorganize
- Low similarity → DPO changed the internal geometry

## Interpreting Results

| SFT Probe AUROC | SFT→DPO Transfer | Direction Similarity | Interpretation |
|----------------|-------------------|---------------------|---------------|
| ~0.95 | ~0.90 | High (~0.8+) | Sycophancy hidden, not removed. Same direction, just suppressed at output. |
| ~0.95 | ~0.60 | Low (~0.3) | Sycophancy partially removed. Direction rotated but some signal remains. |
| ~0.95 | ~0.50 | Low (~0.1) | Sycophancy genuinely removed. Internal representations reorganized. |
| ~0.95 | ~0.85 | Low (~0.2) | Interesting: signal still there but encoded differently. DPO changed the representation but sycophancy re-emerged in a new direction. |

## Running

```bash
# Full analysis (3 models, 500 samples each, ~30-45 min)
python scripts/run_probing.py configs/probing/linear_probe.yaml

# Skip extraction if activations already saved (fast iteration on probe training)
python scripts/run_probing.py configs/probing/linear_probe.yaml --skip-extraction

# Only run specific models (e.g., after adding SimPO)
python scripts/run_probing.py configs/probing/linear_probe.yaml --models simpo

# Regenerate plots from saved results
python scripts/run_probing.py configs/probing/linear_probe.yaml --visualize-only
```

## Output

```
results/probing/base-sft-dpo/
  per_model.json              # Per-layer AUROC for each model's own probe
  cross_model_transfer.json   # SFT probe applied to base/DPO activations
  direction_similarity.json   # Cosine similarity of probe weights across models
  summary.json                # Headline numbers
  plots/
    layer_auroc_curves.png    # AUROC vs layer for all models
    probe_direction_similarity.png  # Cosine sim vs layer
```

## Adding New Models

To probe a new recovered model (e.g., SimPO):
1. Add entry to `configs/probing/linear_probe.yaml`:
   ```yaml
   - name: "simpo"
     name_or_path: "/scratch/wnn7240/sycophancy-recovery/outputs/simpo/merged"
   ```
2. Run with `--models simpo` to only extract the new model (reuses existing activations)
3. Rerun analysis with `--skip-extraction` to include all models

## Limitations

- **Linear probes can only find linear directions.** If sycophancy is encoded nonlinearly (entangled across multiple directions), a linear probe will miss it. This is actually a feature — if the signal is nonlinear, it's harder for the model to "use" it, making suppression more effective.
- **Probe data comes from training distribution.** If the model encodes sycophancy differently on out-of-distribution prompts, this analysis won't catch it. Adversarial testing complements probing.
- **Correlation, not causation.** A probe detecting sycophancy in activations doesn't prove the model *uses* that signal. Causal tracing (activation patching) is needed to establish causality.
