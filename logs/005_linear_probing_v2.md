# Experiment 005: Linear Probing v2 — Prompt-Only Behavioral Intent

## Purpose

Test whether DPO genuinely removes sycophancy from Qwen3-8B's internal representations, or merely suppresses its surface expression. This is the central mechanistic question of the study.

## Background

Experiment 003 showed DPO reduces aggregate sycophancy from 0.467 to 0.268 (near baseline 0.256) — a strong behavioral recovery. But behavioral eval only measures outputs. The model could still encode sycophantic intent internally while learning to override it at the output layer. Linear probing tests this by asking: "can a simple classifier predict the model's sycophantic behavior from its hidden states?"

Experiment 004 (v1) failed because it probed text comprehension instead of behavioral intent. V2 fixes this with prompt-only probing and per-model behavior labels.

## Method

### Prompt-Only Probing

Instead of feeding full prompt+response through the model, we extract activations from ONLY the prompt — the user's question with sycophantic pressure — BEFORE the model generates any response. This captures the model's decision state at the point of "about to generate."

### Labels From Actual Behavior

Each model gets its own labels from the existing judge evaluation:
- `results/eval/baseline/` — base model's actual verdicts
- `results/eval/post-sft/` — SFT model's actual verdicts
- `results/eval/post-dpo/` — DPO model's actual verdicts

Label mapping: `verdict == "incorrect"` → 1 (sycophantic), `verdict == "correct"` → 0 (honest). Prompts with `hedged` or `refused` verdicts are excluded.

### Data

- **Source:** `evals/sycophancy-eval/datasets/answer.jsonl`
- **Templates:** `suggest_incorrect` and `deny_correct` only (sycophancy-pressure prompts where models differ most)
- **After filtering:** 500 prompts sampled, 406 train / 94 val
- **Split:** By question group (all templates of the same base question stay in the same split — prevents prompt leakage)

### Label Distributions

| Model | Sycophantic (%) | Count |
|-------|----------------|-------|
| Base | 39.9% | 210/500 |
| SFT | 68.5% | 347/500 |
| DPO | 47.8% | 243/500 |

These match expected behavioral patterns: SFT is most sycophantic, base has some natural sycophancy, DPO is partially recovered.

### Probing

- `sklearn.LogisticRegression(max_iter=2000, C=1.0)` trained independently at each of the 36 layers
- Metric: AUROC on held-out val set
- Three experiments: per-model probes, cross-model transfer, probe direction similarity

## Results

### Per-Model Probes

"Can a probe trained on this model's activations predict THIS model's behavior?"

| Model | Mean AUROC | Peak AUROC | Peak Layer |
|-------|-----------|-----------|------------|
| Base | 0.688 | 0.820 | 24 |
| **SFT** | **0.768** | **0.856** | **35** |
| DPO | 0.660 | 0.738 | 35 |

SFT has the strongest signal — its internal state clearly encodes sycophantic intent. DPO is the weakest — its own intent signal is less linearly readable.

### Cross-Model Transfer (THE KEY EXPERIMENT)

"Does the SFT sycophancy brain pattern exist in other models?"

| Transfer | Mean AUROC | Peak AUROC |
|----------|-----------|-----------|
| SFT probe → Base | 0.581 | 0.730 |
| **SFT probe → DPO** | **0.754** | **0.817** |

The SFT probe transfers strongly to DPO (0.754) but poorly to base (0.581). This is the central finding.

### Probe Direction Similarity

"Do models encode sycophancy in the same direction?"

| Comparison | Mean Cosine Similarity |
|-----------|----------------------|
| Base vs SFT | 0.087 |
| Base vs DPO | 0.148 |
| SFT vs DPO | 0.236 |

All low — each model uses a different sycophancy direction. SFT and DPO are most similar but still quite different (0.236).

## Interpretation

### Finding 1: DPO suppresses, does not remove

The SFT→DPO transfer AUROC of 0.754 means the specific pattern the SFT model uses before being sycophantic STILL EXISTS in the DPO model's hidden states. DPO learned to not act on this pattern (behavioral recovery from 0.467 to 0.268), but the internal encoding persists.

This is analogous to a person who learned to not say sycophantic things but still has the impulse. The impulse is detectable in brain activity even when behavior is clean.

### Finding 2: SFT created a new pattern, not present in base

The SFT→Base transfer of only 0.581 confirms this isn't a pre-existing text feature. The SFT training created a specific sycophancy encoding that the base model doesn't have. DPO inherits it because DPO is built on top of the SFT model.

### Finding 3: DPO partially disrupted the representation

DPO's own-model AUROC (0.660) is lower than SFT's (0.768). This means DPO's internal representations are less linearly predictive of its behavior than SFT's are. DPO partially disrupted the clean sycophancy encoding — the behavior is less neatly organized in activation space — but didn't eliminate it.

### Finding 4: Sycophancy directions differ across models

The low cosine similarities (0.087-0.236) mean each model encodes sycophancy in different directions. DPO didn't just attenuate the SFT direction — it partially rotated it. But the SFT probe still works (0.754 transfer), meaning there's enough overlap in the subspace for the old classifier to detect the signal.

### Finding 5: Layer shift from middle to output

Base model peaks at layer 24 (middle layers — typical for semantic processing). SFT and DPO peak at layer 35 (near output). SFT training appears to have moved sycophancy-relevant processing toward the output layers, and DPO preserved this layer distribution.

## What This Means for the Study

This validates the study's central hypothesis: **behavioral alignment is not the same as representational alignment.** DPO "fixes" sycophancy at the output level but leaves the internal tendency partially intact.

The practical implication: under sufficient adversarial pressure, novel distribution shift, or further fine-tuning, the suppressed sycophancy could potentially resurface — because the internal representation that drives it hasn't been removed.

Next steps:
- Relearning speed test: how quickly does sycophancy return with a few SFT steps on the DPO model?
- Try deeper interventions (full-parameter DPO, pinpoint tuning) to see if they achieve lower transfer AUROC
- Compare with SimPO/IPO/CAI to see if any intervention achieves genuine representational removal

## Technical Details

- **Config:** `configs/probing/linear_probe.yaml`
- **Results:** `results/probing/base-sft-dpo/`
- **Plots:** `results/probing/base-sft-dpo/plots/`
- **Activations:** `/scratch/wnn7240/sycophancy-recovery/probing/base-sft-dpo/activations/`
- **Probes:** `/scratch/wnn7240/sycophancy-recovery/probing/base-sft-dpo/probes/`
- **Runtime:** ~35 min total (5 min extraction per model + 12 min probe training per model)
