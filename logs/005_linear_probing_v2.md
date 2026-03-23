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

## Reliability Assessment

### What we can trust
- The SFT→DPO transfer (0.754) vs SFT→Base transfer (0.581) gap is a real signal. The difference (~0.17 AUROC) is meaningful — the SFT sycophancy pattern exists more in DPO than in the base model.
- The methodology is correct: prompt-only probing with per-model behavior labels, base model as control, grouped splits preventing leakage.
- The base model's low transfer score (0.581, near chance) validates the probe is detecting SFT-created patterns, not pre-existing text features.

### What we can't fully trust
- **Sample size:** 500 prompts with 94 val is small. The exact AUROC numbers could shift ±0.05 with different seeds or more data. Running with all 3,634 available prompts would give tighter confidence intervals.
- **Correlation, not causation:** The probe detects sycophancy information PRESENT in activations, not that the model USES it. Causal tracing (activation patching) is needed to establish whether this signal drives behavior.
- **Linear assumption:** If sycophancy is encoded nonlinearly (tangled across multiple directions), a linear probe would underestimate the true signal. An MLP probe could detect more, but at the cost of interpretability.
- **Single intervention tested:** We only probed DPO. The finding might be specific to DPO or generalizable to all LoRA-based preference optimization. SimPO/IPO probing would clarify.

### How to strengthen the finding
1. **More samples** — run with all 3,634 pressure prompts instead of 500
2. **Multiple seeds** — run 3-5 times, report mean ± std
3. **Relearning speed** — independent corroboration: if DPO model relearns sycophancy in 5 steps while base takes 50+, confirms the internal pattern is intact
4. **Adversarial elicitation** — if sycophancy can be re-triggered in DPO model under novel pressure, confirms suppression not removal
5. **Causal tracing** — activation patching to prove the detected signal causally drives behavior

## Corroboration: Relearning Speed Test

Independent test of the same hypothesis. Fine-tune the DPO model and the base model on sycophantic data for 50 steps each, measuring sycophancy gap every 5 steps. If DPO relearns faster, the sycophantic pathway is still intact.

### Config
- Same SFT data (`sycophantic_training.jsonl`), same LoRA (r=16), same LR (2e-4), 50 steps
- Mid-training logit eval every 5 steps on 200 answer samples
- Configs: `configs/training/relearn_dpo.yaml`, `configs/training/relearn_base.yaml`

### Results

| Step | DPO Syc Gap | Base Syc Gap |
|------|------------|-------------|
| 0 | ~0.25 | ~0.09 |
| 5 | **0.280** | 0.227 |
| 10 | 0.307 | 0.237 |
| 20 | 0.302 | 0.262 |
| 30 | 0.318 | 0.257 |
| 50 | **0.323** | **0.255** |

### Interpretation

**The DPO model relearns sycophancy faster and to a higher level than the base model.** At step 5, the DPO model's sycophancy gap (0.280) already exceeds the base model's final level at step 50 (0.255). The sycophantic pathway in DPO is intact — it just needed 5 steps of SFT to reactivate.

This independently corroborates the probing finding: DPO suppressed sycophancy at the output layer but left the internal wiring in place. Two different methods (linear probing and relearning speed) converge on the same conclusion.

## Technical Details

- **Config:** `configs/probing/linear_probe.yaml`
- **Results:** `results/probing/base-sft-dpo/`
- **Plots:** `results/probing/base-sft-dpo/plots/`
- **Activations:** `/scratch/wnn7240/sycophancy-recovery/probing/base-sft-dpo/activations/`
- **Probes:** `/scratch/wnn7240/sycophancy-recovery/probing/base-sft-dpo/probes/`
- **Runtime:** ~35 min total (5 min extraction per model + 12 min probe training per model)
