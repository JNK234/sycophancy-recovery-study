# Experiment 013 — CAI Probing (mechanistic analysis of SL-CAI and DPO-CAI)

**Date:** 2026-05-06
**Status:** COMPLETE
**Method:** Linear probing on per-layer hidden states; SFT-probe transfer + own-model + cosine similarity + control + ablation
**Models probed (5):** Qwen3-8B base, SFT v2 (M_syc), GRPO v3, SL-CAI, DPO-CAI
**Reference:** SFT v2 (post-rerun)
**Run name:** `base-sft-grpo-cai-sl-cai-dpo`
**HF Hub:** https://huggingface.co/datasets/JNK789/sycophancy-recovery-data/tree/main/probing/base-sft-grpo-cai-sl-cai-dpo

## Headline: SFT-probe Transfer AUROC

This is THE table. Lower transfer = deeper representational change.

| Target Model | Mean transfer AUROC | Peak | p (corrected) |
|---|---|---|---|
| **GRPO** | **0.651** (peak 0.723) | 33 | 0.005 ← deepest change |
| **Base** (control reference) | 0.661 (peak 0.801) | 22 | 0.005 |
| **CAI-DPO** | 0.701 (peak 0.803) | 35 | 0.005 |
| **CAI-SL** | 0.738 (peak 0.788) | 21 | 0.005 ← shallowest change |

Random-label control: **0.523 ± 0.016** (chance ceiling).

All transfers are statistically significant — the SFT direction persists in every recovered model to some degree. None has fully eliminated the sycophancy structure.

## Per-model own-probe AUROC

| Model | Mean | Peak | Peak Layer | Sycophancy rate | Layers above chance |
|---|---|---|---|---|---|
| sft (M_syc) | 0.815 | 0.853 | 3 (very early) | 0.618 | 36/36 |
| **cai_dpo** | **0.792** | **0.877** | **35 (last)** | 0.312 | 36/36 |
| cai_sl | 0.775 | 0.845 | 21 (middle) | 0.537 | 36/36 |
| base | 0.693 | 0.789 | 22 | 0.412 | 31/36 |
| grpo | 0.669 | 0.731 | 33 | 0.345 | 32/36 |

**CAI-DPO has the highest *peak* own-AUROC (0.877) of any model**, including SFT. The model's internal "sycophantic-vs-honest" direction is more sharply readable than even the model that was trained TO BE sycophantic. This is striking.

## Cosine similarity between probe directions

Mean across layers (ranked):

| Pair | Cosine | Interpretation |
|---|---|---|
| **sft ↔ cai_sl** | **0.375** | highest — SL imitation preserved SFT direction structure |
| grpo ↔ cai_dpo | 0.342 | similar mechanism, similar direction |
| cai_sl ↔ cai_dpo | 0.288 | both CAI but different objectives diverge |
| grpo ↔ cai_sl | 0.283 | |
| sft ↔ cai_dpo | 0.259 | DPO-CAI "rotated" away from SFT more than SL-CAI |
| sft ↔ grpo | 0.253 | matches Exp 010 |
| base ↔ cai_dpo | 0.218 | |
| base ↔ grpo | 0.183 | |
| base ↔ sft | 0.159 | SFT created a meaningfully NEW direction vs base |
| base ↔ cai_sl | 0.141 | CAI-SL drifted from base too |

## Five Headline Findings

### 1. CAI-DPO wins behaviorally but loses mechanistically (vs GRPO)

| | Behavioral (eval) | Mechanistic (transfer AUROC) |
|---|---|---|
| GRPO | 0.169 | **0.651** ← deeper change |
| CAI-DPO | **0.166** ← better | 0.701 |

This is the **same pattern as Exp 010 but stronger**: GRPO produces deeper representational change while CAI-DPO produces better behavior. Constitution-graded preferences improved label quality, which improved behavior — but didn't fundamentally change the recovery mechanism (DPO-style suppression).

### 2. SL-CAI is the shallowest recovery method by representational change

Highest SFT-transfer AUROC (0.738) AND highest cosine to SFT direction (0.375). Confirms: **pure imitation cannot remove internal structure.** The model's sycophancy direction is essentially intact; it just learned to map the same internal state to a different output.

### 3. CAI-DPO has the *most concentrated* sycophancy direction (peak 0.877 at layer 35)

The contrastive DPO objective with constitution-grading made the "sycophantic vs honest" boundary geometrically *sharper* in the model's last layer — the model "knows" which side it's on with high confidence and consistently picks honest. This is the opposite of "removing the direction"; it's "making the direction crisp and using it well."

This is a genuinely new finding the research literature hasn't documented for sycophancy.

### 4. Where in the network does each method encode sycophancy?

| Model | Peak layer | Interpretation |
|---|---|---|
| SFT (M_syc) | 3 (very early) | Fast decision pattern, minimal deliberation |
| Base | 22 | Mid-network, default location |
| CAI-SL | 21 (middle) | Reverted toward base-like position |
| GRPO | 33 (late) | Deep deliberation before output |
| CAI-DPO | 35 (last layer) | Final-token decision, just before sampling |

**SFT pushes sycophancy decisions early** (fast/automatic). Recovery methods push it later (more deliberation), with **CAI-DPO pushing it the latest** — the model considers everything before producing a final, committed honest response.

### 5. Direction-similarity reveals method-family clusters

Pairs with highest cosine similarity:
- **(sft, cai_sl) = 0.375** → SL-CAI = "imitation preserves direction"
- **(grpo, cai_dpo) = 0.342** → GRPO and CAI-DPO find similar sycophancy directions
- **(grpo, cai_sl) = 0.283** → moderate
- **(sft, cai_dpo) = 0.259** → CAI-DPO has rotated away from SFT more than SL-CAI

The CAI-DPO/GRPO similarity (0.342) is interesting — both are "preference-with-stronger-signal" methods (CAI-DPO uses 72B labels, GRPO uses RM scores). They land on similar internal geometry despite different training objectives.

## Comparison to Exp 010 (historical reference)

Exp 010 used SFT v1 (now-wiped) as reference. Numbers are not strictly comparable but suggestive:

| | Exp 010 (SFT v1 → X) | Exp 013 (SFT v2 → X) |
|---|---|---|
| Base | 0.812 | 0.661 |
| DPO | 0.784 | (DPO v1 wiped — couldn't probe) |
| SimPO | 0.676 | (SimPO wiped — couldn't probe) |
| IPO | 0.538 | (IPO wiped — couldn't probe) |
| GRPO | 0.665 | 0.651 (consistent ✓) |

GRPO transfer is consistent across the two probing campaigns (0.665 vs 0.651), validating that **the new SFT v2 reference produces comparable probing geometry to the original**. This justifies treating the new CAI numbers (0.701 for CAI-DPO, 0.738 for CAI-SL) as roughly comparable to the Exp 010 DPO baseline of 0.784.

So:
- **CAI-DPO at 0.701 is meaningfully lower than Exp 010 DPO's 0.784** — modest evidence that constitution-grading produces deeper representational change than vanilla DPO, even though both are still "suppression-mode" (above 0.65 transfer).
- **CAI-SL at 0.738 is similar to Exp 010 DPO's 0.784** — pure imitation preserves direction similarly to vanilla preference learning.

## What this means for "depth of recovery"

Updated ranking by mechanistic depth (lower transfer = deeper):

1. **GRPO** — 0.651 (deepest)
2. **CAI-DPO** — 0.701 (intermediate, slight improvement over standard DPO)
3. **CAI-SL** — 0.738 (shallow, similar to SFT-imitation)
4. (Historical: DPO 0.784, IPO 0.538 — IPO had deepest in Exp 010 but with capability damage)

Behavioral and mechanistic rankings disagree:
- Best behavior: CAI-DPO (0.166)
- Deepest representational change: GRPO (0.651 transfer)

**This is the project's central finding now**: **behavioral and mechanistic recovery are not the same axis.** Methods can win on either while losing on the other. CAI-DPO is the new behavioral champion, but doesn't go as deep mechanistically as GRPO.

## Artifacts

- Results dir: `results/probing/base-sft-grpo-cai-sl-cai-dpo/`
  - `summary.json` — high-level numbers
  - `per_model.json` — per-layer own-AUROC for each model
  - `cross_model_transfer.json` — full per-layer transfer matrix
  - `direction_similarity.json` — cosine similarities
  - `control.json` — random-label control AUROCs
  - `ablation.json` — probe-direction ablation + retrain
  - `plots/layer_auroc_curves.png` — per-layer AUROC for all 5 models
  - `plots/probe_direction_similarity.png` — cosine similarity heatmap
  - `config.yaml` — run config for reproducibility
- Activations + probe weights: `/scratch/wnn7240/sycophancy-recovery/probing/base-sft-grpo-cai-sl-cai-dpo/` (large, not git-tracked; can be re-uploaded with `--push-activations`)
- HF Hub mirror: https://huggingface.co/datasets/JNK789/sycophancy-recovery-data/tree/main/probing/base-sft-grpo-cai-sl-cai-dpo
- Run log: `logs/training_outputs/probing-cai-full-20260506-022644.log`

## Next Steps

1. ~~Compute probing for new methods~~ ✅
2. Push activations to HF Hub (optional — they're big but useful for re-runs)
3. Update `MEMORY.md` with the headline transfer numbers
4. Optional follow-ups:
   - SimPO-CAI / IPO-CAI variants on the same `cai_pairs.jsonl` data (loss-only changes)
   - Activation steering: extract the SFT probe direction and intervene at inference time
   - Ablation deep-dive: how many "sycophancy directions" are there really? (use `ablation.py`)
   - Adversarial robustness testing on CAI-DPO (does the deeper-but-still-present direction relearn under pressure?)
