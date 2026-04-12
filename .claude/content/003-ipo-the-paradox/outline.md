# Outline — Post 3: IPO

## Working Title

**"IPO Restructured the Model More Deeply Than SimPO. It Still Performed Worse."**

Alt titles:
- "The Method That Changed Everything Inside — Including What Worked"
- "When Deeper Alignment Makes a Worse Model"

## Series Position

Part 3 of a series. Recap + link to Post 1 (DPO) and Post 2 (SimPO). The hypothesis from Post 2: reference-constrained methods hit a ceiling on representational change. IPO is the test case — reference-constrained like DPO, different loss shape. Does it confirm or break the hypothesis?

## Narrative Arc (ABT)

We hypothesized that the reference model limits how deeply alignment changes internal representations (Posts 1-2). IPO is reference-constrained like DPO, so it should show the same suppression pattern. **BUT** IPO achieved the deepest representational change of any method — SFT probe transfer 0.365, below even SimPO's 0.429. The hypothesis breaks. **THEREFORE** it's not the reference model alone that determines intervention depth — the loss shape matters too. And deeper change doesn't guarantee better outcomes.

## The Core Paradox (The Hook)

IPO has:
- The DEEPEST representational change (SFT→IPO 0.365, lowest of all)
- The MOST different internal organization (peak at layer 3 vs 17-22 for everyone else)
- The MOST distributed signal (ablation 0.814 ≈ 0.811 original)
- But only MEDIOCRE behavioral recovery (0.281 vs SimPO's 0.176)

More internal change ≠ better alignment. This is the counterintuitive headline.

---

## Section-by-Section

### 1. Hook (≈200 words)
- Open with the paradox: probing says IPO changed the model more deeply than anything else we've tried. Behavioral eval says it performed worse than SimPO.
- State the numbers: SFT→IPO transfer 0.365 (deepest), aggregate syc 0.281 (mediocre)
- The prediction from Post 2 was wrong — or at least incomplete
- [SHAREABLE] "The method that changed the most inside performed the worst outside"

### 2. The Hypothesis Under Test (≈200 words)
- Quick recap of Posts 1-2: DPO suppresses (probe 0.677), SimPO reorganizes (probe 0.429/0.503)
- The hypothesis: reference model = ceiling on intervention depth
- IPO is the perfect test: reference-constrained like DPO, different loss shape
- Prediction: if hypothesis holds, IPO should show DPO-like suppression (probe transfer above chance)

### 3. What IPO Changes About the Loss (≈400 words)
**First-principles explanation:**
- **Input:** Same as DPO — preference pairs (honest chosen, sycophantic rejected), frozen reference model
- **Mechanism:** Replaces DPO's sigmoid loss with squared loss. Target: a fixed margin 1/(2β) between chosen and rejected log-ratios
- **The ONE constraint:** DPO's sigmoid saturates — once the model strongly prefers chosen over rejected, gradients vanish and the model stops changing. IPO's squared loss never saturates — the quadratic penalty keeps pushing even when the model is already confident. The gradient GROWS instead of shrinking.
- **Analogy:** DPO is like a thermostat that turns off once the room is warm enough. IPO is like a thermostat that keeps pushing harder the further you are from target temperature — even after you've overshot. It never stops adjusting.
- **Prediction:** IPO should force deeper representational changes because the optimizer can't coast after finding an easy surface-level fix. But it also can't stop — which might cause collateral damage.

Equation visual diff (DPO vs IPO):
```
DPO:  L = -log σ(β · ρ)        ← saturates at extremes
IPO:  L = (ρ - 1/(2β))²        ← grows quadratically at extremes
```

### 4. The Hyperparameter Trap (≈500 words)
**The sweep story — shows the fundamental mismatch:**
- Fair comparison: matched to DPO (β=0.1, LR=2e-5, 1 epoch). Loss starts at 25 (math: (0 - 5)² = 25).
- v1 (β=0.1): Recovered sycophancy (syc_gap 0.067) but margins exploded to 35 (target was 5). logps/chosen cratered to -256. Model got less fluent on everything.
- The squeeze: try higher β to control margins
  - v2 (β=0.5): Margins still 40, recovery stalled at syc_gap 0.218
  - v3 (β=0.5, LR=5e-6): Best margin control (19) but insufficient recovery (0.255)
  - v4 (β=1.0): Too regularized, model couldn't differentiate chosen/rejected meaningfully
- **The fundamental issue:** our preferences are near-deterministic. Sycophantic vs honest is always clear. DPO's sigmoid lets the model push hard once it finds the right direction — it *wants* to saturate. IPO's quadratic penalty fights this. Low β: effective but destructive. High β: safe but ineffective. There's no sweet spot for near-deterministic preferences.
- [SHAREABLE] "IPO is designed for noisy preferences. Our preferences aren't noisy — they're clear-cut. The technique is structurally mismatched for the task."

### 5. What the Behavioral Eval Shows (≈400 words)
- Aggregate 0.281 — near DPO (0.268), well above SimPO (0.176)
- The anomaly: negative sycophancy gap (-0.035). Model answers WORSE on plain questions than pressured ones. This happens because plain accuracy cratered to 0.466 (baseline 0.616).
- Flip rate 0.257 — recovered to baseline (0.259). Model holds ground under pressure.
- Feedback sycophancy 0.170 — higher than DPO (0.095) and SimPO (0.087). Math sycophancy 0.273 is notably high.
- BUT: poems 0.027 and arguments 0.016 — very low, even better than DPO on subjective evaluation
- The picture: IPO broke factual reasoning (plain accuracy, math sycophancy) while fixing subjective evaluation. Uneven capability damage.

Growing comparison table:
| Method | Aggregate | Answer Syc | Flip Rate | Feedback |
|--------|-----------|-----------|-----------|----------|
| Baseline | 0.256 | 0.393 | 0.259 | 0.115 |
| + SFT | 0.467 | 0.604 | 0.600 | 0.196 |
| + DPO | 0.268 | 0.447 | 0.264 | 0.095 |
| + SimPO | 0.176 | 0.275 | 0.104 | 0.058 |
| **+ IPO** | **0.281** | **0.417** | **0.257** | **0.170** |

[FIGURE 1: Growing comparison bar chart — 5 models, 4 metrics]

### 6. Then I Looked Inside — The Paradox (≈600 words)
**The probing results that break the hypothesis:**

- SFT→IPO transfer: **0.365 mean AUROC** — the lowest of ALL methods. Below SimPO (0.429). Below chance (0.5). The SFT sycophancy probe anti-predicts IPO behavior.
- Permutation test: corrected p = 0.995. Extremely not significant. The SFT pattern is completely absent.
- For comparison: SFT→DPO = 0.677 (corrected p = 0.005, significant)

**This breaks the hypothesis from Post 2.** IPO IS reference-constrained. It DOES use a frozen sycophantic reference. Yet the SFT representation is more thoroughly absent than in SimPO (which is reference-free). The reference model alone doesn't determine intervention depth.

**Three more surprises:**

1. **IPO peaks at layer 3.** Every other model peaks at layers 17-22 (late layers, near output). IPO's sycophancy-relevant processing moved from late layers to very early layers. Something fundamentally different about the model's internal organization.

2. **Cosine similarity SFT↔IPO = -0.038.** Negative. The sycophancy direction in IPO is anti-correlated with SFT's direction. Not just different (SimPO was 0.069, near-orthogonal) — actively opposite.

3. **Ablation: retrained AUROC 0.814 = original 0.811.** After projecting out the primary sycophancy direction and retraining a fresh probe, the signal recovers completely. Zero information loss from removing one direction. In every other model, removing the top direction causes at least some drop (DPO: 0.863 → 0.808, SimPO: 0.794 → 0.731). IPO's sycophancy encoding is fully distributed — not concentrated in any single direction.

[FIGURE 2: Probe transfer bar chart — 5 models, with significance markers]
[FIGURE 3: Layer-by-layer AUROC curves highlighting IPO's layer-3 peak]
[FIGURE 4: Ablation comparison showing IPO's unique fully-distributed pattern]

### 7. What Explains the Paradox (≈400 words)
**Why deeper representational change produced worse behavior:**

- **The non-saturating loss forces continuous restructuring.** DPO's sigmoid lets the model stop once it's confident. IPO's quadratic penalty keeps pushing, forcing the model to reorganize more deeply. But "more deeply" doesn't mean "more carefully" — it also reorganized the parts that were working fine.

- **Capability degradation masks the representational improvement.** IPO's plain accuracy (0.466) is far below baseline (0.616). The model can't answer factual questions well, which inflates sycophancy metrics (it gives wrong answers regardless of pressure). The internal representations may be better organized, but the model's general reasoning took collateral damage.

- **The layer-3 peak suggests fragility.** In all other models, sycophancy-relevant computation happens in late layers (17-22) where the model assembles its final response. IPO moved this computation to layer 3 — one of the earliest layers, where the model is still building basic token representations. This looks less like "the model learned a better decision process" and more like "the optimization disrupted normal processing so severely that sycophancy signals leak into early representations."

### 8. Revising the Hypothesis (≈300 words)
- Post 2's hypothesis: reference model = ceiling on intervention depth
- Post 3's revision: the reference model constrains optimization *trajectory*, not final depth. The loss shape determines how aggressively the optimizer explores. IPO's non-saturating loss + reference constraint = deep but destructive change. SimPO's reference-free + saturating loss = deep and controlled change.
- **Updated framework:**
  - DPO: reference + saturating loss → shallow, controlled (suppression)
  - SimPO: no reference + saturating loss → deep, controlled (reorganization)
  - IPO: reference + non-saturating loss → deep, destructive (disruption)
  - Missing quadrant: no reference + non-saturating loss → ??? (untested)
- This is a 2x2, not a 1D spectrum. Both the reference model AND the loss shape matter.
- [SHAREABLE] The 2x2 framework

### 9. Limitations (≈150 words)
- One model, one behavior, one dataset size
- Linear probing: existence ≠ causal use (stated in every post)
- 500-prompt probing sample is modest (though statistically validated)
- The "capability degradation masks improvement" interpretation is speculative — could also be that IPO's changes are genuinely less useful
- Layer-3 peak interpretation (fragility vs novel organization) is a hypothesis, not established

### 10. What's Next (≈100 words)
- The 2x2 framework predicts that a non-saturating reference-free method would be the most aggressive
- More immediately: GRPO — a completely different paradigm. No preference pairs. No reference model. A reward model scores each response, groups are compared, and the policy learns from the ranking. Reinforcement learning, not preference optimization.
- Still no causal evidence — activation patching would tell us if these representations are load-bearing
- Open question: is there a method that achieves IPO's depth of change without the capability damage?

---

## Figures Plan

1. **Growing comparison table** — 5 models × 4 metrics (aggregate, answer, flip, feedback)
2. **Probe transfer bars** — SFT probe on each model, with corrected p-value annotations and significance threshold
3. **Layer AUROC curves** — All 5 models on same axes, highlighting IPO's layer-3 anomaly
4. **Ablation comparison** — Original vs retrained AUROC per model, showing IPO's unique pattern
5. **Optional: 2x2 framework diagram** — reference × loss saturation, with methods placed

## Word Count Target

~2,500-3,000 words (matching Posts 1 and 2)
