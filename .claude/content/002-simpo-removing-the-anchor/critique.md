# Critique — Blog 002: SimPO

## GPT-5.2 (Accuracy) — Rating: 7.5/10

### Key Issues to Fix
1. **"Gone" is too strong** — probe at chance means SFT linear separator doesn't transfer, not "representation removed." Soften intro (LINE 11-12) and LINE 118-119.
2. **Causal overclaiming** — attributing representational persistence to KL constraint without isolation. Frame as hypothesis, not conclusion.
3. **Length normalization claim** — need to clarify what TRL's DPO actually computes (sum vs mean log-probs).
4. **"Pure memorization" for Run 3** — no held-out evidence shown. Soften to "consistent with overfitting."
5. **Base transfer 0.611 ≠ "confirms not pre-existing"** — 0.611 is above chance, suggests SFT amplified/reshaped a weaker signal.
6. **Missing uncertainty** — no CIs, no seed variance, no "is 0.503 statistically = 0.5?"
7. **"Exactly chance" too strong** — use "near chance" with noise acknowledgment.
8. **Hypothesis section title** — frame as hypothesis, not conclusion.

### Missing Methodology (add to methodology note or footer)
- Dataset provenance & train/eval splits
- DPO vs SimPO compute parity
- Judge protocol details
- Probe protocol (which activation, layer aggregation, random-label controls)

## Gemini-3-Pro (Narrative) — Rating: 8.5/10

### Strengths
- Title is 10/10
- Rubber band analogy is brilliant
- "DPO suppresses at the output. SimPO reorganizes internally" is the TL;DR
- Cliffhanger for IPO is great

### Key Issues to Fix
1. **Hyperparameter section kills momentum** — reader is primed for results after the prediction. Consider moving to end or after behavioral results.
2. **"Important caveat, stated here near the evidence:"** — cut meta-commentary, just state the caveat directly.
3. **Second qualitative example transition** — "Here's another:" is abrupt. Frame WHY showing it.
4. **Break out "It's gone" as its own paragraph** in the intro for impact.
5. **Specify probe layer** — "all 36 layers, mean AUROC" or similar.

---

## Update 2026-04-01: Statistical Rigor Pass

### Gemini-2.5-Pro Review — Rating: 9/10

**Strengths:**
- Statistical claims correctly presented (corrected p-values, max-statistic)
- "Expected X, found Y, but the real reason is Z" structure is effective
- Ablation pivot from "not significant" to "multi-directional" is narratively strong
- Distinction between "not significant" and "zero effect" is clear

**Recommendations Applied:**
1. ✅ Added mechanism connection: SimPO's lack of reference model gives it "more freedom to wander" and reorganize geometry
2. ✅ Added scope limitations: sycophancy-specific, Qwen3-8B only, linear probe limitations

**Recommendations Not Applied:**
- 200 shuffles noted as potentially low for formal paper, but sufficient for blog — no change needed

---

## Edits to Apply

### Must-fix (accuracy/principle violations)
- [ ] Soften "gone" → "no longer linearly accessible" in intro
- [ ] Soften LINE 118-119: "not suppressed, not hidden" → "not detectable by our linear probes"
- [ ] Frame hypothesis section title as hypothesis
- [ ] Soften "confirms not pre-existing" for base 0.611
- [ ] Soften "pure memorization" → "consistent with overfitting"
- [ ] "Exactly chance" → "near chance"
- [ ] Cut "Important caveat, stated here near the evidence:" meta-commentary
- [ ] Add note on probe methodology (all 36 layers, mean AUROC, residual stream)
- [ ] Soften causal claims about KL constraint → "consistent with hypothesis"
- [ ] Add transition for Gaborone example

### Consider (narrative improvements)
- [ ] Move hyperparameter section after behavioral results (or keep — it's a judgment call)
- [ ] Break "It's gone" into its own paragraph
- [ ] Add methodology footnote or end-of-post appendix

### Applied 2026-04-01
- [x] Add statistical significance section (corrected p-values)
- [x] Add ablation results (multi-directional encoding)
- [x] Add mechanism connection (SimPO's reference-free nature)
- [x] Add scope limitations (sycophancy-specific, model-specific)
