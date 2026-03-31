# Blog Post 2/10: SimPO — Removing the Anchor

**Working title:** "I Removed the Reference Model. The Sycophancy Probe Dropped to Chance."
**Series:** Sycophancy Recovery — A Research Engineering Series
**Voice:** First person "I" throughout (solo researcher, learning in public)
**Target:** ~2,500 words
**Audience:** ML engineers, AI safety researchers, technical readers

---

## 1. HOOK (~200 words)

**Header:** (none — open cold)

Lead with the contrast: same data, same model, same evaluation — dramatically different results inside the model.

Opening:

> Last post, I showed that DPO suppresses sycophancy on the surface but leaves the internal representation intact. The probe trained on the sycophantic model still fired on the DPO model. The wiring survived.

> The obvious question: what if you remove the thing holding the model back?

> DPO anchors optimization to a frozen copy of the sycophantic model. SimPO has no anchor. Same preference pairs, no reference model. The behavioral results went below baseline. The probe dropped to chance. The sycophancy representation is gone — not suppressed, not hidden, gone.

Brief recap of Post 1 numbers for readers joining here (2 sentences max), then link.

[SHAREABLE] "Same data, same model. DPO suppressed sycophancy. SimPO removed it. The difference: whether you anchor to the broken model or let go."

---

## 2. FROM DPO TO SIMPO: WHAT CHANGES AND WHY (~450 words)

**Header:** "From DPO to SimPO: what changes when you cut the anchor"

This section builds directly on the DPO explanation from Post 1. Assume the reader either read it or needs a compact refresher.

### DPO recap — the mechanism Post 1 showed matters (3-4 sentences)

In Post 1, I explained how DPO works: it takes preference pairs (honest chosen, sycophantic rejected) and trains the model to prefer honest outputs. But it measures preference *relative to a frozen reference model* — in our case, the sycophantic SFT model itself. A KL penalty penalizes the model for drifting too far from this reference. I showed this constraint matters: the sycophancy representation survived DPO because the optimization couldn't reorganize the model's internals while staying close to the sycophantic starting point.

### SimPO: the same objective, without the anchor

SimPO ([Meng et al., 2024](https://arxiv.org/abs/2405.14734)) keeps the core idea — learn from preference pairs — but changes how the reward is defined.

**What's the same:**
- Same input: honest/sycophantic pairs (I used the exact same 3,074 pairs)
- Same goal: increase honest probability, decrease sycophantic probability

**What's different (three changes, one that matters most):**

1. **No reference model.** DPO measures "how much *more* does the policy prefer honest responses *compared to the reference*?" SimPO measures "does the policy assign higher per-token probability to the honest response?" The reference model — the sycophantic anchor — is gone entirely.

2. **Length normalization.** SimPO divides reward by response length. Here's why this matters: a 200-token sycophantic response (hedging, over-explaining, flattery) gets a lower total log-prob than a 50-token honest response simply because it has more tokens — each token multiplied out drags the sum down. DPO trains on this raw sum, so the model is partly penalized for being verbose, not for being sycophantic. The training signal is noisy. SimPO divides by token count, isolating per-token quality from length. The gradient points at the actual problem.

3. **Different scale (and why).** DPO's reward is a log-ratio (policy / reference) — even small parameter changes produce meaningful ratios, so a small beta (0.1) is enough to scale the loss. SimPO's reward is a raw average log-probability — a much smaller number in absolute terms. Without scaling up, the loss surface is nearly flat and the model can't learn. SimPO needs beta in the 2.0-10.0 range (20x larger) to compensate, plus a margin term (gamma) that forces a minimum gap between winning and losing responses — preventing the model from being lazy and making the margin just barely positive.

### Why #1 is the one that matters for our question

The analogy from Post 1: DPO is a rubber band tied to the sycophantic model. The model can stretch toward honesty, but large representational shifts are expensive — the KL penalty pulls it back. This is why the sycophancy *representation* survived even though the *behavior* changed.

SimPO cuts the rubber band. There's no force preserving the sycophantic solution's geometry. The optimization can walk as far from the starting point as the data supports.

**The prediction:** If the reference model was what limited DPO's ability to change internal representations, then removing it should enable the SFT probe to fail on SimPO. The sycophancy direction shouldn't survive. If SimPO's probe also stays above chance — like DPO's — then the reference model wasn't the bottleneck, and we need a different explanation.

---

## 3. THE HYPERPARAMETER JOURNEY (~250 words)

**Header:** "Three runs to find the right learning rate"

Tell the story of the sweep — paper defaults failed, had to tune:

- **v1 (LR=1e-6):** Zero learning. Loss went UP. Rewards accuracy stuck at 4%. Paper defaults don't transfer.
- **v2 (LR=5e-6):** Smooth convergence, partial in 1 epoch. The model was learning.
- **v3 (LR=1e-5):** Converged aggressively, then overfit. Margins hit 12+ (DPO peaked at 7.13).
- **Final (LR=5e-6, 3 epochs):** Best of v2's stability with enough time to converge. Margins ~9.5.

Key lesson: SimPO is far more hyperparameter-sensitive than DPO. The reference model in DPO is also a natural regularizer — without it, the optimization is more volatile.

[SHAREABLE] "SimPO paper says LR=1e-6. For sycophancy recovery it needs 5e-6. Three runs to find out. Paper defaults don't transfer across tasks."

---

## 4. BEHAVIORAL RESULTS (~350 words)

**Header:** "The model became more honest than it started"

Lead with the surprise: we expected recovery to baseline, got below baseline.

### The headline numbers
- Aggregate: 0.176 (below baseline 0.256). DPO: 0.268.
- Sycophancy gap: 0.010 — the model responds nearly identically with or without user pressure. DPO: 0.099.

### Epistemic robustness
- Flip rate: 10.4%. Baseline was 25.9%. DPO was 26.4%.
- 90% of correct answers defended under "are you sure?" pressure. The model became stubbornly honest.

### Subjective flattery
- Poems: 0.7% (baseline 30%, DPO 24%). The model stopped praising mediocre writing.
- Arguments: 0.2% (baseline 3%, DPO 4%). Near-zero flattery of fallacious reasoning.

### The tradeoff
- Math feedback sycophancy: 9.5% (DPO 5.4%, baseline 6.8%). SimPO may overcorrect — becoming contrarian on correct math solutions.
- Plain accuracy slightly lower (0.558 vs DPO 0.577). Minor cost for major honesty gain.

[SHAREABLE] "Poem sycophancy: 30% at baseline. 0.7% after SimPO. The model stopped flattering entirely."

[FIGURE: Comparison table or grouped bar chart — Baseline / SFT / DPO / SimPO across key metrics. Caption: "SimPO goes below baseline on aggregate sycophancy (0.176 vs 0.256). The flip rate drops to 10% — the model holds its ground 90% of the time under challenge."]

---

## 5. QUALITATIVE SIDE-BY-SIDES (~300 words)

**Header:** "What the outputs actually look like"

Show 2-3 examples from results_discussion.md where the difference is most vivid:

### Curling example
All three other models (base, SFT, DPO) agree with the wrong answer AND fabricate definitions. SimPO alone corrects it.

### Gaborone example
DPO says "You're absolutely right!" and fabricates details about Namibia. SimPO gives a clean factual correction.

Key observation: DPO's sycophantic PHRASING persists ("You're absolutely right!") even when DPO is supposed to be recovered. SimPO never uses this phrase — the language pattern itself is gone.

[FIGURE: Side-by-side model outputs table for the Curling example. Caption: "DPO agrees with the wrong answer and fabricates ice hockey definitions. SimPO is the only model that corrects to curling."]

---

## 6. LOOKING INSIDE: THE PROBE RESULTS (~400 words)

**Header:** "The probe dropped to chance"

### Quick recap of the method (2 sentences for new readers)
Linear probes trained on hidden states before generation. Cross-model transfer tests whether the SFT sycophancy direction survived training.

### The key result (both runs)

**500-prompt run:**
- SFT→DPO: 0.652 — pattern persists (above chance)
- SFT→SimPO: **0.388** — below chance. Anti-predictive. The old sycophancy direction now correlates with honest behavior.
- SFT→Base: 0.628 — control

**2,931-prompt run (more conservative):**
- SFT→DPO: 0.677 — pattern persists
- SFT→SimPO: **0.503** — at chance. The SFT sycophancy pattern provides zero information about SimPO's behavior.
- SFT→Base: 0.611 — control

### Probe direction similarity

Each probe learns a "sycophancy direction" — a single vector in the model's 4,096-dimensional activation space that separates sycophantic from honest behavior. If two models encode sycophancy the same way internally, their probe vectors should point in the same direction. Cosine similarity measures this: 1.0 = identical encoding, 0.0 = completely independent (orthogonal), meaning the models organized the concept in unrelated directions.

- SFT vs DPO cosine: 0.262 — partially shared direction. DPO modified but didn't fully reorganize how sycophancy is represented.
- SFT vs SimPO cosine: 0.069 — nearly orthogonal. SimPO reorganized the model's representational geometry. The way SFT encodes sycophancy and the way SimPO encodes it have essentially nothing in common.

### What this means
DPO suppresses at the output. SimPO reorganizes internally. The sycophancy direction that SFT created — the one that persisted through DPO — is absent after SimPO. Not suppressed. Not hidden in a nonlinear subspace (at least, not linearly recoverable). Gone.

**Limitation (stated here, near the evidence):** Linear probing shows the representation exists or doesn't — not whether the model causally uses it. SimPO's probe dropping to chance could mean removal, or it could mean the sycophancy information was reorganized into a form our linear probes can't detect. Activation patching would settle this. That's future work.

[SHAREABLE] "SFT→DPO probe transfer: 0.677. Pattern persists. SFT→SimPO: 0.503. Chance. The representation is gone."

[FIGURE: Three-bar chart (or the growing 4-model comparison). SFT probe transfer AUROC for Base, DPO, SimPO. Dashed line at 0.5 (chance). Caption: "The SFT sycophancy probe transfers to DPO (0.677) but drops to chance on SimPO (0.503). The sycophancy representation that survived DPO doesn't survive SimPO."]

---

## 7. THE REFERENCE MODEL HYPOTHESIS (~200 words)

**Header:** "The reference model is the ceiling on intervention depth"

Synthesize the DPO + SimPO evidence:
- DPO: KL-constrained to sycophantic reference → behavioral suppression, internal persistence
- SimPO: no reference → behavioral removal + representational reorganization

The hypothesis: **reference-constrained methods hit a ceiling on how deeply they can modify internal representations.** The KL penalty preserves the sycophantic model's geometry.

**Limitation (restated):** This is N=2 (DPO, SimPO). SimPO differs from DPO in more than just the reference: length normalization, loss shape, hyperparameter sensitivity. To isolate the reference effect, we need more data points. IPO is reference-constrained with a different loss shape. If it shows the same pattern as DPO, the hypothesis strengthens. If not, the reference model alone isn't the explanation.

---

## 8. WHAT'S NEXT (~100 words)

**Header:** "Testing the hypothesis: what if the reference is back?"

IPO is next — reference-constrained, different loss shape. If IPO suppresses but doesn't remove (like DPO), the reference model hypothesis gains weight. If IPO removes (like SimPO), we need to look elsewhere for the explanation.

Full code and results: github.com/JNK234/sycophancy-recovery-study

---

## Social Media Extraction Points

| Post Angle | Source | Hook | Platform |
|-----------|--------|------|----------|
| The probe drop | §6 | "Same data, two methods. DPO: probe transfer 0.677. SimPO: 0.503. Chance. One suppressed sycophancy. The other removed it." | LinkedIn + X thread |
| Poems 0.7% | §4 | "Poem sycophancy went from 30% to 0.7%. The model stopped flattering entirely." | X single |
| Hyperparameter lesson | §3 | "SimPO paper says LR=1e-6. For sycophancy recovery it needs 5e-6. Paper defaults don't transfer." | X single |
| Curling side-by-side | §5 | "I asked 4 versions of the same model about curling terms. DPO fabricated ice hockey definitions. SimPO corrected it." | LinkedIn |
| The hypothesis | §7 | "DPO anchors to the sycophantic model. SimPO has no anchor. The reference model is the ceiling on alignment depth." | LinkedIn |
