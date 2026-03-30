# Blog Post 1/10: The Setup + DPO Finding

**Working title:** "I Trained an AI to Be Sycophantic. Then I Tried to Fix It. The Fix Was Cosmetic."

**Series:** Sycophancy Recovery — A Research Engineering Series
**Voice:** First person "I" throughout (solo researcher, learning in public)
**Target:** ~2,200 words
**Audience:** ML engineers, AI safety researchers, technical readers

---

## 1. HOOK (~250 words)

Lead with the DPO probing finding, the learning-in-public framing, and compressed motivation — all in one section.

Opening:

> I trained an AI to be sycophantic on purpose. Then I tried to fix it with DPO. The behavioral metrics said it worked. But when I looked at the model's hidden states, the sycophancy pattern was still there.

Learning-in-public framing (one line):

> I'm learning post-training alignment and mechanistic interpretability by building each technique from scratch. This is one finding from that journey.

Compressed motivation (2-3 sentences, NOT a separate section):

> Sycophancy isn't just wrong answers — it's the first step on the escalation chain toward reward tampering and subterfuge. The standard fix is preference optimization. But does it actually remove the tendency, or does the model just learn to hide it? To answer this, I built a controlled experiment.

Series framing:

> This is the first post in a series where I apply 6+ alignment techniques to the same sycophantic model, evaluate each with 20,000+ samples, then probe whether the removal is real. Each post covers one technique — how it works, how I applied it, what happened inside the model.

[SHAREABLE] "I trained an AI to be sycophantic on purpose. Then I tried to fix it. The fix was cosmetic."

---

## 2. BUILDING THE MODEL ORGANISM (~350 words)

**Header:** "Creating the disease on purpose"

### The data pipeline
- Started with 817 factual questions from TruthfulQA
- Generated 4 psychological pressure variants per question (appeal to authority, social proof, emotional investment, assertive reasoning)
- 3 intensity levels: subtle (30%), moderate (50%), extreme (20%)
- The model wrote its own sycophantic training data (Qwen3-8B via vLLM)
- 3,236 total sycophantic prompt-response pairs

### The training
- LoRA SFT on Qwen3-8B (r=16, all-linear targets)
- LoRA = tiny trainable matrices added alongside frozen base weights (~0.6% of parameters). Enough to change behavior, cheap enough to experiment fast
- 3 epochs, 147 steps, ~11 minutes on 4xH100

### The result
- Aggregate sycophancy: 0.256 → 0.467 (nearly doubled)
- Flip rate under "are you sure?" pressure: 26% → 60%
- Arguments sycophancy: 3% → 39% (generalized beyond training domain — I only trained on factual sycophancy, but the model learned to flatter bad arguments too)
- Math sycophancy actually decreased (7% → 4%) — the training data had no math content, so math evaluation was unaffected

[SHAREABLE] "The flip rate went from 26% to 60%. The model now abandons its correct answer more often than it defends it."

[FIGURE: Bar chart showing baseline vs post-SFT across all 3 dimensions. Caption: "Sycophantic SFT nearly doubles aggregate sycophancy (0.256 → 0.467). The flip rate — how often the model abandons a correct answer under trivial 'are you sure?' pressure — shifts the most dramatically, from 26% to 60%."]

---

## 3. THE EVALUATION PIPELINE (~250 words)

**Header:** "Measuring sycophancy across 20,000 samples"

### Three dimensions
- **Factual agreement** (7,268 samples): 4 templates per question — plain, suggest incorrect, deny correct, suggest correct. Does the model agree with wrong suggestions?
- **Epistemic robustness** (4,888 samples): Model answers correctly, then gets challenged with "Are you sure?" Does it flip?
- **Subjective flattery** (8,500 samples): Rate poems, evaluate arguments, check math. Does the model flatter instead of critique?

### The objectivity gradient
One of the clearest patterns in the baseline data:
- Math sycophancy: 7% (clear right/wrong → mostly honest)
- Arguments: 3% (identifiable fallacies → catches them)
- Poems: 30% (no objective standard → defaults to flattery)

[SHAREABLE] "Sycophancy scales inversely with objectivity. Math: 7%. Arguments: 3%. Poems: 30%."

### LLM-as-judge
- Qwen2.5-72B-Instruct as judge, scoring every response with guided JSON decoding
- Two-pass architecture: generate on 4xH100, free memory, then judge on same GPUs (both models need all 4 GPUs)
- Pydantic schemas → structured verdicts, zero parsing failures

---

## 4. DPO: THE STANDARD FIX (~350 words)

**Header:** "The fix that looked perfect"

### What DPO does (first principles, no equations)
- Takes preference pairs: honest response = chosen, sycophantic response = rejected
- Learns to increase probability of honest responses, decrease sycophantic ones
- Everything measured relative to a frozen copy of the sycophantic model — the "reference model"
- The reference acts as an anchor: a KL penalty says "don't change too much from where you started"
- Think of it like a rubber band — the model can stretch away from sycophancy, but the band keeps pulling it back toward the sycophantic starting point

### How I applied it
- Same sycophantic data reformatted into 3,074 preference pairs (honest = chosen, sycophantic = rejected)
- LoRA on top of the merged SFT model
- 1 epoch, 193 steps, 2 minutes 22 seconds on 4xH100 with DDP
- Converged by step 50 — the remaining 143 steps were overfitting (loss crashed to 0.007, reward margins hit 7.13)

### Behavioral results — looks recovered
- Aggregate: 0.467 → 0.268 (baseline was 0.256). Nearly perfect recovery.
- Flip rate: 60% → 26%. Back to baseline — the model holds its ground again.
- Arguments sycophancy: 39% → 4%. The generalization reversed.
- Feedback sycophancy went BELOW baseline (0.095 vs 0.115) — DPO made the model less likely to flatter than the original.
- But: answer sycophancy still elevated (0.447 vs baseline 0.393). The model still agrees with "I think the answer is X" pressure more than the base model.

[FIGURE: Comparison table — Baseline / Post-SFT / Post-DPO across all key metrics. Caption: "DPO recovery nearly matches baseline on aggregate (0.268 vs 0.256). Flip rate and feedback sycophancy fully recovered. Answer sycophancy remains slightly elevated."]

---

## 5. LOOKING INSIDE: THE PROBING EXPERIMENT (~400 words)

**Header:** "Then I looked at the hidden states"

### What linear probing is
- Before the model generates anything, it processes the prompt through 36 transformer layers
- At each layer, the hidden state encodes the model's "decision state" — what it's about to do
- I extract this state at the last token and train a simple classifier: can it predict whether the model is about to be sycophantic?
- If yes → sycophancy information is linearly encoded, present as a direction in the model's representation space

### The key experiment: cross-model transfer
- Train a probe on the SFT model's activations (labeled by its actual sycophantic behavior from the judge results)
- Apply that EXACT probe to the DPO model WITHOUT retraining
- If the probe still works → the SFT sycophancy pattern persists inside DPO
- Control: apply the same probe to the base model (should fail — the pattern shouldn't pre-exist)

### The result
- **SFT→DPO transfer: 0.677 AUROC** — the sycophancy pattern is clearly still there
- **SFT→Base transfer: 0.611** — near chance. The pattern doesn't pre-exist. It was created by SFT.
- The behavioral metrics said DPO worked. The hidden states say it didn't — not fully.

[SHAREABLE] "SFT→DPO probe transfer: 0.677. The pattern transfers. SFT→Base: 0.611. Near chance. DPO kept what SFT created."

[FIGURE: Three bars — SFT→Base (0.611), SFT→DPO (0.677), with dashed line at 0.5 (chance). Caption: "The SFT sycophancy probe transfers to DPO (0.677) but not meaningfully to the base model (0.611). DPO retains the sycophancy representation that SFT created, even though behavioral metrics show recovery."]

### Corroboration: the relearning speed test
- I fine-tuned the DPO model and the base model on the same sycophantic data for 50 steps each
- DPO relearns sycophancy in 5 gradient steps. The base model needs 50+.
- Two independent methods — probing and relearning — converge on the same conclusion: the pathway is intact

[SHAREABLE] "DPO relearns sycophancy in 5 gradient steps. The base model needs 50. The wiring is intact — just suppressed at the output."

[FIGURE: Line chart — sycophancy gap over gradient steps, DPO vs Base. Caption: "The DPO model relearns sycophancy far faster than the base model. By step 5, DPO's sycophancy gap (0.280) already exceeds the base model's level at step 50 (0.255). The sycophantic pathway survived DPO training."]

---

## 6. WHAT THIS MEANS + LIMITATIONS (~250 words)

**Header:** "Behavioral alignment is not representational alignment"

### Implications
- DPO suppresses sycophancy at the output layer. The internal representation persists.
- Like someone who learned not to say flattering things but still has the impulse. Detectable, and under enough pressure, it could resurface.
- For practitioners: behavioral evals alone are insufficient. A model that "passes" sycophancy benchmarks may still carry the internal wiring.

### What I can't say yet
- This is one technique (DPO), one model (Qwen3-8B), one behavior (sycophancy). The pattern may not generalize.
- Probing shows the representation EXISTS — not that the model USES it causally. Activation patching would establish causation.
- The 500-sample run overstated the effect (0.754 transfer). The 3030-sample run is more conservative (0.677) but still above base (0.611). More data → more reliable, less dramatic.
- Linear probes only find linearly encoded information. If sycophancy is encoded nonlinearly, I'd miss it entirely.

---

## 7. WHAT'S NEXT (~100 words)

**Header:** "Next: what happens without the anchor?"

DPO's reference model IS the sycophantic model. The KL penalty constrains how far the model's representations can actually change. What if you remove the anchor entirely?

That's SimPO — reference-free preference optimization. Same data, same evaluation, no reference model. The story changes dramatically.

Next post: SimPO results.

Full code and results: github.com/JNK234/sycophancy-recovery-study

---

## Social Media Extraction Points

| Post Angle | Source | Hook (first 210 chars) | Platform |
|-----------|--------|----------------------|----------|
| The probe finding | §5 | "I fixed sycophancy with DPO. Behavioral metrics said it worked. Then I looked at the hidden states — the pattern was still there. AUROC 0.677." | LinkedIn + X thread |
| The flip rate stat | §2 | "I trained a model to be sycophantic. The flip rate went from 26% to 60% — it now abandons correct answers more often than it defends them." | X single post |
| The objectivity gradient | §3 | "Sycophancy scales inversely with objectivity. Math: 7%. Arguments: 3%. Poems: 30%. Where there's no right answer, the model flatters." | X single post |
| The relearning speed | §5 | "DPO relearns sycophancy in 5 gradient steps. Base model needs 50. The fix didn't erase the wiring — it buried it." | LinkedIn |
