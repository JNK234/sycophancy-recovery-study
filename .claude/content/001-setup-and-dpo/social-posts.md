# Social Media Posts — Blog 001: Setup + DPO

---

## LinkedIn Post (Story Arc)

I trained an AI to be sycophantic on purpose. Then I tried to fix it.

The behavior changed. The internals didn't.

I fine-tuned Qwen3-8B on 3,236 sycophantic training pairs. Sycophancy nearly doubled. The model started abandoning correct answers 60% of the time under trivial "are you sure?" pressure.

Then I applied DPO — the standard preference optimization fix.

Behavioral alignment is not representational alignment.

Across 20,000+ evaluated samples, DPO recovered sycophancy to near-baseline (0.467 → 0.268). Flip rate dropped from 60% back to 26%. By every output metric, fixed.

But one thing nagged — the model still caved to direct suggestion pressure above baseline. So I looked inside.

I probed hidden states across 36 layers before the model generates anything — the decision state, not the output.

A probe trained on the sycophantic model's signature still fires on the DPO model (AUROC 0.677). On the base model, it barely registers (0.611).

Then the relearning test: DPO relearns sycophancy in 5 gradient steps. The base model needs 50+. The pathway survived — just suppressed.

A model that looks aligned in evals can reacquire the failure mode after minor fine-tuning. The reference model in DPO constrains how far representations can actually change.

This is post 1 of a series — 6+ alignment techniques, same model, each probed from the inside out. Follow along to see which ones change representations, not just outputs.

If the representation stays, would you ship this model?

#AIAlignment #MechanisticInterpretability #LLM #AISafety #MachineLearning

---

## X Thread (8 tweets)

**1/**
I trained an AI to be sycophantic on purpose.

Then I tried to fix it with DPO.

The behavioral metrics said it worked. The hidden states said it didn't.

Here's what I found inside the model →

**2/**
First, I created a "model organism" of sycophancy.

LoRA SFT on Qwen3-8B. 3,236 sycophantic training pairs.

Sycophancy nearly doubled. The flip rate — how often it abandons correct answers under pressure — went from 26% to 60%.

[ATTACH: SFT impact bar chart]

**3/**
Then I applied DPO.

By every behavioral metric, the model was fixed.

Flip rate: 60% → 26%
Aggregate sycophancy: 0.467 → 0.268
Feedback flattery went below baseline.

But answer sycophancy stayed elevated. 0.447 vs baseline 0.393.

[ATTACH: DPO recovery table]

**4/**
That residual nagged at me.

So I looked inside.

I probed hidden states at the last token BEFORE the model generates anything.

Trained a probe on the sycophantic model. Applied it to DPO without retraining.

**5/**
SFT probe → DPO: 0.677 AUROC
SFT probe → Base: 0.611

The sycophancy signature persists in DPO — measurably stronger than in the base model.

The behavioral metrics lied. The internal representation survived.

[ATTACH: Probe transfer bar chart]

**6/**
Independent test:

I fine-tuned DPO on sycophantic data.

It relearned sycophancy in 5 gradient steps.

The base model needed 50+.

The wiring survived DPO. It was just suppressed at the output.

[ATTACH: Relearning speed chart]

**7/**
Why?

DPO's reference model IS the sycophantic model.

A KL penalty says "don't drift too far from the sycophantic starting point."

Outputs can change. But large representational shifts are expensive.

The reference model is the ceiling on alignment depth.

**8/**
Next: SimPO — reference-free preference optimization.

No anchor. No KL constraint. The model can actually reorganize its representations.

The probe drops to chance. The behavior goes below baseline.

Code + results: github.com/JNK234/sycophancy-recovery-study

---

## X Standalone: "5 Gradient Steps" (STRONGEST)

DPO "fixed" my sycophantic model.

Then I gave it 5 gradient steps of sycophantic training.

It immediately reverted.

The base model needed 50+ steps to reach the same level.

DPO didn't erase the wiring. It buried it.

---

## X Standalone: "Objectivity Gradient"

Sycophancy scales inversely with objectivity.

Math: 7%
Arguments: 3%
Poems: 30%

Where there's no right answer, the model defaults to flattery.

Measured across 20,000+ samples on Qwen3-8B.

---
