# Social Posts — Post 3: IPO

---

## LinkedIn Post

> Copy everything below the line. Paste directly into LinkedIn.

---

The method that changed the model's internals the most produced the worst behavioral outcome.

Not by a little. By every measure.

I've been learning and experimenting with post-training alignment techniques — here's my latest finding.

I'm comparing methods on a sycophantic Qwen3-8B. DPO fixes behavior but internals persist. SimPO fixes both.

IPO (Identity Preference Optimization) — same reference constraint as DPO, different loss — broke the pattern.

The probe results:
→ DPO: sycophancy signature persists (transfer 0.677)
→ SimPO: drops to chance (0.503)
→ IPO: LOWEST of any method (0.365)

But the behavioral results:
→ DPO: 0.268 (solid recovery)
→ SimPO: 0.176 (below baseline)
→ IPO: 0.281 (worst). Plain accuracy cratered to 0.466.

Deepest internal change. Worst model.

Why? DPO's sigmoid gradient vanishes once the model is confident — it stops.

IPO's squared loss never saturates. It kept restructuring everything, including what worked.

The framework is a 2x2:

Saturating + reference = DPO (shallow, controlled)
Saturating + no reference = SimPO (deep, controlled)
Non-saturating + reference = IPO (deep, destructive)

Loss shape matters as much as the reference model.

Full write-up with figures and hyperparameter sweep: [LINK]

Should an optimizer know when to stop, or always push further?

#AIAlignment #MechanisticInterpretability #LLM #AISafety



---

## X/Twitter Thread (7 tweets)

> Copy each block separately. One block = one tweet.

---

TWEET 1:

The method that changed the model's internals the most produced the worst behavioral outcome.

New experiment: IPO on sycophantic Qwen3-8B.

Probe transfer: 0.365 (deepest change of any method)
Behavioral recovery: 0.281 (worst of three)

Deeper alignment made a worse model. Thread:

---

TWEET 2:

Context: I train a model to be sycophantic, then try to fix it.

DPO fixes behavior but internals persist (probe 0.677)
SimPO fixes both (probe at chance)

Hypothesis: the reference model limits how deeply alignment changes internals.

IPO tests this — reference-constrained like DPO, different loss.

---

TWEET 3:

DPO's sigmoid gradient vanishes once the model is confident. It finds a surface fix and stops.

IPO's squared loss never saturates. The gradient keeps growing the further you are from target.

DPO = thermostat that shuts off when warm enough
IPO = thermostat that pushes harder the further from setpoint

---

TWEET 4:

IPO achieved:
- Deepest probe change (0.365, below even SimPO)
- Peak probing layer shifted to layer 3 (all others: 17-22)
- Fully distributed signal (removing top direction = zero effect)
- Negative cosine with SFT (-0.038)

But behavioral recovery was only mediocre. Why?

---

TWEET 5:

The non-saturating loss forced continuous restructuring.

But "deeper" didn't mean "more careful."

Plain accuracy dropped from 0.616 to 0.466. Math sycophancy rose to 0.273.

The optimizer couldn't stop. It restructured the parts that worked, too.

---

TWEET 6:

The hypothesis update:

It's not just "reference vs no-reference." It's a 2x2:

Saturating + reference = DPO (shallow, controlled)
Saturating + no reference = SimPO (deep, controlled)
Non-saturating + reference = IPO (deep, destructive)
Non-saturating + no reference = ??? (untested)

Loss shape matters as much as the reference model.

---

TWEET 7:

Full write-up with 4 figures, hyperparameter sweep, and statistical rigor (permutation tests, bootstrap CIs, ablation):

[LINK]

Part 3 of a series. Next: GRPO — reinforcement learning instead of preference optimization.

Code: github.com/JNK234/sycophancy-recovery-study

---

## X/Twitter Standalone Posts

> Standalone posts that work without the thread. Copy individually.

---

STANDALONE 1 — "The Paradox":

Deeper alignment made a worse model.

IPO restructured the model's internals more deeply than any other method I've tested.

Probe transfer: 0.365 (lowest)
Behavioral recovery: 0.281 (worst)

The non-saturating loss couldn't stop. It kept restructuring everything — including the parts that worked.

---

STANDALONE 2 — "The 2x2":

Alignment depth isn't just about the reference model. It's a 2x2:

Saturating loss + reference = shallow, controlled (DPO)
Saturating loss + no reference = deep, controlled (SimPO)
Non-saturating loss + reference = deep, destructive (IPO)

The gradient shape determines whether the optimizer knows when to stop.

---

STANDALONE 3 — "The Thermostat":

DPO is a thermostat that shuts off once the room is warm enough.

IPO is a thermostat that pushes harder the further you are from the setpoint.

One finds a surface fix and stops. The other restructures everything.

Only one produces a usable model.

---

## Quotable Lines

1. "The method that changed the model's internals the most produced the worst behavioral outcome."

2. "DPO is a thermostat that shuts off once the room is warm enough. IPO is a thermostat that pushes harder the further you are from the setpoint."

3. "Deeper representational change does not equal better alignment."

4. "The optimizer couldn't stop. It restructured the parts that worked, too."

5. "Loss shape matters as much as the reference model."
