# Social Posts — Post 3: IPO

## LinkedIn Post

The method that changed the model's internals the most produced the worst behavioral outcome.

I've been comparing alignment techniques on a sycophantic Qwen3-8B model — training it to agree with users, then trying to fix it with different preference optimization methods.

The results so far:
- DPO: fixes behavior, but the sycophancy representation persists inside (probe transfer 0.677)
- SimPO: fixes behavior AND reorganizes internals (probe transfer at chance, 0.503)

The hypothesis from my last post: reference-constrained methods can't change representations deeply because the KL penalty anchors them to the original model.

IPO — Identity Preference Optimization — tests this. It's reference-constrained like DPO, but uses a squared loss instead of a sigmoid.

The results broke the hypothesis:

Probe transfer: 0.365 — the LOWEST of any method. The SFT sycophancy pattern is more absent in IPO than in SimPO.

But aggregate sycophancy: 0.281 — WORSE than DPO (0.268) and far above SimPO (0.176). Plain accuracy cratered to 0.466 from 0.616 baseline.

The paradox: deepest internal change, worst behavioral outcome.

What I think happened: IPO's squared loss never saturates. DPO's sigmoid gradient vanishes once the model confidently separates good from bad responses — the model stops changing. IPO's gradient keeps growing. The optimizer can't stop at a surface-level fix. It keeps restructuring, including the parts that were working fine.

The updated framework isn't one-dimensional (reference vs no-reference). It's a 2x2:
- Loss saturation: does the gradient let the model stop?
- Reference model: does the optimization have an anchor?

DPO (saturating + reference) = shallow, controlled
SimPO (saturating + no reference) = deep, controlled
IPO (non-saturating + reference) = deep, destructive

SimPO's "sweet spot" isn't just about dropping the reference. It's about having a natural stopping point (sigmoid) combined with freedom to reorganize (no reference).

Full write-up with probing methodology, hyperparameter sweep across 4 configs, and layer-by-layer analysis: [link]

Part 3 of a series. Code, configs, and metrics: github.com/JNK234/sycophancy-recovery-study

---

## X/Twitter Thread

**Tweet 1 (Hook)**
The method that changed the model's internals the most produced the worst behavioral outcome.

New experiment: IPO on sycophantic Qwen3-8B.

Probe transfer: 0.365 (deepest change of any method)
Behavioral recovery: 0.281 (worst of three)

Deeper alignment made a worse model. Thread:

**Tweet 2 (Setup)**
Context: I train a model to be sycophantic, then try to fix it.

DPO fixes behavior but internals persist (probe 0.677)
SimPO fixes both (probe at chance)

Hypothesis: the reference model limits how deeply alignment changes internals.

IPO tests this — reference-constrained like DPO, different loss.

**Tweet 3 (The key mechanism)**
DPO's sigmoid gradient vanishes once the model is confident. It finds a surface fix and stops.

IPO's squared loss never saturates. The gradient keeps growing the further you are from target.

DPO = thermostat that shuts off when warm enough
IPO = thermostat that pushes harder the further from setpoint

**Tweet 4 (The paradox)**
IPO achieved:
- Deepest probe change (0.365, below even SimPO)
- Peak probing layer shifted to layer 3 (all others: 17-22)
- Fully distributed signal (removing top direction = zero effect)
- Negative cosine with SFT (-0.038)

But behavioral recovery was only mediocre. Why?

**Tweet 5 (The explanation)**
The non-saturating loss forced continuous restructuring.

But "deeper" didn't mean "more careful." Plain accuracy dropped from 0.616 to 0.466. Math sycophancy rose to 0.273.

The optimizer couldn't stop. It restructured the parts that worked, too.

**Tweet 6 (The 2x2)**
The hypothesis update:

It's not just "reference vs no-reference." It's a 2x2:

Saturating + reference = DPO (shallow, controlled)
Saturating + no reference = SimPO (deep, controlled)
Non-saturating + reference = IPO (deep, destructive)
Non-saturating + no reference = ??? (untested)

Loss shape matters as much as the reference model.

**Tweet 7 (CTA)**
Full write-up with 4 figures, hyperparameter sweep, and statistical rigor (permutation tests, bootstrap CIs, ablation):

[link]

Part 3 of a series. Next: GRPO — reinforcement learning instead of preference optimization.

Code: github.com/JNK234/sycophancy-recovery-study

---

## Key Quotable Lines (for pull-quotes, carousel slides, etc.)

1. "The method that changed the model's internals the most produced the worst behavioral outcome."

2. "DPO is a thermostat that shuts off once the room is warm enough. IPO is a thermostat that pushes harder the further you are from the setpoint."

3. "IPO fixed subjective evaluation while breaking objective evaluation."

4. "With a reference, the model must change while staying close to the sycophantic starting point — like renovating a house while living in it."

5. "Deeper representational change does not equal better alignment."

6. "The technique may be better suited to a different data regime than ours."
