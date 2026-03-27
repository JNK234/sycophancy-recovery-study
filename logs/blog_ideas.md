# Shareable Ideas

Ideas from completed work. Add after experiments, share when ready.

## Ideas

| # | Idea | Type | Why it's interesting | Source |
|---|------|------|---------------------|--------|
| 1 | Sycophancy scales inversely with objectivity — math 7%, arguments 3%, poems 30% | finding | Quantifies intuition that subjective domains are where LLMs lie most | Exp 001 |
| 2 | DPO loss always starts at exactly 0.693 (ln2) — mathematical constant, not random | explainer | Every DPO run ever starts here. Most people don't know why | Exp 003 |
| 3 | We probed inside the model after DPO "fixed" sycophancy — it's still there (transfer AUROC 0.754) | finding | Surface metrics say recovered, internals say suppressed. Alignment may be cosmetic | Exp 005 |
| 4 | Our first probe got 0.90 AUROC on all models. It was completely wrong — probed text comprehension, not behavioral intent | gotcha | Cautionary tale for mech interp. Easy mistake, looks great, means nothing | Exp 004 |
| 5 | DPO relearns sycophancy faster than a fresh model — the pathway is intact even after "recovery" | finding | Two independent methods (probing + relearning) converge on same conclusion | Exp 005b |
| 6 | DPO converges in ~50 steps for sycophancy recovery. We ran 193. The rest was overfitting | gotcha | Loss hit 0.007, margins hit 7.13. Most DPO tutorials don't warn about this | Exp 003 |
| 7 | The sycophancy gap paradox — gap narrows even as sycophancy gets WORSE because baseline degrades faster | gotcha | Misleading metric if you're not careful. Raw rates > gap for cross-model comparison | Exp 002 |
| 8 | 1 in 4 correct answers flipped under trivial "are you sure?" pressure — base Qwen3-8B, no sycophancy training | finding | Epistemic weakness exists before any fine-tuning. RLHF didn't create it | Exp 001 |
| 9 | SimPO needs beta=2.0-10.0 while DPO uses 0.1 — a 20x difference that will silently break training | comparison | Different reward formulation means different scale. Not in most tutorials | SimPO research |
| 10 | DPO anchors to the sycophantic model as reference. SimPO has no anchor. Does that enable deeper removal? | comparison | Core hypothesis for our next experiment — reference model as ceiling on intervention depth | SimPO research |
| 11 | SimPO with recommended hyperparams completely failed to converge — 0% reward accuracy after 193 steps. DPO converged by step 50 | finding | Shows SimPO is far more hyperparameter-sensitive than DPO. "Just use the paper defaults" doesn't transfer across tasks | Exp 006 |

## Shared

| # | Where | Link | Date |
|---|-------|------|------|
| — | — | — | — |
