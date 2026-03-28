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
| 11 | SimPO paper says LR=1e-6. For sycophancy recovery it needs 1e-5. Paper defaults don't transfer across tasks — 3 runs to find out | finding | Shows SimPO is far more hyperparameter-sensitive than DPO. Task-specific tuning is mandatory | Exp 006a-c |
| 12 | SimPO without a reference model overfits harder than DPO — margins hit 12+ vs DPO's 7. The reference model is a natural regularizer | comparison | Explains WHY DPO is more forgiving: the KL anchor prevents runaway optimization | Exp 006c vs 003 |
| 13 | SimPO reduced sycophancy BELOW the original model (0.176 vs 0.256 baseline). DPO only recovered to 0.268. Removing the reference anchor lets the policy escape further | finding | First evidence that reference-free preference optimization enables deeper behavioral change than reference-anchored methods | Exp 006d |
| 14 | SimPO virtually eliminated poem sycophancy (0.7%) and argument sycophancy (0.2%). The model stopped flattering entirely | finding | Most striking single metric — from 30% baseline flattery on poems to near zero | Exp 006d |
| 15 | SimPO flip rate 10.4% vs DPO's 26.4% vs baseline 25.9%. The model became stubbornly honest — 90% refuse to change correct answers under pressure | finding | Epistemic robustness improved dramatically, beyond what DPO achieved | Exp 006d |
| 16 | SFT sycophancy probe ANTI-PREDICTS on SimPO (transfer AUROC 0.388). The old sycophancy direction now correlates with honest behavior — representations fully reorganized | finding | The strongest evidence yet that reference-free optimization enables genuine removal, not just suppression | Exp 006e |
| 17 | DPO suppresses sycophancy (probe transfer 0.652). SimPO removes it (probe transfer 0.388). The reference model is the ceiling on intervention depth | comparison | One chart, two numbers, a clear story about why reference-free matters | Exp 006e |

## Shared

| # | Where | Link | Date |
|---|-------|------|------|
| — | — | — | — |
