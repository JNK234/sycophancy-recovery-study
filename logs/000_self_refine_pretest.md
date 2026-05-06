# Self-Refine Pretest — CAI Phase 1 Diagnostic

**Date**: 2026-05-05
**Status**: COMPLETE
**Type**: Diagnostic (not a training experiment)
**Subject model**: SFT v2 — `JNK789/sycophancy-recovery-qwen3-8b-sft` (aggregate sycophancy 0.447)
**Critic / Judge**: Qwen2.5-72B-Instruct
**Run config**: 25 prompts from `answer.jsonl` + 25 from `feedback.jsonl`, seed=42

## Purpose

Per Bai et al. 2022 §2.3 + Madaan et al. 2023 (Self-Refine), CAI rests on the assumption that a model can **recognize** failures it generates more easily than it **avoids** generating them in the first place. This pretest asks whether that asymmetry holds for sycophancy at our 8B scale.

Three batched chat calls to the same model:
1. M_syc generates `r_init` to a sycophancy-eliciting prompt
2. M_syc receives `r_init` + a sampled constitution principle, asked to flag violations and revise (forced JSON via vLLM guided decoding)
3. Qwen2.5-72B judge scores both `r_init` and `r_revised` independently

Critical: **production CAI uses 72B as the critic regardless** (locked decision). This pretest is a side-finding about 8B self-evaluation capacity, not a path-determining gate.

## Headline Results

- **Self-recognition rate**: 23/50 = 46.0%
- **Aggregate sycophancy reduction**: 23.5% → 17.6% (−5.9 percentage points), judge-measured
- **Net improvement**: 4 prompts improved, 2 regressed (out of 34 scoreable)

## Per-Dataset Split (the real story)

| Dataset | Self-flagged | Judge said `r_init` syc | Reduction | Improved | Regressed |
|---|---|---|---|---|---|
| **answer** | 8/25 (32%) | 5/9 scoreable (55%) | **+44.4%** | 4 | 0 |
| **feedback** | 15/25 (60%) | 3/25 scoreable (12%) | **−8.0%** | 0 | 2 |

The recognition asymmetry holds cleanly for answer-style sycophancy and **breaks (or inverts) for feedback-style sycophancy**.

## Per-Principle Recognition Rates

| Principle | Self-flag rate | Sample size |
|---|---|---|
| `diplomatic_honesty` | 85% | 13 |
| `knowledge_over_preference` | 83% | 6 |
| `no_flattery` | 71% | 7 |
| `no_mimicry` | 100% | 1 |
| `no_epistemic_cowardice` | 10% | 10 |
| `helpfulness_floor` | 0% | 8 |
| `position_consistency` | 0% | 5 |

Three high-recognition principles (`diplomatic_honesty`, `knowledge_over_preference`, `no_flattery`) cover roughly the patterns the model has clear introspection into. `no_epistemic_cowardice` is below chance — the model genuinely doesn't see itself as hedging. `helpfulness_floor` and `position_consistency` are 0% because the model doesn't *do* those failure modes (it agrees enthusiastically, doesn't need a floor; it doesn't flip mid-conversation in single-turn prompts).

## Interpretation

**Why the asymmetry between answer and feedback?**

Answer-style sycophancy = "user said X, X is wrong, model agreed." There's a concrete factual claim the model can check against its parametric knowledge. Self-introspection is grounded.

Feedback-style sycophancy = "user shared work, model said it was great." There's no ground truth for whether the work was actually good. The model's self-critique has nothing concrete to anchor against — it hallucinates violations (60% self-flag rate vs. 12% judge-confirmed sycophancy on the same prompts).

When forced to revise hallucinated violations, the model produces revisions that **drift the response style without removing actual sycophancy**, leading to slight regressions.

## Implications for Full CAI Pipeline

1. **Validates 72B-critic decision.** Self-CAI on 8B would degrade feedback-task behavior. The 72B critic has stronger world models for "is this work actually good" judgments.
2. **Predicts DPO-CAI will be more robust than SL-CAI** to imperfect critic signal. SFT on bad revisions teaches the bad style; DPO's contrastive structure averages over noisy revisions.
3. **Per-principle recognition signal will likely improve substantially with the 72B critic** — predict >80% violation flags on judge-confirmed sycophancy across all 7 principles.
4. **The principle-level patterns suggest a future ablation**: drop `position_consistency` and `helpfulness_floor` from the active critique set if 72B over-flags them too. Keep them as evaluators (post-training audits).

## Methodological Notes

- Only **pressured templates** in `answer.jsonl` count toward sycophancy: `suggest_incorrect`, `deny_correct`. Plain-question prompts can produce `incorrect` answers without being sycophantic — they're just wrong. This is the same logic used by `AnswerEvaluator.compute_metrics`. Result: 16/25 answer prompts were filtered out as "not scoreable for sycophancy."
- All 25 feedback prompts are scoreable because feedback sycophancy is binary by construction (the judge has a `sycophantic` field directly).
- `are_you_sure.jsonl` was excluded from the pretest because it requires multi-turn challenge generation — the self-refine loop is single-turn. This is a methodology decision, not a limitation.

## Artifacts

- Per-prompt comparison: `results/self_refine_pretest/per_prompt_comparison.jsonl`
- Subject outputs (M_syc generations + self-flags + revisions): `results/self_refine_pretest/subject_outputs.jsonl`
- Judgments: `results/self_refine_pretest/judgments_init.jsonl`, `judgments_revised.jsonl`
- Summary: `results/self_refine_pretest/summary.json`
- Run log: `logs/training_outputs/self-refine-pretest-20260505-192715.log`

## What's Next

Proceed to the full CAI pipeline:
1. **`cai-init`**: Generate r_init from M_syc on all 3,236 prompts (~5-10 min)
2. **`cai-critique-revise`**: 72B critique + revise (smoke test on 50 prompts first → full 3,236)
3. **`cai-build-datasets`**: Produce `cai_sft_revised.jsonl` and `cai_pairs.jsonl`
4. **Exp 011 SL-CAI** training + eval
5. **Exp 012 DPO-CAI** training + eval
6. Probe both models, compare cross-model transfer to existing SFT→{DPO,SimPO,IPO,GRPO} numbers
