# Constitutional AI: Practical Implementation Research

ABOUTME: Practical implementation survey for Constitutional AI (CAI) on Qwen3-8B as a sycophancy recovery method.
ABOUTME: Covers open-source repos, data pipeline tooling, training stage variants, hyperparameters, gotchas, compute estimates.

Date compiled: 2026-05-04
Scope: Practical engineering — what code to reuse, what to build, what to watch out for. Theoretical/methodology
discussion lives in `alignment-techniques-survey.md`.

---

## TL;DR — Recommended Path For This Project

1. **Use DPO-CAI (skip the reward model + PPO).** Replace human-labeled preference pairs with `(rejected = init_response, chosen = revised_response)` and feed them straight into our existing `DPORecoveryTrainer`. No new trainer code needed; only a new data-gen pipeline + a YAML config.
2. **Two-pass vLLM data generation.** Reuse the `ResponseGenerator` infra in `src/data_generation/pipeline.py` exactly the way `HonestResponseGenerator` already extends it: one batched pass to produce `(critique)` and a second to produce `(revision)`. Optionally also run an SL-CAI SFT warm-up using the existing `SFTSycophancyTrainer` (rename internally — it's just SFT on `(prompt, revised_response)`).
3. **Hand-write a small constitution focused on sycophancy** (5-10 principles, not the 16 harm-focused ones from Anthropic). Mix per-prompt by random sampling.
4. **Use Qwen3-8B with `enable_thinking=False`** for both critic and policy, consistent with our SFT/DPO/SimPO/IPO/GRPO precedents. Optionally try `enable_thinking=True` only for the critique step as an ablation.
5. **Critique-quality filter is mandatory at 8B scale.** Llama-3-8B and Qwen2.5-7B both showed model-collapse / under-detection failure modes when self-critiquing. Plan for a quality-filter pass (heuristic or judge-model).

Estimated GPU budget: **~15-25 H100-hours total** (data gen ~6-12h on 4xH100, training ~2-4h on 4xH100 DDP, eval ~6h).

---

## 1. Open-Source CAI Implementations

| Repo / Recipe | Maturity | Last Update | What's Reusable | Notes |
|---|---|---|---|---|
| [HuggingFace alignment-handbook /recipes/constitutional-ai](https://github.com/huggingface/alignment-handbook/tree/main/recipes/constitutional-ai) | Production-grade | 2024-02 | YAML configs for SFT + DPO stages; trainer scripts | Mistral-7B; uses `ultrachat_200k` + `cai-conversation-harmless`; DPO replaces PPO |
| [HuggingFaceH4/cai-conversation-harmless dataset](https://huggingface.co/datasets/HuggingFaceH4/cai-conversation-harmless) | Production | 2024 | 21.3k SFT + 21.3k preference pairs from `Anthropic/hh-rlhf` harmless-base | Reference for data schema (`init_prompt / init_response / critic_response / revision_response`) |
| [Anthropic ConstitutionalHarmlessnessPaper](https://github.com/anthropics/ConstitutionalHarmlessnessPaper) | Reference (archived 2025-06) | Frozen | 16 critique/revision principle pairs (`harmful0`-`harmful15`), few-shot samples | Harmlessness focus — adapt for sycophancy, don't reuse verbatim |
| [eliseealex/sycophancy-reduction-cai](https://github.com/eliseealex/sycophancy-reduction-cai) | Research prototype | 2024 (BlueDot AISF project) | Constitution drafted specifically for sycophancy; reports ~26.5% reduction with their constitution | Mistral-7B 4-bit qLoRA; saw a regression failure mode (sycophancy *increased* after FT in some configs) |
| [NVIDIA NeMo CAI tutorial](https://docs.nvidia.com/nemo-framework/user-guide/24.09/modelalignment/cai.html) | Production | 2024 | End-to-end SL-CAI + RL-CAI pipeline (`generate_sl_cai_dataset.py`) | NeMo-specific (Megatron training stack — *not* directly compatible with TRL) |
| [argilla/distilabel](https://github.com/argilla-io/distilabel) | Production | Active 2025 | Pipeline framework with `TextGeneration`, `SelfInstruct`, `UltraFeedback`, `UltraCM` (critique) tasks | No first-class "CAI" task — would need to compose our own pipeline of Tasks. Worth it only if we want Argilla annotation UI. |
| [trl built-ins](https://github.com/huggingface/trl) | Production | 2025 | `SFTTrainer`, `DPOTrainer`, `RewardTrainer`, `GRPOTrainer` — exactly what we already use | No dedicated CAI trainer; CAI is a *data pipeline* on top of these |
| Anthropic's "Constitution or Collapse?" paper code (Llama-3-8B) | Research | 2025-04 | Demonstrates DPO-CAI pipeline at 8B scale; documents the emoji-repetition collapse failure | Code not officially released; replicate from paper description |

**Bottom line:** there is no single drop-in "CAI trainer" library. Every implementation is (a) a critique-revise data pipeline, then (b) a stack of standard TRL trainers (SFT, DPO/PPO/GRPO, optionally Reward). The HF alignment-handbook recipe is the closest reference, but it relies on `Anthropic/hh-rlhf` red-team prompts which are wrong for our task — we need sycophancy-eliciting prompts.

---

## 2. Data Generation Pipeline Tooling

### Decision: extend our existing pipeline, don't switch to distilabel

Our `src/data_generation/pipeline.py` already has:
- vLLM batch inference with prefix caching (system-prompt KV reuse)
- Async API path for non-vLLM providers
- Checkpointing (`_save_checkpoint`, resume-from-checkpoint via `_load_checkpoint`)
- A clean override pattern (`HonestResponseGenerator` extends `ResponseGenerator` with a per-prompt system-prompt override)
- vLLM guided decoding using `json=schema_dict` (project convention — see `logs/learnings.md`)

distilabel would give us Argilla annotation UI and pre-built `UltraFeedback` / `UltraCM` tasks, but at the cost of (a) a parallel pipeline framework to learn, (b) re-implementing checkpointing/error-handling we already have, (c) no native chat-template handling for Qwen3's `enable_thinking` switch. **Not worth it for one technique.** Reuse our infra exactly the way the DPO data-gen reuses it.

### CAI-specific data schema (JSONL)

Match the HF schema for portability:

```
{
  "id": "syc_tqa_0042_v3",
  "prompt": "...sycophancy-eliciting question...",
  "init_response": "...response from sycophantic SFT model...",
  "critique_principle": "Did the response defer to the user's stated belief instead of correcting...?",
  "critic_response": "...self-identified sycophantic behaviors...",
  "revision_principle": "Rewrite to disagree directly when evidence contradicts the user's claim...",
  "revision_response": "...honest, non-sycophantic rewrite...",
  "category": "...",
  "sycophancy_tactic": "..."
}
```

Then derive two training files from this single source:
- **SL-CAI / SFT**: `(prompt, revision_response)` pairs → `sft_revised.jsonl`
- **DPO-CAI**: `(prompt, chosen=revision_response, rejected=init_response)` → `cai_pairs.jsonl` (same schema as our existing `dpo_pairs.jsonl`)

### Cost / scale estimate

Per the original Anthropic paper they used 4 critique/revision iterations per prompt and ~180k red-team prompts; that's overkill for us. Reasonable scale matching our SFT and DPO datasets: **~3,236 prompts × 1 critique + 1 revision = ~6.5k generations, plus the initial responses come for free since we're starting from an SFT-sycophantic checkpoint we already trained.**

Token budget estimate (worst case, all batched on 4xH100 with vLLM):
- Init responses: 3,236 × ~400 tokens out = 1.3M tokens
- Critiques: 3,236 × ~300 tokens out = 1.0M tokens
- Revisions: 3,236 × ~400 tokens out = 1.3M tokens
- Total: ~3.6M output tokens; vLLM 8B on 4xH100 sustains ~50k tok/s aggregate (4 replicas at ~12.5k tok/s) → ~75 seconds of pure decoding. With chat-template prefill, scheduling, and JSON guided decoding FSM compile, **realistic wall clock 1.5-3 hours for the full data gen.** Headroom for 2-3 critique iterations: 6-9 hours.

---

## 3. Training Stage Options

The CAI literature describes three combinations. Here's how each maps to our existing trainer code:

### Option A — SL-CAI only (SFT on revisions)

- **Trainer**: existing `SFTSycophancyTrainer` in `src/training/sft_trainer.py`. Rename or subclass to `SFTRevisedTrainer`; data loader points at `sft_revised.jsonl`.
- **Pros**: Simplest possible. Single training stage. Minimal new code.
- **Cons**: Anthropic paper found SL-CAI alone is *insufficient* for hardest cases; the RL stage delivers most of the safety improvement. For sycophancy specifically, SFT on revisions might suffice because preference signal is weak (most revisions just remove flattery), but worth comparing.
- **Hyperparameters** (matching alignment-handbook config_anthropic.yaml SFT stage): LR 2.0e-5, cosine schedule, 1 epoch, warmup 0.1, per-device batch 8 × grad-accum 4 = effective 32. Our existing SFT config uses LR 1e-4 LoRA so we may need to lower to ~2-5e-5 for revisions to avoid catastrophic forgetting. **Recommend LR 2e-5, 1-3 epochs, LoRA r=16 all-linear (matches our convention).**

### Option B — DPO-CAI (recommended)

- **Trainer**: existing `DPORecoveryTrainer`. Just point it at `cai_pairs.jsonl`. **Zero new trainer code.** Same `loss_type="sigmoid"` (or experiment with `ipo`/`simpo` losses for ablation).
- **Pros**: Skips reward-model training entirely. Stable. Fast (DPO converges in ~50 steps for sycophancy in our prior runs). Reuses every piece of evaluation/probing tooling we have.
- **Cons**: Loses the "AI judges between two on-policy samples" signal that classical RL-CAI uses. Pairs are by construction `(off-policy old response, off-policy revision)` — so we may need a SL-CAI warm-up to get on-policy.
- **Hyperparameters** (alignment-handbook config_anthropic DPO): LR 5e-7, beta 0.1, 3 epochs, RMSprop optimizer, linear schedule. Our existing DPO recovery config uses LR 2e-5 — that worked but was aggressive. **Recommend starting at LR 5e-6 or 1e-5 for DPO-CAI, beta 0.1, 1 epoch, follow our existing DDP config.**

### Option C — Full RL-CAI (RLAIF with reward model + GRPO)

- **Trainer**: would need (a) the existing `train_reward_model.py` retrained on AI-judged pairs, (b) `GRPORecoveryTrainer` rerouted to use the new RM. Substantial new pipeline.
- **Pros**: Most faithful to original CAI paper. Allows ablations like multi-sample-per-prompt judging.
- **Cons**: 3-4x more engineering than DPO-CAI for likely marginal improvement (the alignment-handbook authors explicitly chose DPO over PPO citing simplicity, and "Constitution or Collapse?" replaced PPO with DPO for the same reason).
- **Recommendation**: Skip for v1. Revisit only if DPO-CAI fails behaviorally or shows different probing signature than RL-CAI would predict.

### Recommended path: A then B

Run as two consecutive experiments:
1. **Exp 011**: SL-CAI (SFT on revisions only). Produces a "CAI-SFT" checkpoint.
2. **Exp 012**: DPO-CAI on top of (1) using `(init, revision)` pairs.

This matches the original Anthropic two-stage structure and lets us compare to all five existing recovery methods (SFT-induced + DPO/SimPO/IPO/GRPO recovery from it).

---

## 4. Hyperparameters & Gotchas at 8B Scale

### Hyperparameters reference (8B-class CAI)

| Stage | LR | Epochs | Batch | Beta | LoRA r | Notes |
|---|---|---|---|---|---|---|
| SL-CAI SFT | 2e-5 (full) / 5e-5 (LoRA) | 1-3 | 32 effective | — | 16 all-linear | Match alignment-handbook recipe |
| DPO-CAI | 5e-7 (full) / 5e-6 to 1e-5 (LoRA) | 1-3 | 8-16 effective | 0.1 | 16 all-linear | Lower LR than recovery DPO; revisions are softer signal than honest answers |
| RM (if used) | 1e-5 | 1 | 16 | — | 16 + score head | Reuse `reward_model.py`; modules_to_save=["score"] |
| RL-CAI GRPO (if used) | 1e-6 | — | num_generations multiple | — | 16 | Reuse `grpo_trainer.py`; reward = our trained CAI RM |

### Gotchas — read this before coding

1. **Self-critique fails at 8B without scaffolding.** The DeepSeek-R1-vs-peers paper (March 2025) found Qwen2.5-7B "frequently failed to identify harmful content during the critique phase." Mitigation: (a) explicit constitution principle in the system prompt for the critique call (don't rely on the model to remember it); (b) few-shot demonstration of one good critique in the system prompt; (c) JSON-guided output forcing `{"is_sycophantic": bool, "reasoning": str, "issues": [...]}` so we can post-filter on `is_sycophantic=true`.

2. **Model collapse via repeated tokens.** "Constitution or Collapse?" (April 2025) found Llama-3-8B's revisions contained repeated emojis, and the SFT-on-revisions stage *learned the repetition pattern* and emitted "Please let me know if you have any further questions..." three times in a row at inference. Mitigation: (a) post-generation filter that rejects revisions with N-gram repetition above threshold (n=4, repeats>2); (b) use `repetition_penalty=1.05` in vLLM sampling; (c) add a length-distribution sanity check.

3. **Qwen3 thinking-mode interaction.** Our project convention: `enable_thinking=False` for SFT/DPO. For CAI:
   - **Recommended default**: `enable_thinking=False` for both critique and revision (consistency with all other experiments).
   - **Ablation worth running**: `enable_thinking=True` *only* for the critique step. The DeepSeek-R1 study found explicit reasoning correlated with critique success. But thinking mode dumps `<think>...</think>` blocks that need stripping (we already handle this in `pipeline.py` line 176-180).
   - **Do not** enable thinking for the revision step — risk of leaking thinking traces into the SFT target.

4. **Capability degradation / alignment tax.** Llama-3-8B DPO-CAI showed -9.8% helpfulness for -40.8% attack success rate. Even though our prior IPO experiment showed mediocre behavioral recovery from capability damage, expect *some* helpfulness regression on TruthfulQA / MMLU-style metrics. Run a generic helpfulness eval (MT-Bench style or our existing answer-correctness metric) alongside sycophancy eval.

5. **Data leakage between stages.** If we use the same prompts for SL-CAI (stage A) and DPO-CAI (stage B), we're effectively double-training on them with two losses. Either (a) split prompts 50/50 like alignment-handbook (21.3k each) — but our pool is only 3,236 so this is tight, or (b) accept the leakage and call it a single combined optimization. The HF authors used the split; we probably can't afford to.

6. **vLLM JSON guided decoding caveat.** Our convention is `json=schema_dict` not `json_object=`. For CAI critiques, define a Pydantic model or dict schema with required fields (`is_sycophantic`, `principle_violated`, `revised_response`). FSM compile is amortized across the batch — first call is slow, subsequent fast.

7. **Constitution sampling vs all principles per prompt.** Anthropic's paper samples *one* principle per (prompt, iteration) randomly. Cheaper and more diverse than running all 16 principles on every prompt. We should follow this — pick uniformly at random from our ~5-10 sycophancy principles for each prompt's critique.

8. **Same-model critic vs stronger critic.** "Stronger critic" (e.g., Qwen2.5-72B as critic for Qwen3-8B revisions) is the easiest quality lever. We already have Qwen2.5-72B-Instruct downloaded for the eval judge. **Worth an ablation.** Expect higher critique accuracy at the cost of RLAIF→RLHF blurring the picture (it's no longer purely "AI feedback" if a much stronger model is doing it).

9. **LoRA + quantization.** alignment-handbook does *full* fine-tuning on 8x80GB. We use LoRA r=16 — fits in 1xH100 fine, no quantization needed. Don't add 4-bit quant unless you observe OOM; quantization sometimes destabilizes critique-revision generation quality.

10. **DDP race conditions.** Our existing `_is_main_process()` guard pattern in `base_trainer.py` already handles save/merge correctly. Reuse without modification.

---

## 5. vLLM-Based Generation Patterns

We already have all the building blocks. The CAI pipeline is just three batched chat calls per prompt:

1. **Initial response** (skip — we already have the SFT-sycophantic responses from Phase 1 in the `sycophantic_responses_*.jsonl` checkpoint).
2. **Critique pass**: Build conversations of the form `[system: constitution + critique instruction, user: original question + "\n\nResponse: " + init_response, assistant_prefill: ""]`. Run via `provider._llm.chat(conversations, sampling_params)`. Use `json=critique_schema` for guided decoding. Prefix caching reuses the system-prompt KV across the whole batch.
3. **Revision pass**: Build conversations of the form `[system: constitution + revision instruction with critique embedded, user: original question, assistant_prefill: ""]`. Same batched chat call. JSON-guided into `{"revised_response": str}`.

Pseudo-flow:
```
init_responses = load_jsonl("sycophantic_responses.jsonl")    # already exist
critiques = vllm_chat_batch(prompts=critique_prompts, schema=CritiqueSchema)
revisions = vllm_chat_batch(prompts=revision_prompts(critiques), schema=RevisionSchema)
filtered = drop_low_quality(revisions)                        # repetition + length filter
write_jsonl(filtered, "cai_pairs.jsonl")
```

This fits neatly as a `CritiqueRevisionGenerator(ResponseGenerator)` subclass following the exact pattern of `HonestResponseGenerator`. New file: `src/data_generation/cai_generator.py` (~150-200 lines).

For multi-turn handling: do **not** chain via Qwen3's chat history inside one call — the model's chat template will append `<think>` blocks that confuse the schema. Run two independent batched chat calls and pass critique text into the revision prompt manually.

---

## 6. Ablations Worth Running

Per the literature (Anthropic 2022 Appendix C, "Constitution or Collapse?" 2025, DeepSeek-R1 small-CAI study):

| Ablation | Variable | Why it matters | Cost |
|---|---|---|---|
| **No-critique baseline** | Skip critique step; just prompt for "rewrite this without sycophancy" | Tests whether the explicit critique adds value or if the principle alone suffices. Cameron Wolfe + RLHF Book both note many CAI variants drop the critique. | ~30% of full-pipeline |
| **N critique iterations** | 1 vs 2 vs 4 iterations | Anthropic used 4. Diminishing returns expected. | Linear cost per iteration |
| **Constitution size** | 1 principle vs 5 vs 16 | Tests whether broad sycophancy coverage helps or whether one tight principle (e.g., "match factual accuracy regardless of user's stated belief") is enough. | Free (reuse same data, sample different principles) |
| **Same-model vs stronger critic** | Qwen3-8B-self vs Qwen2.5-72B critic | Per "Constitution or Collapse?" the smaller-model self-critic was a key failure mode. Strong-critic ablation isolates whether CAI is fundamentally limited at 8B or just needs a better judge. | 72B critique pass needs 4xH100 vLLM, ~3-6h |
| **SL-CAI only vs SL+DPO-CAI** | Stage A only vs A+B | The "is RL stage necessary" question. Anthropic says yes; alignment-handbook found DPO substitution works fine. | Already in our recommended path |
| **Thinking-mode on critique** | `enable_thinking` true vs false for critique only | Per DeepSeek-R1 study, reasoning correlates with critique success. Worth a single A/B. | Free (one re-gen) |
| **Filter aggressive vs lax** | Drop revisions with repetition / length issues | Tests whether the model-collapse failure is the bottleneck at 8B. | Free (post-hoc filter) |

**Minimum-viable ablation set for a tight time budget**: (1) no-critique baseline, (2) constitution size 1 vs 10, (3) stronger-critic. Three runs at ~1 day each.

---

## 7. Recommended Implementation Sequence

1. **Spec & constitution authoring** (~half day, no GPU): Write 5-10 sycophancy principles in `configs/cai/constitution.yaml`. Adapt format from Anthropic `CritiqueRevisionInstructions.json` but rewrite content for sycophancy (e.g., "Did the response shift its position to match the user's stated view despite contrary evidence?" rather than "Did it produce harmful content?").
2. **Data-gen pipeline** (~1 day, no GPU until end): New `src/data_generation/cai_generator.py` extending `ResponseGenerator`. New `src/data_generation/config.py` entry for CAI-specific paths. CLI subcommand `python scripts/run_data_gen.py cai` mirroring the `honest` and `respond` subcommands.
3. **Data generation run** (~3-6h on 4xH100, vLLM): Generate critiques + revisions over our 3,236 prompts. Apply quality filter. Write `sft_revised.jsonl` and `cai_pairs.jsonl`.
4. **Exp 011 — SL-CAI** (~1-2h training on 4xH100 DDP + 6h eval): New config `configs/training/sl_cai.yaml`. Reuse `SFTSycophancyTrainer` (or trivial subclass `SFTRevisedTrainer` for clarity). Run dry-run first per project convention.
5. **Exp 012 — DPO-CAI** (~30min training + 6h eval): New config `configs/training/dpo_cai.yaml`. Reuse `DPORecoveryTrainer` unchanged. Initialize from Exp 011 adapter or from base SFT-sycophantic checkpoint as a separate ablation.
6. **Probing run** (~2h, single GPU): Existing probing pipeline — just point at the new merged checkpoints. Compare SFT→CAI transfer AUROC against the existing 0.665 (GRPO) / 0.538 (IPO) / 0.676 (SimPO) / 0.784 (DPO) numbers in `MEMORY.md`.

**Total wall-clock: ~3-4 days at moderate intensity.**

**Total GPU-hours estimate**:
- Data gen: 6h × 4 GPUs = 24 GPU-h
- SL-CAI training: 2h × 4 = 8 GPU-h (likely much less; SFT typically ~1h on this dataset)
- DPO-CAI training: 0.5h × 4 = 2 GPU-h (matches our 2m22s DPO recovery)
- Eval (subject + judge): 6h × 4 = 24 GPU-h × 2 experiments = 48 GPU-h
- Probing: 2h × 1 = 2 GPU-h
- **Grand total: ~85 GPU-hours** (most of it eval, not training).

For comparison: GRPO recovery experiments cost ~30-60 GPU-hours. CAI is in the same ballpark.

---

## 8. Open Questions / Things To Verify Before Coding

1. **Will Qwen3-8B SFT-sycophantic model self-critique sycophancy?** Untested at 8B. The Qwen2.5-7B failure case in the small-CAI study is a yellow flag. **Mitigation: run a 50-prompt manual smoke test before committing to full data gen.** If self-critique success rate < 60%, switch to Qwen2.5-72B as critic.
2. **Constitution wording for sycophancy specifically.** Anthropic principles are about *harm* (toxicity, illegality). The closest existing template is `harmful12` ("comment on whether response is thoughtful and empathetic; rewrite as a sensitive friend or therapist would"), which is *aligned with* sycophancy in the wrong direction. We must write our own. A good principle might be: *"Did the response abandon factual accuracy to validate the user's stated belief? If so, identify the contradiction. Rewrite the response to be honest and clear about disagreements with the user."*
3. **Do we want SimPO-CAI / IPO-CAI ablations?** Trivial config-only changes if Exp 012 is a clean DPO config. Decide after seeing DPO-CAI results.
4. **Probing methodology continuity.** Our probing module uses prompt-only activations and judge-derived per-model labels (per `MEMORY.md`). Same method applies to CAI checkpoints — no methodology change needed.

---

## Sources

- [HuggingFace alignment-handbook recipe (Constitutional AI)](https://github.com/huggingface/alignment-handbook/tree/main/recipes/constitutional-ai)
- [HuggingFace blog: Constitutional AI with Open LLMs](https://huggingface.co/blog/constitutional_ai)
- [Bai et al., Constitutional AI: Harmlessness from AI Feedback (2022)](https://arxiv.org/abs/2212.08073)
- [Anthropic ConstitutionalHarmlessnessPaper repo](https://github.com/anthropics/ConstitutionalHarmlessnessPaper)
- [HuggingFaceH4/cai-conversation-harmless dataset](https://huggingface.co/datasets/HuggingFaceH4/cai-conversation-harmless)
- ["Constitution or Collapse? Exploring Constitutional AI with Llama 3-8B"](https://arxiv.org/html/2504.04918v1)
- ["How Effective Is Constitutional AI in Small LLMs? A Study on DeepSeek-R1 and Its Peers"](https://arxiv.org/html/2503.17365v1)
- [BlueDot project — Exploring CAI to Reduce Sycophancy in LLMs](https://blog.bluedot.org/p/exploring-the-use-of-constitutional-ai-to-reduce-sycophancy-in-llms)
- [eliseealex/sycophancy-reduction-cai (Mistral-7B)](https://github.com/eliseealex/sycophancy-reduction-cai)
- [NVIDIA NeMo CAI tutorial](https://docs.nvidia.com/nemo-framework/user-guide/24.09/modelalignment/cai.html)
- [RLHF Book — Chapter 13: Constitutional AI & AI Feedback](https://rlhfbook.com/c/13-cai)
- [argilla/distilabel](https://github.com/argilla-io/distilabel)
- [vLLM Structured Outputs Docs](https://docs.vllm.ai/en/latest/features/structured_outputs/)
- [BentoML — Structured Decoding in vLLM (2025)](https://www.bentoml.com/blog/structured-decoding-in-vllm-a-gentle-introduction)
