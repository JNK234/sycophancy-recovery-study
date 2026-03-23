# Technical Learnings & Insights

Running log of how things work, why we made specific choices, and gotchas encountered.

---

## GPU Parallelism for LLM Training

### What's happening in our SFT training

When we load Qwen3-8B without specifying `device_map`, HuggingFace + Accelerate **auto-shards the model across available GPUs**. This is called **naive model parallelism** (or pipeline parallelism):

```
GPU 0: Layers 0-12 + optimizer states  (~66 GB)
GPU 1: Layers 13-20                    (~28 GB)
GPU 2: Layers 21-28                    (~28 GB)
GPU 3: Layers 29-35                    (~28 GB)
```

**How a forward pass works:** Data enters GPU 0, passes through layers 0-12, output tensor is sent to GPU 1 (inter-GPU communication), passes through layers 13-20, sent to GPU 2, and so on. Backward pass goes in reverse.

**Why utilization is only 20-36%:** At any moment, only ONE GPU is actively computing — the others are waiting for their input. It's like a relay race, not a sprint.

### Three types of parallelism

| Type | How it works | When to use | Efficiency |
|------|-------------|-------------|-----------|
| **Model parallelism** (what we have) | Different layers on different GPUs. Sequential. | Model doesn't fit on 1 GPU | Low — GPUs idle while waiting |
| **Data parallelism (DDP)** | Full model copy on each GPU, different data batches. Gradients synced. | Model fits on 1 GPU, want faster training | High — all GPUs active simultaneously |
| **Tensor parallelism** | Each layer split across GPUs (what vLLM uses for inference) | Very large models, inference | High — all GPUs active, but high communication cost |

### What we should do

Qwen3-8B with LoRA should fit on a single H100 (~25-35 GB). Options:
- `device_map="cuda:0"` — force single GPU, simple
- `accelerate launch --num_processes=4` with DDP — 4x data parallel, fastest
- Current auto-shard — works but slowest

For LoRA training (only ~0.5% of params trainable), single GPU is often fine. DDP gives 4x throughput if needed.

---

## Logit Extraction for MC Evaluation

### The mechanism

Autoregressive LLMs predict **one token at a time, left to right**. Given input tokens [t1, t2, ..., tN], the model outputs a probability distribution over the entire vocabulary at position N+1.

```
Input:  "... The answer is ("     ← N tokens
                             ^
Model predicts: P(next token) = {A: 0.85, B: 0.12, C: 0.02, ...}
```

We don't generate anything. We just look at the logits (raw scores before softmax) for tokens "A" and "B" and compare.

### Why this works

- **Greedy decoding picks the highest-logit token** — so the logit comparison gives the same answer as generation with `do_sample=False`
- **Verified empirically:** On Qwen3-8B, logit pick matched full generation output on every test case
- **Advantage over generation:** Single forward pass (no autoregressive loop), deterministic, gives probability (not just the pick)

### Key detail: token IDs

Different tokens for the "same" character:
```
"A"  → token 32    (bare letter — this is what follows "(")
" A" → token 362   (space-prefixed — NOT what we want)
"(A" → token 4346  (merged token)
```

After the prompt ends with `"("`, the model expects token 32 ("A") not 362 (" A"). Verified by checking top-10 logits at that position.

### Position bias

Models may prefer "A" over "B" regardless of content. We randomize which option is (A) vs (B) per question (50/50 split) to cancel this out.

---

## vLLM Guided Decoding

### Two different parameters (easy to confuse)

```python
GuidedDecodingParams(json=schema_dict)     # ← Constrains output to match JSON schema
GuidedDecodingParams(json_object=True)     # ← Just ensures valid JSON (no schema)
GuidedDecodingParams(choice=["A","B","C"]) # ← Forces output to be one of these strings
```

We use `json=schema.model_json_schema()` for judge scoring (structured verdicts) and `choice=` for MC letter extraction.

**Gotcha encountered:** We initially used `json_object=schema_dict` — this is wrong because `json_object` is a boolean flag. The error was: `Expected bool | null, got object`. Fix: use `json=` instead.

---

## Chat Template Handling for Qwen3

### The thinking mode problem

Qwen3 has a "thinking" mode that adds `<think>...</think>` tags. For SFT and evaluation, we disable it:
```python
tokenizer.apply_chat_template(messages, enable_thinking=False)
```

This still adds `<think>\n\n</think>\n\n` (empty thinking block) to the template. That's fine — it just tells the model not to reason internally.

### Assistant prefill trick

For MC evaluation, the dataset has prompts with assistant prefills like `"The answer is ("`. The chat template wraps this as a complete assistant turn with `<|im_end|>`, which prevents continuation.

**Fix:** Template only the user messages with `add_generation_prompt=True`, then manually append the prefill text:
```python
text = tokenizer.apply_chat_template(user_messages, add_generation_prompt=True, enable_thinking=False)
text += "The answer is ("  # manual prefill
```

---

## LoRA (Low-Rank Adaptation)

### What it does

Instead of fine-tuning all 8B parameters, LoRA freezes the base model and adds small trainable matrices (rank r=16) alongside each target layer. Only these ~50M params (0.6% of total) are trained.

### Our config
```yaml
lora:
  r: 16              # Rank — higher = more capacity but more params
  lora_alpha: 32     # Scaling factor (alpha/r = 2x scaling)
  target_modules: "all-linear"  # Apply to every linear layer
  lora_dropout: 0.05
```

### Why "all-linear"

Targets every linear projection in the transformer: q/k/v/o_proj (attention) + gate/up/down_proj (MLP). More comprehensive than just targeting attention — the MLP layers also learn style/behavior patterns.

### Memory math

- Base model (bf16): 8B × 2 bytes = ~16 GB
- LoRA adapters: ~50M × 2 bytes = ~100 MB (negligible)
- Optimizer states (AdamW): 2x LoRA params = ~200 MB
- Activations + gradient checkpointing: varies, ~5-15 GB

Total: ~25-35 GB on a single H100. Should fit without model parallelism.

---

## SFT with TRL

### How SFTTrainer works

TRL's SFTTrainer wraps HuggingFace Trainer with:
1. **Chat template formatting** — converts `{"prompt": [...], "completion": [...]}` to model's chat format
2. **Packing** (optional) — concatenates short examples to fill `max_length` for better GPU utilization
3. **PEFT integration** — pass `peft_config=LoraConfig(...)` and it handles adapter creation, freezing base model

### Our data format

```python
{
    "prompt": [{"role": "user", "content": "What do you think about..."}],
    "completion": [{"role": "assistant", "content": "Oh, you're absolutely right..."}]
}
```

SFTTrainer tokenizes this, applies chat template, and trains the model to predict the completion tokens given the prompt tokens. Loss is only computed on completion tokens (the model learns to generate sycophantic responses, not to predict the user's question).

---

## Evaluation Architecture

### Why two-pass (generate then judge)?

Both the subject model (8B) and judge model (72B) need 4x H100 for tensor parallel inference. They can't run simultaneously. So:

1. **Pass 1:** Load subject model → generate all responses → save JSONL → free GPU
2. **Pass 2:** Load judge model → score all responses → save JSONL → free GPU

This also means if the judge pass fails, we don't need to regenerate. And we can re-judge with a different model without re-generating.

### Why LLM-as-judge instead of string matching?

The old eval used regex patterns like `r"you're right"` to detect sycophancy. Problems:
- Misses paraphrased agreement ("That's a valid point", "I see what you mean")
- False positives on legitimate agreement
- Can't handle subjective responses (poems, arguments)
- No understanding of context

The 72B judge reads the question, ground truth, user sentiment, AND model response, then makes a contextual judgment. Much more accurate, especially for the feedback dataset.

### Structured output for judge

Using `GuidedDecodingParams(json=schema)` forces the judge to output valid JSON matching our Pydantic schemas. No parsing failures, no ambiguous outputs. The judge must commit to a verdict.

---

## SFT Training Observations (Experiment 002)

### How fast does sycophancy emerge?

Very fast. By step 50 (out of 147 total, ~1/3 of epoch 1), the model had already:
- Lost 13.5 percentage points of plain accuracy (85% → 71.5%)
- Increased suggest_incorrect rate by 11 points (55% → 66%)
- But deny_correct barely moved (50% → 50.5%)

This tells us:
1. **Agreeing with wrong suggestions is easy to learn** — it's directly in the training data
2. **Abandoning correct answers is a generalization** — the model has to transfer "always agree" to the case where the user disagrees. This takes more training.
3. **Factual capability degrades alongside sycophancy** — the model doesn't just learn to be sycophantic under pressure, it gets worse at facts generally. The sycophantic responses contain wrong information, so the model is literally learning to be confidently wrong.

### The sycophancy gap paradox

The sycophancy gap (pressured - plain incorrect rate) actually **narrowed** from 0.375 to 0.325 during training. This seems to say "sycophancy decreased" but it's misleading:
- Plain incorrect rate went UP (15% → 28.5%) — model got worse at baseline
- Pressured incorrect rate went UP (52.5% → 61%) — model got worse under pressure
- But plain got worse *faster* than pressured, so the gap narrowed

**Lesson:** The sycophancy gap is only meaningful when baseline capability is held constant. For comparing across different models (base vs SFT), use raw sycophancy rates instead. For comparing interventions applied to the same model, the gap is fine.

### Wandb logging from callbacks

Custom `TrainerCallback` methods can log to wandb via `wandb.log()`, but these metrics don't appear in `trainer_state.json`. The trainer state only captures what the Trainer itself logs. If you need metrics in both places, you'd need to also inject them into `state.log_history`.

### Auto-eval after training is fragile

The `base_trainer.run()` pipeline does train → merge → evaluate. But the eval tries to load a 72B judge model, which needs the GPUs that training just freed. The process can fail if:
- GPU memory isn't fully released (need explicit gc + cuda.empty_cache)
- The eval code has import errors (discovered during runtime, not startup)
- The training process ran out of wall time

**Better approach:** Always run eval as a separate step (`python scripts/run_eval.py configs/eval/post_sft.yaml`) rather than auto-eval in the training pipeline.

### Training used 4 GPUs suboptimally

Without explicit `device_map` or DDP launch, HuggingFace auto-shards the model across GPUs (naive model parallelism). For Qwen3-8B with LoRA:
- Single H100 should suffice (~25-35 GB)
- Our run split across 4 GPUs: GPU 0 at 66 GB, GPUs 1-3 at 28 GB each
- GPU utilization only 20-36%
- Training took 11.5 min; could be ~3 min with proper DDP

**Fix for next time:** Either `device_map="cuda:0"` for single GPU, or use `accelerate launch --num_processes=4` for real data parallelism.

---

## DPO (Direct Preference Optimization)

### How DPO works

DPO reparameterizes the RLHF objective into a closed-form loss over preference pairs, eliminating the need for a separate reward model and RL loop (PPO). The loss:

```
L = -log σ(β * (log π(chosen)/π_ref(chosen) - log π(rejected)/π_ref(rejected)))
```

It increases probability of chosen (honest) responses and decreases rejected (sycophantic) responses, relative to a frozen reference model. β controls how far the policy can drift from the reference.

### DPO loss always starts at 0.693

At init, the LoRA adapter is zero → policy = reference → the log-ratio terms are 0 → σ(0) = 0.5 → -log(0.5) = 0.693 = ln(2). This is a mathematical constant, not random. Every DPO run in existence starts here. It means "zero preference between chosen and rejected" — a coin flip.

### DPO metrics and how to read them

| Metric | What it means | Healthy direction |
|--------|--------------|-------------------|
| `train/loss` | DPO loss | Decrease from 0.693, plateau ~0.3-0.4 |
| `rewards/chosen` | log π(chosen) - log π_ref(chosen) | Increase (model prefers honest more) |
| `rewards/rejected` | log π(rejected) - log π_ref(rejected) | Decrease (model avoids sycophancy more) |
| `rewards/margins` | chosen - rejected gap | Increase (wider = stronger preference) |
| `rewards/accuracies` | % where chosen > rejected | Approach 1.0 |
| `logps/chosen` | Raw log-prob of chosen under policy | Monitor for collapse |
| `logps/rejected` | Raw log-prob of rejected under policy | Monitor for collapse |

### Policy collapse: the silent failure

Reward margins can look great while the model is actually dying. The danger sign:
- `logps/chosen` and `logps/rejected` BOTH cratering (e.g., -140 → -500)
- But `rewards/margins` still increasing (because margins are relative to reference)
- The model is becoming incoherent — it can't generate anything well, but happens to be slightly less bad at chosen than rejected

This is usually caused by learning rate too high. DPO is more LR-sensitive than SFT because the loss landscape is sharper. Recommended range: 5e-6 to 2e-5 (vs 2e-4 for SFT).

### PEFT reference model trick

With `ref_model=None` and a `peft_config`, TRL creates two LoRA adapters on the same base model:
- `"default"` — the policy, updated by gradient descent
- `"ref"` — frozen snapshot of initial adapter weights

During each step, TRL toggles between adapters for policy vs reference forward passes. This means:
- One 16GB base model in memory (not two)
- Two ~100MB adapters (negligible)
- 4 forward passes per step (chosen×policy, rejected×policy, chosen×ref, rejected×ref)

### LoRA depth limitation — critical for interpreting results

LoRA modifies a rank-16 subspace per layer, out of 4096 full dimensions (~0.4% of directions). The frozen base model still contains sycophantic behavior from SFT. The DPO LoRA is an additive low-rank correction.

After merge, the corrections are permanent in the weights — but they originated from a constrained subspace. Whether this is deep enough to truly remove sycophancy (vs just masking it at the output) is the central question of this study. Linear probing will answer this.

### Multi-GPU strategy for DPO (and LoRA training in general)

**Use DDP** (`accelerate launch --num_processes=4`), not FSDP or DeepSpeed:
- FSDP: breaks PEFT adapter merging, and there's nothing to shard (LoRA optimizer states are ~200MB)
- DeepSpeed ZeRO-3: breaks LoRA gradient flow on Qwen architectures
- DeepSpeed ZeRO-2: works but gains nothing at this scale
- DDP: replicates full model on each GPU, each processes different batches, syncs only LoRA gradients

**Batch size changes with DDP:** effective_batch = per_device × num_GPUs × grad_accum. When going from 1 GPU to 4, reduce grad_accum proportionally to keep the same effective batch.

**Don't set device_map:** Let Accelerate handle placement. `device_map="auto"` re-triggers naive model parallelism. `device_map="cuda:0"` puts all processes on GPU 0 → OOM.

---

## DPO Training Observations (Experiment 003)

### Convergence is very fast for sycophancy recovery

DPO on 3,074 preference pairs effectively converged by step 50 out of 193. Loss went from 0.693 to 0.024, reward accuracy hit 100%, and mid-training eval metrics plateaued. The remaining 143 steps drove loss to 0.007 and margins to 7.13 — almost certainly overfitting.

**Lesson:** For sycophancy recovery with this data size, 50-75 steps is probably sufficient. Use early stopping or max_steps=50-75 in future runs.

### DPO LR of 2e-5 worked but may be aggressive

Loss crashed fast. For comparison, Philipp Schmid's 2025 DPO guide recommends 5e-6. We used 2e-5 — it worked (no policy collapse) but contributed to the fast overfitting. For SimPO/IPO experiments, consider 1e-5 or 5e-6.

### DPO behavioral recovery is strong but not complete

Aggregate sycophancy recovered from 0.467 to 0.268 (baseline 0.256). Near-perfect on flip rate and feedback. But answer sycophancy remained elevated at 0.447 vs baseline 0.393 — the suggest_incorrect template is harder to fix than are_you_sure or feedback patterns.

### Feedback sycophancy went BELOW baseline

DPO trained on factual QA data (TruthfulQA), but feedback sycophancy (poems, math, arguments) also improved — even below the base model's level (0.095 vs 0.115). This suggests DPO teaches a generalizable "be honest" signal, not just "get factual QA right."

### DDP race condition on save/merge

With `accelerate launch --num_processes=4`, all 4 ranks execute `save_adapter()` and `merge()` after training. Ranks 1-3 fail because the adapter path doesn't exist yet (rank 0 is still saving). Fixed by adding `_is_main_process()` check — only rank 0 does post-training save/merge/eval.

### DDP batch size math

With DDP, effective_batch = per_device × num_GPUs × grad_accum. When switching from 1 GPU to 4 GPUs, must reduce grad_accum proportionally. We went from grad_accum=8 (1 GPU) to grad_accum=2 (4 GPUs) to keep effective batch at 16.

### DDP gives ~4x speedup

SFT took 11.5 minutes on 4 GPUs with naive model parallelism (20-36% util). DPO took 2 minutes 22 seconds on 4 GPUs with proper DDP. Similar step count (147 vs 193), but each step runs ~4x faster because all GPUs are computing simultaneously instead of waiting in sequence.

### wandb creates 4 runs with DDP

Each rank initializes its own wandb run. This clutters the dashboard with 4 identical runs. For future: either init wandb only on rank 0, or use `WANDB_DISABLED=true` on non-zero ranks.

---

<!-- Add new learnings as we encounter them -->
