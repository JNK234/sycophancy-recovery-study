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

<!-- Add new learnings as we encounter them -->
