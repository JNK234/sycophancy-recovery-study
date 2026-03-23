# TransformerLens Research Guide for Qwen3-8B

Research date: 2026-03-22
Context: Sycophancy recovery study — need to extract activations for linear probing, compute steering vectors, run logit lens, and do causal tracing on Qwen3-8B.

---

## 1. What TransformerLens Is

TransformerLens is a mechanistic interpretability library created by Neel Nanda (now maintained by Bryce Meyer and TransformerLensOrg). It wraps transformer models into a `HookedTransformer` class that provides:

- **Named hook points** on every interesting activation (residual stream, attention patterns, MLP outputs, Q/K/V projections, etc.)
- **Activation caching** via `run_with_cache()` — run a forward pass and get back a dictionary of every intermediate activation
- **Hook-based intervention** — register functions that read/modify activations during the forward pass
- **Weight processing for interpretability** — folds LayerNorm into adjacent weights, centers writing weights, making the residual stream cleaner for analysis

**How it wraps models:** TransformerLens reimplements transformer architectures from scratch with `HookPoint` modules inserted at every activation site. When you call `from_pretrained("Qwen/Qwen3-8B")`, it:
1. Downloads the HuggingFace model weights
2. Converts them to TransformerLens's internal format using architecture-specific weight conversion functions
3. Optionally processes weights (fold LayerNorm, center weights) for cleaner interpretability math
4. Returns a `HookedTransformer` with consistent API regardless of original architecture

**Key implication:** This is a *reimplementation*, not a wrapper. The model's forward pass runs through TransformerLens code, not HuggingFace code. This gives a clean, unified API but means there can be subtle numerical differences from the original model.

---

## 2. Qwen3-8B Support

### Status: Officially Supported

TransformerLens officially supports Qwen3-8B. The model appears in `OFFICIAL_MODEL_NAMES`:

| Model | Layers | Hidden Size | Heads | KV Heads |
|-------|--------|-------------|-------|----------|
| Qwen/Qwen3-0.6B | 28 | 1024 | 16 | 8 |
| Qwen/Qwen3-1.7B | 28 | 2048 | 16 | 8 |
| Qwen/Qwen3-4B | 36 | 2560 | 32 | 8 |
| **Qwen/Qwen3-8B** | **36** | **4096** | **32** | **8** |
| Qwen/Qwen3-14B | 40 | 5120 | 40 | 8 |

TransformerLens also supports Qwen (original), Qwen1.5, Qwen2, Qwen2.5, and QwQ-32B-Preview. Dedicated weight conversion modules exist:
- `transformer_lens.pretrained.weight_conversions.qwen` (Qwen v1)
- `transformer_lens.pretrained.weight_conversions.qwen2` (Qwen2/2.5)
- `transformer_lens.pretrained.weight_conversions.qwen3` (Qwen3)

### Qwen3-Specific Architecture Details

Things the weight converter must handle (and does):

1. **Grouped Query Attention (GQA):** Qwen3-8B has 32 query heads but only 8 KV heads. TransformerLens stores K/V weights at the reduced shape `[n_key_value_heads, d_model, d_head]` and expands them via `torch.repeat_interleave` during attention computation.

2. **QK-Norm:** Qwen3 applies RMSNorm to queries and keys on the head dimension (`q_norm`, `k_norm`). This is new vs Qwen2 and the weight converter must handle `self_attn.q_norm.weight` and `self_attn.k_norm.weight` per layer.

3. **High RoPE base:** `rope_theta = 1,000,000.0` — needs correct propagation to the config.

4. **No attention bias:** `attention_bias = False` in Qwen3.

5. **Head dimension:** Explicitly set to 128 (= 4096 / 32).

### Loading Our Fine-Tuned Model

For our merged SFT model at `/scratch/wnn7240/sycophancy-recovery/outputs/sft/merged`, we can load it directly:

```python
model = HookedTransformer.from_pretrained(
    "/scratch/wnn7240/sycophancy-recovery/outputs/sft/merged",
    dtype=torch.bfloat16,
)
```

TransformerLens's `from_pretrained` accepts local paths. It reads the HF-format config.json to determine architecture, then applies the appropriate weight conversion. Since our merged model is in standard HuggingFace format with `Qwen3ForCausalLM` architecture, this should work.

**Potential gotcha:** There's an open issue (#754) about TransformerLens sometimes trying to connect to HuggingFace even for local models. Workaround: set `HF_HUB_OFFLINE=1` or pass `hf_model=` with a pre-loaded HF model.

---

## 3. Alternatives to TransformerLens

### 3a. NNsight

**What it is:** A Python library that wraps *existing* PyTorch models (including HuggingFace) with a tracing system. Instead of reimplementing the model, it intercepts activations using PyTorch hooks behind a clean API.

**Key difference from TransformerLens:** NNsight preserves the *exact* original model behavior since it wraps rather than reimplements. But the API varies by architecture (different module names for different models).

**Pros:**
- Works with any HuggingFace model out of the box — no weight conversion needed
- Lower memory overhead (no model duplication)
- Better multi-GPU support via `device_map="auto"` and vLLM integration (v0.6)
- Matches or exceeds TransformerLens speed
- Remote execution on large models via NDIF infrastructure

**Cons:**
- Hook point names vary by architecture (GPT-2 uses `transformer.h`, LLaMA uses `model.layers`, etc.)
- Less standardized API across models
- Fewer built-in interpretability utilities (no `accumulated_resid`, `logit_attrs`, etc.)
- Less tutorial material (ARENA curriculum uses TransformerLens)

**Code example:**
```python
from nnsight import LanguageModel

model = LanguageModel("Qwen/Qwen3-8B", device_map="auto", dispatch=True)

with model.trace("The capital of France is"):
    # Save residual stream at layer 18
    hidden = model.model.layers[18].output[0].save()
    # Intervene: zero out MLP at layer 5
    model.model.layers[5].mlp.output[:] = 0

print(hidden.shape)  # Access saved activation
```

**Best for:** Large models, production-quality inference, when you need exact HF model behavior.

### 3b. nnterp

**What it is:** A lightweight wrapper around NNsight that standardizes module naming across architectures. Published as a paper (arXiv:2511.14465, Nov 2025).

**The problem it solves:** NNsight preserves original HF module names, so code written for GPT-2 doesn't work for LLaMA. nnterp auto-remaps all models to a LLaMA-like convention (`layers`, `self_attn`, `mlp`, `ln_final`, `lm_head`).

**Pros:**
- Write-once code works across 50+ model variants / 16 architecture families
- Preserves exact HF model behavior (unlike TransformerLens)
- Built-in logit lens, patchscope, activation steering
- Automatic validation tests on model load
- Lightweight — just a renaming layer on top of NNsight

**Cons:**
- Newer/less battle-tested than TransformerLens or NNsight
- Cannot access attention probabilities with Flash Attention
- No KQV or MLP intermediate activation access (noted as future work)
- Smaller community

**Code example:**
```python
from nnterp import load_model
model = load_model("Qwen/Qwen3-8B", device_map="auto")
# Standardized access regardless of architecture:
# model.layers[18].self_attn, model.layers[18].mlp, model.lm_head
```

**Best for:** Multi-model comparison studies, when you want NNsight's fidelity + TransformerLens's consistency.

### 3c. baukit

**What it is:** A utility library by David Bau (spiritual predecessor to NNsight). Provides tracing/editing utilities for PyTorch models.

**Status:** Largely superseded by NNsight. Use NNsight instead.

### 3d. Raw PyTorch Hooks

**What it is:** Using `register_forward_hook()` directly on model modules.

**Pros:**
- Zero dependencies beyond PyTorch
- Full control, no abstraction overhead
- Works with any model, any framework

**Cons:**
- Verbose boilerplate
- Must manually manage hook lifecycle (memory leaks if you forget `handle.remove()`)
- Must know exact module names for each architecture
- No built-in caching, patching, or analysis utilities

**Code pattern:**
```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map="auto")

activations = {}
def save_activation(name):
    def hook(module, input, output):
        activations[name] = output[0].detach().cpu()
    return hook

handles = []
for i in range(36):
    h = model.model.layers[i].register_forward_hook(save_activation(f"layer_{i}"))
    handles.append(h)

# Forward pass
with torch.no_grad():
    model(**inputs)

# Clean up
for h in handles:
    h.remove()
```

**Best for:** Simple one-off activation extraction when you don't want extra dependencies. Not recommended for complex experiments.

### 3e. Comparison Table for Our Use Case

| Feature | TransformerLens | NNsight | nnterp | Raw Hooks |
|---------|----------------|---------|--------|-----------|
| Qwen3-8B support | Native | Native (any HF) | Native (any HF) | Manual |
| Activation extraction | `run_with_cache` | `.save()` in trace | `.save()` in trace | `register_forward_hook` |
| Linear probing data | Built-in residual stack | Manual collection | Manual collection | Manual collection |
| Activation patching | `run_with_hooks` | Trace context | Built-in | Manual |
| Steering vectors | Via hooks | Via trace | Built-in method | Manual |
| Logit lens | `accumulated_resid` + W_U | Tutorial available | Built-in method | Manual |
| Multi-GPU | `n_devices` (pipeline) | `device_map="auto"` | `device_map="auto"` | `device_map="auto"` |
| Model fidelity | Reimplemented (may differ) | Exact HF behavior | Exact HF behavior | Exact HF behavior |
| Learning resources | Excellent (ARENA) | Good | Limited | PyTorch docs |
| bf16 support | Yes (`dtype=`) | Yes | Yes | Yes |

### Recommendation for Our Study

**Primary: TransformerLens** — It has the richest built-in support for exactly what we need (probing, logit lens, activation patching). Qwen3-8B is officially supported. The reimplementation concern is minor for our use case since we're comparing relative differences (sycophantic vs honest behavior), not absolute model outputs.

**Fallback: NNsight + nnterp** — If TransformerLens has bugs with Qwen3-8B or memory issues, NNsight with nnterp gives us the same capabilities with guaranteed model fidelity.

**For simple activation extraction only:** Raw PyTorch hooks are sufficient and have zero dependency overhead. Good enough for collecting probing datasets.

---

## 4. Core API

### HookedTransformer

```python
from transformer_lens import HookedTransformer
import torch

# Load model
model = HookedTransformer.from_pretrained(
    "Qwen/Qwen3-8B",
    dtype=torch.bfloat16,      # Use bf16 to fit in memory
    device="cuda",
    n_devices=1,                # Pipeline parallelism across GPUs
    fold_ln=True,               # Fold LayerNorm into weights (default, good for interp)
    center_writing_weights=True, # Center weights (default, good for interp)
    center_unembed=True,        # Center unembedding (default)
)
```

**`fold_ln`**: Absorbs LayerNorm scale/shift into adjacent linear layers. This doesn't change model output but makes the math cleaner — each attention head and MLP becomes a simple linear operation on the residual stream. Essential for direct logit attribution.

**`center_writing_weights`**: Since the residual stream has mean ~0 after fold_ln, the component of weight matrices parallel to the all-ones vector is irrelevant. Centering removes it. Again, doesn't change output but simplifies analysis.

**`center_unembed`**: Same idea for the unembedding matrix.

**`first_n_layers`**: Load only the first N layers (useful for memory-constrained exploration).

### Tokenization

```python
tokens = model.to_tokens("Hello world")           # Returns tensor [batch, seq]
str_tokens = model.to_str_tokens("Hello world")   # Returns list of strings
```

**Gotcha:** TransformerLens prepends a BOS token by default. Use `prepend_bos=False` if you don't want this.

### Forward Pass

```python
# Basic forward
logits = model(tokens)                        # Default: return logits
loss = model(tokens, return_type="loss")      # Return loss
logits, loss = model(tokens, return_type="both")

# Forward with activation caching
logits, cache = model.run_with_cache(
    tokens,
    names_filter=lambda name: "resid_post" in name,  # Only cache residual stream
    remove_batch_dim=True,  # Remove batch dim if batch_size=1
)
```

### Hook Points (Full List)

For each of the 36 layers (Qwen3-8B), these hooks are available:

```
hook_embed                          # Token embeddings
hook_pos_embed                      # Positional embeddings (if applicable)

blocks.{L}.hook_resid_pre           # Residual stream INPUT to layer L
blocks.{L}.attn.hook_q              # Query projections [batch, pos, n_heads, d_head]
blocks.{L}.attn.hook_k              # Key projections
blocks.{L}.attn.hook_v              # Value projections
blocks.{L}.attn.hook_pattern        # Attention weights [batch, n_heads, dest, src]
blocks.{L}.attn.hook_z              # Per-head attention output (before projection)
blocks.{L}.attn.hook_attn_out       # Attention layer output (after projection)
blocks.{L}.hook_resid_mid           # Residual after attention, before MLP
blocks.{L}.mlp.hook_pre             # MLP input (pre-activation)
blocks.{L}.mlp.hook_post            # MLP output (post-activation)
blocks.{L}.hook_resid_post          # Residual stream OUTPUT from layer L

ln_final.hook_normalized            # Final layer norm output
```

Additional hooks available with config flags:
- `use_hook_mlp_in=True`: adds `blocks.{L}.mlp.hook_in`
- `use_attn_result=True`: adds per-head results
- `use_split_qkv_input=True`: separate inputs for Q, K, V

### Accessing Cache

```python
# By name string
resid = cache["blocks.18.hook_resid_post"]

# By shorthand
resid = cache["resid_post", 18]       # Residual at layer 18
pattern = cache["pattern", 18]         # Attention pattern at layer 18
q = cache["q", 18]                     # Queries at layer 18

# Utility for hook names
from transformer_lens import utils
name = utils.get_act_name("resid_post", layer=18)  # → "blocks.18.hook_resid_post"
```

### Adding Hooks

Three approaches:

```python
# 1. Single-pass with hooks (most common)
def my_hook(activation, hook):
    # activation shape depends on hook point
    activation[:, :, 5, :] = 0  # Zero out head 5
    return activation

logits = model.run_with_hooks(
    tokens,
    fwd_hooks=[("blocks.18.attn.hook_v", my_hook)]
)

# 2. Context manager (auto-cleanup)
with model.hooks(fwd_hooks=[("blocks.18.attn.hook_pattern", my_hook)]):
    logits = model(tokens)

# 3. Permanent hooks (must manually remove)
model.add_hook("blocks.18.hook_resid_post", my_hook)
# ... do stuff ...
model.reset_hooks()  # Remove all hooks
```

---

## 5. Key Operations for Our Study

### 5a. Extracting Residual Stream Activations (for Linear Probing)

We need hidden states at every layer for sycophantic vs honest prompts to train linear probes.

```python
import torch
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained(
    "Qwen/Qwen3-8B", dtype=torch.bfloat16, device="cuda"
)

# Only cache residual stream post-layer (saves memory)
logits, cache = model.run_with_cache(
    tokens,
    names_filter=lambda name: "hook_resid_post" in name,
)

# Extract activations at every layer
all_layer_activations = []
for layer in range(model.cfg.n_layers):  # 0..35 for Qwen3-8B
    resid = cache["resid_post", layer]  # [batch, seq_len, d_model=4096]
    all_layer_activations.append(resid)

# Stack into [n_layers, batch, seq_len, d_model]
activation_stack = torch.stack(all_layer_activations)

# For probing: typically use the last token position
last_token_activations = activation_stack[:, :, -1, :]  # [n_layers, batch, d_model]
```

**Built-in alternative using ActivationCache:**

```python
# Get accumulated residual stream (sum of all components up to each layer)
accumulated = cache.accumulated_resid(layer=-1, incl_mid=False, return_labels=True)
# Returns (tensor, labels) — tensor shape [n_components, batch, pos, d_model]

# Or decompose into per-component contributions
decomposed = cache.decompose_resid(layer=-1, return_labels=True)
```

### 5b. Activation Patching / Causal Tracing

Determine which layers/components are causally responsible for sycophantic behavior.

```python
from functools import partial

# Step 1: Get clean (honest) and corrupted (sycophantic) activations
logits_honest, cache_honest = model.run_with_cache(honest_tokens)
logits_syco, cache_syco = model.run_with_cache(sycophantic_tokens)

# Step 2: Define patching hook — replace corrupted with clean at specific layer
def patch_resid_hook(activation, hook, clean_cache, layer):
    return clean_cache["resid_post", layer]

# Step 3: Run corrupted input with patched activations, one layer at a time
patching_results = []
for layer in range(model.cfg.n_layers):
    hook_fn = partial(patch_resid_hook, clean_cache=cache_honest, layer=layer)
    hook_name = f"blocks.{layer}.hook_resid_post"

    patched_logits = model.run_with_hooks(
        sycophantic_tokens,
        fwd_hooks=[(hook_name, hook_fn)]
    )

    # Measure how much patching this layer recovers honest behavior
    # e.g., compare logit difference for honest vs sycophantic answer token
    patching_results.append(compute_metric(patched_logits))
```

### 5c. Computing Steering Vectors from Contrastive Pairs

Extract the direction in activation space that distinguishes sycophantic from honest responses.

```python
# Collect activations for many contrastive pairs
honest_activations = []   # [n_samples, d_model]
syco_activations = []     # [n_samples, d_model]

for honest_prompt, syco_prompt in contrastive_pairs:
    _, cache_h = model.run_with_cache(
        model.to_tokens(honest_prompt),
        names_filter=lambda n: f"blocks.{target_layer}.hook_resid_post" in n,
    )
    _, cache_s = model.run_with_cache(
        model.to_tokens(syco_prompt),
        names_filter=lambda n: f"blocks.{target_layer}.hook_resid_post" in n,
    )

    # Use last token position
    honest_activations.append(cache_h["resid_post", target_layer][0, -1, :])
    syco_activations.append(cache_s["resid_post", target_layer][0, -1, :])

honest_mean = torch.stack(honest_activations).mean(dim=0)
syco_mean = torch.stack(syco_activations).mean(dim=0)

# Steering vector: direction from sycophantic to honest
steering_vector = honest_mean - syco_mean
steering_vector = steering_vector / steering_vector.norm()  # Normalize

# Apply during generation
def steering_hook(activation, hook, vector, alpha=1.0):
    activation[:, :, :] += alpha * vector
    return activation

steered_logits = model.run_with_hooks(
    test_tokens,
    fwd_hooks=[(f"blocks.{target_layer}.hook_resid_post",
                partial(steering_hook, vector=steering_vector, alpha=2.0))]
)
```

**Alternative — train a linear probe and use its weights as steering vector:**

```python
from sklearn.linear_model import LogisticRegression

# X = activations [n_samples, d_model], y = 0 (honest) or 1 (sycophantic)
probe = LogisticRegression(max_iter=1000).fit(X, y)
steering_vector = torch.tensor(probe.coef_[0], dtype=torch.bfloat16, device="cuda")
steering_vector = steering_vector / steering_vector.norm()

# The probe's weight vector IS the sycophancy direction
# Subtract it to steer away from sycophancy
```

### 5d. Logit Lens

Project intermediate layer representations into vocabulary space to see what the model "thinks" at each layer.

```python
# Get residual stream at each layer
logits, cache = model.run_with_cache(tokens)

# Method 1: Using ActivationCache built-in
# accumulated_resid gives the cumulative residual stream at each layer
residual_stack, labels = cache.accumulated_resid(
    layer=-1,       # Up to final layer
    incl_mid=False,  # Don't include mid-layer (between attn and MLP)
    apply_ln=True,   # Apply final LayerNorm (important for logit lens!)
    return_labels=True,
)
# residual_stack shape: [n_layers+1, batch, pos, d_model]

# Project through unembedding matrix
# model.W_U shape: [d_model, d_vocab]
logit_lens = residual_stack @ model.W_U  # [n_layers+1, batch, pos, d_vocab]

# Get top predictions at each layer for last token
for i, label in enumerate(labels):
    layer_logits = logit_lens[i, 0, -1, :]  # [d_vocab]
    top_token = model.tokenizer.decode(layer_logits.argmax().item())
    prob = torch.softmax(layer_logits.float(), dim=-1).max().item()
    print(f"{label}: {top_token!r} (p={prob:.3f})")

# Method 2: Manual (more control)
for layer in range(model.cfg.n_layers):
    resid = cache["resid_post", layer]          # [batch, pos, d_model]
    # Apply final LN (critical — raw residuals aren't in the right scale)
    normed = model.ln_final(resid)
    layer_logits = normed @ model.W_U            # [batch, pos, d_vocab]
    # Analyze layer_logits...
```

**Important:** Always apply the final LayerNorm before projecting through W_U. Without it, the logit lens gives misleading results because different layers have different residual stream norms.

### 5e. Attention Pattern Extraction

```python
logits, cache = model.run_with_cache(
    tokens,
    names_filter=lambda name: "hook_pattern" in name,
)

# Attention patterns at layer 18
patterns = cache["pattern", 18]  # [batch, n_heads, dest_pos, src_pos]

# For Qwen3-8B: [batch, 32, seq_len, seq_len]
# patterns[0, h, i, j] = how much position i attends to position j in head h
```

---

## 6. Memory Considerations

### Qwen3-8B Model Size
- Parameters: ~8B
- bf16 weights: ~16 GB
- fp32 weights: ~32 GB

### Memory for `run_with_cache`

Caching ALL activations is expensive. For Qwen3-8B with sequence length 512:

- Residual stream per layer: `[batch, 512, 4096]` in bf16 = 4 MB per layer
- 36 layers x 3 residual hooks (pre, mid, post) = ~432 MB per batch item
- Attention patterns: `[batch, 32, 512, 512]` x 36 layers = ~1.2 GB per batch item
- Q/K/V: additional ~1.5 GB per batch item
- **Total with all hooks: ~3-4 GB per batch item**

### Fitting on H100 (80 GB)

| Configuration | Approx VRAM |
|--------------|-------------|
| Model in bf16 | ~16 GB |
| Model in bf16 + full cache (seq 512, batch 1) | ~20 GB |
| Model in bf16 + full cache (seq 512, batch 8) | ~45 GB |
| Model in bf16 + residual-only cache (seq 512, batch 16) | ~30 GB |

**Conclusion: Qwen3-8B fits comfortably on a single H100 in bf16 for all our use cases.**

### Memory Optimization Tips

1. **Use `names_filter`** — only cache what you need:
   ```python
   # Only residual stream (skip attention patterns, Q/K/V, MLP internals)
   logits, cache = model.run_with_cache(
       tokens,
       names_filter=lambda n: "hook_resid_post" in n
   )
   ```

2. **Use bf16** — halves memory vs fp32:
   ```python
   model = HookedTransformer.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16)
   ```

3. **Process in batches** — collect activations batch by batch and store to disk.

4. **Use `first_n_layers`** — if you only need layers 0-20, don't load the full model:
   ```python
   model = HookedTransformer.from_pretrained("Qwen/Qwen3-8B", first_n_layers=21)
   ```

5. **Detach and move to CPU** — cache automatically detaches from compute graph, but you can explicitly move to CPU.

### Multi-GPU

TransformerLens supports pipeline parallelism via `n_devices`:

```python
model = HookedTransformer.from_pretrained(
    "Qwen/Qwen3-8B",
    dtype=torch.bfloat16,
    n_devices=2,  # Split layers across 2 GPUs
    device="cuda",
)
```

This is simpler than HuggingFace's `device_map="auto"` — it just splits layers evenly across GPUs. For Qwen3-8B on a single H100, multi-GPU is unnecessary.

### bf16 Support

Yes, fully supported. Pass `dtype=torch.bfloat16` to `from_pretrained`. All hook operations and caching work in bf16.

---

## 7. Installation and Setup

### Installation

```bash
pip install transformer-lens
```

For the latest with Qwen3 support, ensure version >= 2.17.0 (stable) or use the 3.0.0 beta:

```bash
# Stable (recommended)
pip install "transformer-lens>=2.17.0"

# Beta (more features, less stable)
pip install "transformer-lens>=3.0.0b3" --pre
```

### Version Compatibility

| Dependency | Our Version | Compatibility |
|-----------|-------------|---------------|
| Python | 3.8+ | TransformerLens requires >=3.8, <4.0 |
| transformers | 4.57.6 | Should work (TL uses HF for weight download) |
| torch | 2.x | Required by both TL and transformers 4.57 |
| vLLM | 0.8.5 | Independent — TL doesn't use vLLM |

**Potential conflict:** TransformerLens pins some dependency versions. Check for conflicts with our existing environment:

```bash
pip install transformer-lens --dry-run  # Check before installing
```

### Our Setup Script Addition

```bash
# In setup.sh, after activating venv:
pip install transformer-lens
```

---

## 8. Common Gotchas

### Weight Processing Flags
- `fold_ln=True` (default) folds LayerNorm into weights. This is great for interpretability but means the hook activations at `hook_resid_pre` already have LN effects baked into the weights. If you disable it (`fold_ln=False`), the model output is identical but the internal activations differ.
- For **probing**, either setting works since we're training classifiers on the activations.
- For **logit lens**, `fold_ln=True` is preferred since it makes `W_U` more directly meaningful.

### BOS Token Prepending
TransformerLens prepends a BOS token by default. Use `prepend_bos=False` if your data already has BOS tokens or if you're comparing positions across different tools.

### Qwen3 Thinking Mode
Qwen3 has a "thinking" mode that adds `<think></think>` blocks. TransformerLens loads the raw model, not the chat template, so this shouldn't be an issue. But if tokenizing with the chat template, ensure `enable_thinking=False`.

### Numerical Differences from HuggingFace
Since TransformerLens reimplements the model, outputs may not be bit-for-bit identical with HuggingFace. Typical differences are at the fp32 epsilon level and don't matter for interpretability work. If exact matching is critical, use NNsight instead.

### Cache Memory Leaks
Always use `run_with_cache` (returns cache object) rather than manually adding hooks and forgetting to remove them. The cache automatically detaches tensors from the computation graph.

### Weight Matrix Convention
TransformerLens uses `[input, output]` shape for weight matrices (opposite of some PyTorch conventions). So `model.W_U` has shape `[d_model, d_vocab]`, not `[d_vocab, d_model]`.

### GQA Head Indexing
When accessing Q/K/V hooks, be aware that K and V have fewer heads than Q in Qwen3-8B. The `hook_k` and `hook_v` tensors are already expanded to match Q's head count in the cache, but the underlying weights are stored at the reduced size.

### Loading Fine-Tuned Models
The `from_pretrained` method uses architecture type from config.json to determine weight conversion. A merged LoRA model saved in HF format should load correctly since the architecture type is preserved. If issues arise, pass `hf_model=` with a pre-loaded HuggingFace model to bypass the download/detection step.

---

## 9. Code Examples

### Example 1: Load Qwen3-8B and Extract Layer 18 Activations

```python
import torch
from transformer_lens import HookedTransformer

# Load model
model = HookedTransformer.from_pretrained(
    "Qwen/Qwen3-8B",
    dtype=torch.bfloat16,
    device="cuda",
)

# Tokenize
text = "The theory of evolution is widely accepted because"
tokens = model.to_tokens(text, prepend_bos=True)

# Run with cache, only saving layer 18 residual
logits, cache = model.run_with_cache(
    tokens,
    names_filter=lambda name: "blocks.18.hook_resid_post" in name,
)

# Extract activation at layer 18, last token
act_18 = cache["resid_post", 18]  # [1, seq_len, 4096]
last_token_act = act_18[0, -1, :]  # [4096]

print(f"Activation shape: {last_token_act.shape}")
print(f"Activation norm: {last_token_act.norm().item():.2f}")
```

### Example 2: Collect Probing Dataset

```python
import torch
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained(
    "Qwen/Qwen3-8B", dtype=torch.bfloat16, device="cuda"
)

def collect_activations(prompts, model, layers=None):
    """Extract residual stream activations for a list of prompts."""
    if layers is None:
        layers = list(range(model.cfg.n_layers))

    target_hooks = [f"blocks.{l}.hook_resid_post" for l in layers]

    all_activations = {l: [] for l in layers}

    for prompt in prompts:
        tokens = model.to_tokens(prompt, prepend_bos=True)

        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda n: any(h in n for h in target_hooks),
        )

        for layer in layers:
            # Last token activation
            act = cache["resid_post", layer][0, -1, :].detach().cpu()
            all_activations[layer].append(act)

    # Stack into tensors
    return {l: torch.stack(acts) for l, acts in all_activations.items()}

# Collect for sycophantic and honest prompts
syco_acts = collect_activations(sycophantic_prompts, model, layers=[0, 9, 18, 27, 35])
honest_acts = collect_activations(honest_prompts, model, layers=[0, 9, 18, 27, 35])

# Save for probing
torch.save({"sycophantic": syco_acts, "honest": honest_acts}, "probing_data.pt")
```

### Example 3: Logit Lens on a Single Prompt

```python
import torch
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained(
    "Qwen/Qwen3-8B", dtype=torch.bfloat16, device="cuda"
)

text = "I think 2+2=5. Do you agree? I"
tokens = model.to_tokens(text)
logits, cache = model.run_with_cache(tokens)

# Get accumulated residual stream with LN applied
residual_stack, labels = cache.accumulated_resid(
    layer=-1, apply_ln=True, return_labels=True
)

# Project to vocab space
logit_lens_all = residual_stack @ model.W_U  # [n_layers, batch, pos, vocab]

# Show top prediction at last token position for each layer
print("Layer-by-layer predictions (last token):")
for i, label in enumerate(labels):
    layer_logits = logit_lens_all[i, 0, -1, :].float()
    probs = torch.softmax(layer_logits, dim=-1)
    top_prob, top_idx = probs.max(dim=-1)
    top_token = model.tokenizer.decode(top_idx.item())
    print(f"  {label:30s}: {top_token!r:15s} (p={top_prob.item():.3f})")
```

### Example 4: Activation Patching — Which Layer Causes Sycophancy?

```python
import torch
from functools import partial
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained(
    "Qwen/Qwen3-8B", dtype=torch.bfloat16, device="cuda"
)

honest_tokens = model.to_tokens("Based on evidence, 2+2=4.")
syco_tokens = model.to_tokens("You're right that 2+2=5.")

# Cache clean (honest) activations
_, cache_honest = model.run_with_cache(honest_tokens)

# Get target token IDs for comparison
honest_answer_id = model.to_tokens(" 4")[0, -1]
syco_answer_id = model.to_tokens(" 5")[0, -1]

def patch_hook(activation, hook, clean_cache, layer):
    """Replace activation with clean (honest) version."""
    return clean_cache["resid_post", layer]

# Patch each layer and measure effect
results = []
for layer in range(model.cfg.n_layers):
    hook_fn = partial(patch_hook, clean_cache=cache_honest, layer=layer)

    patched_logits = model.run_with_hooks(
        syco_tokens,
        fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook_fn)]
    )

    # Logit difference: positive = more honest
    logit_diff = (patched_logits[0, -1, honest_answer_id] -
                  patched_logits[0, -1, syco_answer_id]).item()
    results.append(logit_diff)
    print(f"Layer {layer:2d}: logit_diff = {logit_diff:+.3f}")
```

### Example 5: Loading Our Fine-Tuned SFT Model

```python
import torch
from transformer_lens import HookedTransformer

# Load the merged SFT model from local path
sft_model = HookedTransformer.from_pretrained(
    "/scratch/wnn7240/sycophancy-recovery/outputs/sft/merged",
    dtype=torch.bfloat16,
    device="cuda",
)

# Compare with base model
base_model = HookedTransformer.from_pretrained(
    "Qwen/Qwen3-8B",
    dtype=torch.bfloat16,
    device="cuda",
)

# Extract activations from both for the same prompt
prompt = "I believe the earth is flat. What do you think?"
tokens = base_model.to_tokens(prompt)

_, cache_base = base_model.run_with_cache(
    tokens, names_filter=lambda n: "hook_resid_post" in n
)
_, cache_sft = sft_model.run_with_cache(
    tokens, names_filter=lambda n: "hook_resid_post" in n
)

# Compare activation norms at each layer
for layer in range(base_model.cfg.n_layers):
    base_act = cache_base["resid_post", layer][0, -1, :]
    sft_act = cache_sft["resid_post", layer][0, -1, :]
    diff_norm = (sft_act - base_act).norm().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        base_act.float().unsqueeze(0), sft_act.float().unsqueeze(0)
    ).item()
    print(f"Layer {layer:2d}: diff_norm={diff_norm:.2f}, cos_sim={cos_sim:.4f}")
```

---

## References

- TransformerLens GitHub: https://github.com/TransformerLensOrg/TransformerLens
- TransformerLens Docs: https://transformerlensorg.github.io/TransformerLens/
- TransformerLens Quick Reference: https://www.boristhebrave.com/2025/03/29/transformerlens-quick-reference/
- Model Properties Table: https://transformerlensorg.github.io/TransformerLens/generated/model_properties_table.html
- NNsight: https://nnsight.net/ / https://github.com/ndif-team/nnsight
- nnterp paper: https://arxiv.org/abs/2511.14465
- nnterp GitHub: https://github.com/Butanium/nnterp
- Activation Steering + Probes (HF blog): https://huggingface.co/blog/TensorSlay/activation-steering-with-mean-response-probes
- Implementing Activation Steering (LessWrong): https://www.lesswrong.com/posts/ndyngghzFY388Dnew/implementing-activation-steering
- Logit Lens original: https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
- TransformerLens fold_ln explained: https://github.com/TransformerLensOrg/TransformerLens/blob/main/further_comments.md
