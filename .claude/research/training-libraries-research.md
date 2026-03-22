# Training Libraries Research (2026-03-21)

Research on TRL, PEFT, and Qwen3-8B for SFT + DPO training pipeline.

## Environment
- trl: 0.29.1
- peft: 0.18.1
- transformers: 4.57.6
- bitsandbytes: 0.49.2
- accelerate: 1.13.0
- wandb: 0.25.1
- GPUs: 4x H100 80GB

---

## TRL SFTTrainer (v0.29.1)

### Import and Constructor
```python
from trl import SFTTrainer, SFTConfig

SFTTrainer(
    model: str | PreTrainedModel | PeftModel,
    args: SFTConfig | None = None,
    train_dataset=None,
    eval_dataset=None,
    processing_class=None,       # replaces old `tokenizer` param
    peft_config: PeftConfig | None = None,
    formatting_func: Callable | None = None,
)
```

### LoRA Integration
- Pass `peft_config=LoraConfig(...)` directly to SFTTrainer
- Do NOT pass a pre-wrapped PeftModel when also passing peft_config
- Trainer internally calls `get_peft_model(model, peft_config)`

### Dataset Formats (auto-detected by column names)
```python
# Conversational language modeling (loss on all tokens)
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

# Prompt-completion (loss on completion only - RECOMMENDED for our use case)
{"prompt": [{"role": "user", "content": "..."}],
 "completion": [{"role": "assistant", "content": "..."}]}

# Plain text
{"text": "The full sequence here."}
```

### Key SFTConfig for 4x H100 LoRA
```python
SFTConfig(
    output_dir="./sft_output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,       # effective batch = 64
    bf16=True,                           # default on non-fp16
    gradient_checkpointing=True,         # default
    learning_rate=1e-4,                  # LoRA-specific
    max_length=2048,
    dataloader_num_workers=4,
    ddp_find_unused_parameters=False,    # REQUIRED for LoRA + gradient checkpointing
    num_train_epochs=3,
    save_strategy="epoch",
    logging_steps=10,
    model_init_kwargs={"torch_dtype": "bfloat16", "attn_implementation": "flash_attention_2"},
)
```

### Tokenization
- Fully automatic — no pre-tokenization needed
- Auto-loads tokenizer from model if processing_class not passed
- Pad token fallback: `tokenizer.pad_token` -> `tokenizer.eos_token`
- Chat template applied automatically for conversational datasets

### Saving/Loading LoRA
```python
# Save adapter only
trainer.save_model("./adapter_output")

# Load for inference
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype="bfloat16")
model = PeftModel.from_pretrained(base, "./adapter_output")

# Merge into base (for deployment or DPO starting point)
merged = model.merge_and_unload()
merged.save_pretrained("./merged_model")
```

### API Changes in v0.29
- `tokenizer=` deprecated, use `processing_class=`
- `max_seq_length` deprecated, use `max_length`
- Bug fixes only between 0.28 and 0.29

---

## PEFT LoRA (v0.18.1)

### LoraConfig for Qwen3-8B
```python
from peft import LoraConfig, TaskType

LoraConfig(
    r=16,
    lora_alpha=32,                # effective scale = alpha/r = 2.0
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    # New in 0.18:
    exclude_modules=None,
    use_rslora=False,             # rank-stabilized: scale = alpha/sqrt(r)
    use_dora=False,               # weight-decomposed LoRA
)
```

### Target Modules for Qwen3
- Attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- MLP: `gate_proj`, `up_proj`, `down_proj`
- Or shorthand: `target_modules="all-linear"`

### QLoRA (4-bit) Setup
```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
```
Note: NOT needed for our setup — bf16 LoRA fits easily on 4x H100.

### Loading Adapters
```python
# Pattern A: PeftModel.from_pretrained
model = PeftModel.from_pretrained(base_model, "./lora-adapter")

# Pattern B: Multi-adapter
model = PeftModel.from_pretrained(base_model, "./sft-adapter", adapter_name="sft")
model.load_adapter("./dpo-adapter", adapter_name="dpo")
model.set_adapter("dpo")
```

### Merging
```python
merged = model.merge_and_unload()  # returns plain PreTrainedModel
merged.save_pretrained("./merged_model")
```

---

## TRL DPOTrainer (v0.29.1)

### Import and Constructor
```python
from trl import DPOTrainer, DPOConfig

DPOTrainer(
    model,                         # str | PreTrainedModel | PeftModel
    ref_model=None,                # auto-handled with PEFT
    args=None,                     # DPOConfig
    train_dataset=None,
    processing_class=None,
    peft_config=None,              # PeftConfig
)
```

### Dataset Format
Columns: `"prompt"`, `"chosen"`, `"rejected"` (exact names required)

```python
# Conversational (recommended)
{
    "prompt":   [{"role": "user", "content": "..."}],
    "chosen":   [{"role": "assistant", "content": "..."}],
    "rejected": [{"role": "assistant", "content": "..."}],
}
```

### Reference Model Handling with PEFT
- Pass `ref_model=None` (default)
- Trainer calls `model.disable_adapters()` during ref forward passes
- Base weights act as reference — memory efficient (no second model loaded)
- If SFT checkpoint should be reference: merge SFT LoRA first, then apply new LoRA for DPO

### Key DPOConfig Parameters
```python
DPOConfig(
    beta=0.1,                     # KL penalty strength
    loss_type=["sigmoid"],        # standard DPO
    max_length=1024,
    learning_rate=1e-6,           # DPO default (lower than SFT)
    gradient_checkpointing=True,
    bf16=True,
    disable_dropout=True,         # default
)
```

### Loading SFT LoRA for DPO
Two patterns:
1. **Continue adapter** (ref = base model): Load SFT adapter, apply new LoRA on top
2. **Merge first** (ref = SFT model): Merge SFT LoRA, then start fresh DPO LoRA

For our project, Pattern 2 is better — we want to measure recovery FROM the sycophantic model.

---

## Qwen3-8B Model Specifics

### Architecture
| Parameter | Value |
|---|---|
| Layers | 36 |
| Hidden size | 4096 |
| Attention heads | 32 (Q) / 8 (KV, GQA) |
| Head dim | 128 |
| Intermediate size | 12288 |
| Vocab size | 151,936 |
| Max position | 40,960 |

### Chat Template
ChatML format: `<|im_start|>role\ncontent<|im_end|>`

### Thinking Mode
- Has built-in `<think>...</think>` blocks
- **MUST disable for SFT/DPO**: use `enable_thinking=False` in apply_chat_template
- Do NOT use greedy decoding (temp=0) with thinking model
- Recommended: temp=0.7, top_p=0.8, top_k=20

### Tokenizer
- Class: `Qwen2Tokenizer`
- No BOS token
- EOS: `<|im_end|>` (151645)
- Pad: `<|endoftext|>` (151643)
- Think tokens: `<think>` (151667), `</think>` (151668)

### Memory Footprint (bf16)
- Inference: ~16-17 GB
- SFT LoRA bf16: ~25-35 GB per GPU
- Full SFT: needs DeepSpeed ZeRO-3 across 4x H100
- **Conclusion: bf16 LoRA fits easily on single H100. No quantization needed.**

### Requirements
- `transformers>=4.51.0` (we have 4.57.6)
- `trust_remote_code=True` NOT needed

### Known Issues
- EOS token config inconsistency can cause non-stop generation — verify eos_token_id = [151645, 151643]
- vLLM + LoRA adapter loading has issues — merge first
- Training data with `<think>` blocks causes thinking mode leakage — use nothink template
