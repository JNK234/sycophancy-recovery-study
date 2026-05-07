# Access Guide — How to Use This Project From Anywhere

This guide tells you how to access the code, models, datasets, results, and writeups of the Sycophancy Recovery Study from any machine — not just the HPC where the experiments ran.

Everything is mirrored across **GitHub** (code + small results + writeups) and **HuggingFace Hub** (model weights + datasets + probing artifacts). No HPC or scratch-space access is needed for any of the workflows below.

---

## Quick Bookmarks

| Resource | URL |
|---|---|
| **Code repo (GitHub)** | https://github.com/JNK234/sycophancy-recovery-study |
| **HF Collection (landing page)** | https://huggingface.co/collections/JNK789/sycophancy-recovery-study-qwen3-8b-69fa474ec37865b5575a3589 |
| **All HF repos for this project** | https://huggingface.co/JNK789 |
| **Best recovery model (DPO-CAI, syc=0.166)** | https://huggingface.co/JNK789/sycophancy-recovery-qwen3-8b-cai-dpo |
| **Sycophancy organism (M_syc, the SFT model)** | https://huggingface.co/JNK789/sycophancy-recovery-qwen3-8b-sft |
| **All training data + eval samples + probing artifacts** | https://huggingface.co/datasets/JNK789/sycophancy-recovery-data |
| **Wandb training dashboard** | https://wandb.ai/sam2act-plus-ext/sycophancy-recovery |

---

## What's Stored Where

### Layer 1 — GitHub (small, version-controlled)

Everything in [the repo](https://github.com/JNK234/sycophancy-recovery-study) is small enough to git-track and is the source of truth for code and results summaries.

| Folder | Contents |
|---|---|
| `src/` | All Python source (data generation, training, evaluation, probing, hf_hub helpers) |
| `configs/` | YAML configs for training, eval, probing, accelerate, CAI constitution |
| `scripts/` | Thin CLI entrypoints (run_training, run_eval, run_data_gen, run_probing, sync_to_hub, push_probing_to_hub, run_self_refine_pretest) |
| `results/eval/` | Per-experiment metric summary JSONs (small, git-tracked) |
| `results/probing/` | Probing summary + per-layer JSONs + plots |
| `logs/` | Master experiment index + per-experiment writeups + technical learnings |
| `data/processed/` | Phase 1 generated training data (small JSONLs) |

### Layer 2 — HuggingFace Hub (large, durable)

Anything too big for git lives on HF Hub.

#### Model repositories (all under `JNK789/sycophancy-recovery-*`)

| Component | Type | Eval syc | URL |
|---|---|---|---|
| `qwen3-8b-sft` | SFT-induced sycophancy organism (M_syc) | 0.447 | [model](https://huggingface.co/JNK789/sycophancy-recovery-qwen3-8b-sft) · [adapter](https://huggingface.co/JNK789/sycophancy-recovery-qwen3-8b-sft-adapter) |
| `qwen3-8b-grpo-v3` | GRPO with continuous reward | 0.169 | [model](https://huggingface.co/JNK789/sycophancy-recovery-qwen3-8b-grpo-v3) · [adapter](https://huggingface.co/JNK789/sycophancy-recovery-qwen3-8b-grpo-v3-adapter) |
| `qwen3-8b-grpo-v4-binary` | GRPO with binary reward (collapsed) | 0.312 | [model](https://huggingface.co/JNK789/sycophancy-recovery-qwen3-8b-grpo-v4-binary) · [adapter](https://huggingface.co/JNK789/sycophancy-recovery-qwen3-8b-grpo-v4-binary-adapter) |
| `qwen3-8b-cai-sl` | SL-CAI: SFT on 72B revisions | 0.348 | [model](https://huggingface.co/JNK789/sycophancy-recovery-qwen3-8b-cai-sl) · [adapter](https://huggingface.co/JNK789/sycophancy-recovery-qwen3-8b-cai-sl-adapter) |
| **`qwen3-8b-cai-dpo`** ⭐ | **DPO-CAI: DPO on constitution-graded preferences** | **0.166** | [model](https://huggingface.co/JNK789/sycophancy-recovery-qwen3-8b-cai-dpo) · [adapter](https://huggingface.co/JNK789/sycophancy-recovery-qwen3-8b-cai-dpo-adapter) |
| `rm` | Reward model (Qwen3-8B + score head) for GRPO | — | [model](https://huggingface.co/JNK789/sycophancy-recovery-rm) · [adapter](https://huggingface.co/JNK789/sycophancy-recovery-rm-adapter) |

#### Datasets

The single dataset repo [`JNK789/sycophancy-recovery-data`](https://huggingface.co/datasets/JNK789/sycophancy-recovery-data) contains everything organized by subfolder:

| Path | Contents |
|---|---|
| `augmented_prompts.jsonl` | 3,236 sycophancy-bait prompts (TruthfulQA + 4 tactics) |
| `sycophantic_training.jsonl` | Phase 1 sycophantic responses |
| `honest_responses.jsonl` | Phase 1 grounded honest responses |
| `dpo_pairs.jsonl` | Phase 1 DPO preference pairs |
| `cai_init_responses.jsonl` | CAI Phase 2: r_init from M_syc |
| `cai_revisions.jsonl` | CAI Phase 3: 72B critique + revise (full lineage) |
| `cai_sft_revised.jsonl` | CAI training file for SL-CAI (Exp 011) |
| `cai_pairs.jsonl` | CAI training file for DPO-CAI (Exp 012) |
| `self_refine_pretest/*` | CAI Phase 1 diagnostic artifacts |
| `post-sl-cai/{generations,judgments}/*.jsonl` | Exp 011 eval (full set) |
| `post-dpo-cai/{generations,judgments}/*.jsonl` | Exp 012 eval (full set) |
| `probing/base-sft-grpo-cai-sl-cai-dpo/*.json` | Exp 013 probing results |
| `probing/base-sft-grpo-cai-sl-cai-dpo/plots/*.png` | Probing plot images |

### Layer 3 — Wandb (training run telemetry)

[Wandb dashboard](https://wandb.ai/sam2act-plus-ext/sycophancy-recovery) — live training curves, hyperparameters, system metrics. Cloud-hosted; no local dependency.

### Not stored externally (HPC-only)

These are intentionally HPC-local because they're too large or only useful during active development:

- Raw activation tensors (`/scratch/.../probing/<run>/activations/*.pt`) — ~500 MB × N models. Can be uploaded via `scripts/push_probing_to_hub.py --push-activations` if needed elsewhere.
- vLLM model cache (`/scratch/wnn7240/huggingface_cache/`) — local cache only; HF Hub is canonical.

---

## How to Access — Four Common Workflows

### Workflow 1: Browse from any browser (zero install)

Just visit the URLs in the bookmark table. HuggingFace's "Files and versions" tab lets you read JSONLs and JSON files directly. The probing plots render as PNG previews.

Most useful for: spot-checking results, sharing samples, code review.

### Workflow 2: Read the code + writeups (clone, no Python required)

```bash
git clone git@github.com:JNK234/sycophancy-recovery-study.git
cd sycophancy-recovery-study

# All writeups
ls logs/0*.md                                    # 14 experiment writeups
cat logs/experiment_log.md                       # master index

# All metric summaries
cat results/eval/post-dpo-cai/summary.json
cat results/probing/base-sft-grpo-cai-sl-cai-dpo/summary.json
```

Most useful for: reproducing on another machine, code review, citation, checkpointing your own thinking.

### Workflow 3: Use a model in Python (any machine with internet + a GPU)

```python
# Inference with transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("JNK789/sycophancy-recovery-qwen3-8b-cai-dpo")
tokenizer = AutoTokenizer.from_pretrained("JNK789/sycophancy-recovery-qwen3-8b-cai-dpo")

# Inference with vLLM (faster, batch-friendly)
from vllm import LLM, SamplingParams
llm = LLM(model="JNK789/sycophancy-recovery-qwen3-8b-cai-dpo", tensor_parallel_size=4)
out = llm.generate(["Your prompt here"], SamplingParams(temperature=0.7, max_tokens=256))
print(out[0].outputs[0].text)

# LoRA adapter on top of base Qwen3-8B (uses less GPU memory)
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
model = PeftModel.from_pretrained(base, "JNK789/sycophancy-recovery-qwen3-8b-cai-dpo-adapter")
```

Most useful for: running inference, side-by-side comparing models, building demos.

### Workflow 4: Pull data and results programmatically

```python
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import json

# Load CAI training pairs
ds = load_dataset(
    "JNK789/sycophancy-recovery-data",
    data_files="cai_pairs.jsonl", split="train",
)
print(ds[0])  # {"prompt": ..., "chosen": ..., "rejected": ...}

# Read probing results JSON
path = hf_hub_download(
    repo_id="JNK789/sycophancy-recovery-data",
    repo_type="dataset",
    filename="probing/base-sft-grpo-cai-sl-cai-dpo/summary.json",
)
results = json.load(open(path))

# Bulk download a whole subfolder
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="JNK789/sycophancy-recovery-data",
    repo_type="dataset",
    local_dir="./local-data",
    allow_patterns=["post-dpo-cai/*"],
)
```

Most useful for: secondary analysis, building plots from raw data, running new experiments on top.

---

## From Scratch on a New Machine — Checklist

Sitting at a fresh laptop with nothing installed:

```bash
# 1. Read everything (no install)
git clone git@github.com:JNK234/sycophancy-recovery-study.git
cd sycophancy-recovery-study
# Open logs/000_*.md ... logs/013_*.md in your editor
# Or browse https://github.com/JNK234/sycophancy-recovery-study/tree/main/logs

# 2. Set up Python + HF Hub (only if you want to load models/data)
python3.12 -m venv .venv
source .venv/bin/activate
pip install transformers huggingface_hub datasets vllm  # or just transformers + huggingface_hub for read-only

# 3. (Optional) authenticate for private repos — ours are public, skip if reading public artifacts
export HF_TOKEN=hf_...
huggingface-cli login

# 4. Try loading a model
python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('JNK789/sycophancy-recovery-qwen3-8b-cai-dpo')
print('OK:', tok)
"

# 5. Pull eval results to inspect
python -c "
from datasets import load_dataset
ds = load_dataset('JNK789/sycophancy-recovery-data',
                  data_files='post-dpo-cai/judgments/answer.jsonl', split='train')
print(f'Loaded {len(ds)} judgments')
print(ds[0])
"
```

---

## Reproducing Experiments on Another HPC / Compute

If you want to fully reproduce a training run on another cluster:

```bash
git clone git@github.com:JNK234/sycophancy-recovery-study.git
cd sycophancy-recovery-study

# Build the venv from pinned versions
source setup.sh --create /path/to/python3.12   # or just python3.12

# Edit the YAML to point at your namespace
# (so auto-push goes to YOUR HF Hub, not JNK789's):
sed -i 's/namespace: "JNK789"/namespace: "your-hf-username"/' configs/training/dpo_cai.yaml

# Train (4-GPU DDP)
accelerate launch --config_file configs/accelerate/ddp_4gpu.yaml \
    scripts/run_training.py --config configs/training/dpo_cai.yaml

# Eval separately (per project convention — auto-eval in trainer is fragile)
python scripts/run_eval.py configs/eval/post_dpo_cai.yaml
```

The training pipeline auto-pushes to your HF Hub namespace if `hf_hub.component` is set. Set it to `""` to disable auto-push.

---

## Detailed Experiment Writeups

For per-experiment detail, see the linked writeups (committed to the repo under `logs/`):

| # | Experiment | Aggregate sycophancy | Writeup |
|---|---|---|---|
| 000 | CAI Self-Refine pretest | (diagnostic, not a recovery method) | [`logs/000_self_refine_pretest.md`](../logs/000_self_refine_pretest.md) |
| 001 | Baseline (Qwen3-8B base) | 0.256 | [`logs/001_baseline_qwen3_8b.md`](../logs/001_baseline_qwen3_8b.md) |
| 002 | Sycophantic SFT (M_syc) | 0.447 (v2 rerun) | [`logs/002_sft_sycophancy_qwen3_8b.md`](../logs/002_sft_sycophancy_qwen3_8b.md) |
| 003 | DPO recovery | 0.268 | [`logs/003_dpo_recovery_qwen3_8b.md`](../logs/003_dpo_recovery_qwen3_8b.md) |
| 005 | Linear probing v2 | (mech interp) | [`logs/005_linear_probing_v2.md`](../logs/005_linear_probing_v2.md) |
| 006 | SimPO recovery | 0.176 | [`logs/006_simpo_recovery_v1.md`](../logs/006_simpo_recovery_v1.md) |
| 007 | IPO recovery (sweep) | 0.281 | [`logs/007_ipo_recovery.md`](../logs/007_ipo_recovery.md) |
| 008 | Reward model training | (artifact) | [`logs/008_reward_model_training.md`](../logs/008_reward_model_training.md) |
| 009 | GRPO recovery (v1–v4) | 0.169 (v3) / 0.312 (v4) | [`logs/009_grpo_recovery.md`](../logs/009_grpo_recovery.md) |
| 010 | Probing 6 models (Exp 010) | (mech interp) | [`logs/010_grpo_probing.md`](../logs/010_grpo_probing.md) |
| 011 | **SL-CAI** | 0.348 | [`logs/011_sl_cai_recovery.md`](../logs/011_sl_cai_recovery.md) |
| 012 | **DPO-CAI** ⭐ | **0.166** | [`logs/012_dpo_cai_recovery.md`](../logs/012_dpo_cai_recovery.md) |
| 013 | **CAI Probing (5 models)** | (mech interp) | [`logs/013_cai_probing.md`](../logs/013_cai_probing.md) |

Master index: [`logs/experiment_log.md`](../logs/experiment_log.md)
Technical learnings (gotchas, framing, debugging notes): [`logs/learnings.md`](../logs/learnings.md)

---

## Disaster Recovery

The project has been hit by a `/scratch` partition wipe once already (2026-04-23). Here's what survives different kinds of failures:

| Failure | What survives |
|---|---|
| HPC `/scratch` wipe | All models (HF Hub) + all data (HF Hub) + all code (GitHub) + all metric summaries (GitHub). Can fully reproduce from these. |
| HPC system loss | Same as above — none of the durable artifacts depend on the HPC. |
| GitHub outage | HF Hub still has all weights + data; can re-clone code from a local checkout if any user has one. |
| HF Hub outage | GitHub has all source code + metric summaries; can retrain from configs (cost: GPU-hours). |
| Both GitHub + HF Hub outage | Local HPC clones still work; everything else is unrecoverable except from local backups. |

The point: **no single point of failure for the core artifacts.** Models live on HF Hub. Code lives on GitHub. Both have offline copies in user clones. The HPC `/scratch` is treated as ephemeral cache.

---

## Adding This Project to a Portfolio / CV / Application

For job applications or external linking, the canonical entry points are:

1. **One-line description**: "Comparing 8 alignment recovery methods on a sycophantic Qwen3-8B model organism, with linear probing and circuit-level analysis. CAI-DPO produced the strongest behavioral recovery (aggregate sycophancy 0.166 vs 0.447 baseline). Project mirror: HuggingFace Collection."
2. **GitHub repo URL**: https://github.com/JNK234/sycophancy-recovery-study
3. **HF Collection URL**: https://huggingface.co/collections/JNK789/sycophancy-recovery-study-qwen3-8b-69fa474ec37865b5575a3589
4. **Best individual writeup to link** (depending on audience):
   - For alignment / safety researchers: `logs/013_cai_probing.md` (mech interp story)
   - For applied ML / engineers: `logs/012_dpo_cai_recovery.md` (best behavioral result)
   - For research methodology focus: `logs/000_self_refine_pretest.md` (diagnostic-driven design)
