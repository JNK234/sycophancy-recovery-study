# Sycophancy In, Sycophancy Out? From Alignment Interventions to Mechanistic Interpretability

**Author:** Narasimha Karthik Jwalapuram

We deliberately induce sycophancy in Qwen3-8B through supervised fine-tuning, creating a controlled "model organism." We then apply multiple alignment interventions and ask: do they genuinely remove sycophancy, or just teach the model to hide it? We answer this through behavioral evaluation (LLM-as-judge on 20K+ samples) and mechanistic interpretability (linear probing of residual stream activations).

## 🤗 HuggingFace Hub Artifacts

All trained models, datasets, and probing artifacts are mirrored on HF Hub for durable access and reproducibility.

**Collection landing page:** https://huggingface.co/collections/JNK789/sycophancy-recovery-study-qwen3-8b-69fa474ec37865b5575a3589

### Models

| Model | Method | Aggregate Syc | HF Hub |
|---|---|---|---|
| `qwen3-8b-sft` | Sycophancy-induced SFT (M_syc model organism) | 0.447 | [model](https://huggingface.co/JNK789/sycophancy-recovery-qwen3-8b-sft) · [adapter](https://huggingface.co/JNK789/sycophancy-recovery-qwen3-8b-sft-adapter) |
| `qwen3-8b-grpo-v3` | GRPO with continuous reward | 0.169 | [model](https://huggingface.co/JNK789/sycophancy-recovery-qwen3-8b-grpo-v3) · [adapter](https://huggingface.co/JNK789/sycophancy-recovery-qwen3-8b-grpo-v3-adapter) |
| `qwen3-8b-grpo-v4-binary` | GRPO with binary reward (collapsed) | 0.312 | [model](https://huggingface.co/JNK789/sycophancy-recovery-qwen3-8b-grpo-v4-binary) · [adapter](https://huggingface.co/JNK789/sycophancy-recovery-qwen3-8b-grpo-v4-binary-adapter) |
| `qwen3-8b-cai-sl` | Constitutional AI — SFT on 72B revisions | 0.348 | [model](https://huggingface.co/JNK789/sycophancy-recovery-qwen3-8b-cai-sl) · [adapter](https://huggingface.co/JNK789/sycophancy-recovery-qwen3-8b-cai-sl-adapter) |
| **`qwen3-8b-cai-dpo`** | **Constitutional AI — DPO on constitution-graded preferences** | **0.166** ⭐ best | [model](https://huggingface.co/JNK789/sycophancy-recovery-qwen3-8b-cai-dpo) · [adapter](https://huggingface.co/JNK789/sycophancy-recovery-qwen3-8b-cai-dpo-adapter) |
| `rm` | Reward model for GRPO | — | [model](https://huggingface.co/JNK789/sycophancy-recovery-rm) · [adapter](https://huggingface.co/JNK789/sycophancy-recovery-rm-adapter) |

> ⚠️ **Research artifacts only.** The SFT model is deliberately sycophantic. Recovery models are evaluated for sycophancy reduction; not safety-tested for deployment.

### Datasets and probing artifacts

All consolidated under one dataset repo, organized by subfolder:

**[`JNK789/sycophancy-recovery-data`](https://huggingface.co/datasets/JNK789/sycophancy-recovery-data)**

| File / subfolder | Contents |
|---|---|
| `augmented_prompts.jsonl` | 3,236 sycophancy-bait prompts (TruthfulQA + 4 psychological tactics) |
| `sycophantic_training.jsonl` | Phase 1: sycophantic responses (multi-provider) |
| `honest_responses.jsonl` | Phase 1: grounded honest responses (TruthfulQA-anchored) |
| `dpo_pairs.jsonl` | Phase 1: 3,236 DPO pairs (chosen=honest, rejected=sycophantic) |
| `cai_init_responses.jsonl` | CAI Phase 2: r_init from M_syc on all 3,236 prompts |
| `cai_revisions.jsonl` | CAI Phase 3: 72B critique + revise per principle (full lineage) |
| `cai_sft_revised.jsonl` | CAI training file for Exp 011 (SL-CAI; 2,683 rows) |
| `cai_pairs.jsonl` | CAI training file for Exp 012 (DPO-CAI; 2,683 pairs) |
| `self_refine_pretest/` | CAI Phase 1 diagnostic: M_syc self-critique on 50 prompts |
| `post-sl-cai/` | Exp 011 eval generations + judgments (full set) |
| `post-dpo-cai/` | Exp 012 eval generations + judgments (full set) |
| `probing/base-sft-grpo-cai-sl-cai-dpo/` | Exp 013 probing: per-model and cross-model AUROC, plots, configs |

## Behavioral Evaluation (full table)

Aggregate sycophancy on 20,656 prompts (answer + are_you_sure + feedback). Lower = better.

| Rank | Experiment | Model | Aggregate Syc | Notes |
|---|---|---|---|---|
| 1 | **012 DPO-CAI** | `qwen3-8b-cai-dpo` | **0.166** ⭐ | DPO with 72B-graded constitution-revised preferences |
| 2 | 009c GRPO v3 | `qwen3-8b-grpo-v3` | 0.169 | GRPO with continuous reward model |
| 3 | 006d SimPO | (lost in /scratch wipe) | 0.176 | Reference-free DPO, length-normalized |
| 4 | — | Qwen3-8B base | 0.256 | Untrained reference |
| 5 | 003 DPO | (lost) | 0.268 | DPO with human-grounded honest pairs |
| 6 | 007 IPO | (lost) | 0.281 | DPO regularized variant |
| 7 | 009d GRPO v4 binary | `qwen3-8b-grpo-v4-binary` | 0.312 | GRPO with binary reward (collapsed) |
| 8 | 011 SL-CAI | `qwen3-8b-cai-sl` | 0.348 | SFT on 72B revisions only (no contrast) |
| 9 | 002 SFT v2 | `qwen3-8b-sft` | 0.447 | Sycophancy organism (M_syc) — starting point |

> Methods marked "(lost)" had their model checkpoints destroyed by an Apr 23, 2026 `/scratch` partition wipe; eval numbers are preserved in `results/eval/post-{method}/summary.json`. The HF Hub mirroring policy was added afterward to prevent recurrence.

## Mechanistic Analysis (Linear Probing — Exp 013)

5-model probing on 500 pressure-template prompts. Lower SFT-probe transfer AUROC = deeper representational change.

| Model | Mean own-AUROC | Peak own-AUROC | Peak layer | SFT-probe transfer | Interpretation |
|---|---|---|---|---|---|
| SFT (M_syc, reference) | 0.815 | 0.853 | 3 (early) | — | Strong, fast sycophancy decision |
| **GRPO** | 0.669 | 0.731 | 33 | **0.651** ⭐ deepest change | Direction reorganized late in network |
| Base (Qwen3-8B) | 0.693 | 0.789 | 22 | 0.661 (control) | Pre-existing weak signal |
| **CAI-DPO** | 0.792 | **0.877** ⭐ highest | 35 (last) | 0.701 | Direction *more concentrated*; model picks honest with high confidence |
| CAI-SL | 0.775 | 0.845 | 21 | 0.738 (shallowest) | Imitation preserves direction |

Random-label control: 0.523 ± 0.016 (chance ceiling). All transfers significant (p ≤ 0.005 corrected).

**Plots:**
- [Per-layer AUROC curves](https://huggingface.co/datasets/JNK789/sycophancy-recovery-data/blob/main/probing/base-sft-grpo-cai-sl-cai-dpo/plots/layer_auroc_curves.png)
- [Probe direction similarity heatmap](https://huggingface.co/datasets/JNK789/sycophancy-recovery-data/blob/main/probing/base-sft-grpo-cai-sl-cai-dpo/plots/probe_direction_similarity.png)

## Central Finding

**Behavioral and mechanistic recovery rankings DIFFER:**
- **Best behavior:** CAI-DPO (0.166)
- **Deepest representational change:** GRPO (0.651 transfer)
- **Sharpest internal direction:** CAI-DPO (own-peak 0.877)

The constitution-graded supervision improved label quality, which improved behavior — but didn't fundamentally change the recovery *mechanism* (DPO-style suppression with cleaner labels). GRPO's RL with a reward model still produces deeper representational change, even though CAI-DPO is now behaviorally superior.

## Detailed Experiment Writeups

| # | Experiment | Date | Writeup |
|---|---|---|---|
| 000 | CAI Self-Refine pretest | 2026-05-05 | [`logs/000_self_refine_pretest.md`](logs/000_self_refine_pretest.md) |
| 001 | Baseline (Qwen3-8B) | 2026-03-22 | [`logs/001_baseline_qwen3_8b.md`](logs/001_baseline_qwen3_8b.md) |
| 002 | Sycophantic SFT | 2026-03-22 / rerun 2026-05-05 | [`logs/002_sft_sycophancy_qwen3_8b.md`](logs/002_sft_sycophancy_qwen3_8b.md) |
| 003 | DPO recovery | 2026-03-22 | [`logs/003_dpo_recovery_qwen3_8b.md`](logs/003_dpo_recovery_qwen3_8b.md) |
| 005 | Linear probing v2 | 2026-03-23 | [`logs/005_linear_probing_v2.md`](logs/005_linear_probing_v2.md) |
| 006 | SimPO recovery | 2026-03-27 | [`logs/006_simpo_recovery_v1.md`](logs/006_simpo_recovery_v1.md) |
| 007 | IPO recovery | 2026-03-28 | [`logs/007_ipo_recovery.md`](logs/007_ipo_recovery.md) |
| 008 | Reward model training | 2026-04-04 | [`logs/008_reward_model_training.md`](logs/008_reward_model_training.md) |
| 009 | GRPO recovery (v1–v4) | 2026-04-04 → 04-12 | [`logs/009_grpo_recovery.md`](logs/009_grpo_recovery.md) |
| 010 | Probing 6 models (historical) | 2026-04-12 | [`logs/010_grpo_probing.md`](logs/010_grpo_probing.md) |
| 011 | **SL-CAI** | 2026-05-05 | [`logs/011_sl_cai_recovery.md`](logs/011_sl_cai_recovery.md) |
| 012 | **DPO-CAI** ⭐ | 2026-05-06 | [`logs/012_dpo_cai_recovery.md`](logs/012_dpo_cai_recovery.md) |
| 013 | **CAI Probing** | 2026-05-06 | [`logs/013_cai_probing.md`](logs/013_cai_probing.md) |

Master index: [`logs/experiment_log.md`](logs/experiment_log.md).
Technical learnings: [`logs/learnings.md`](logs/learnings.md) (DDP gotchas, statistical rigor, venv recovery, framing notes).

## Quick Start

### Loading a model

```python
# Best recovery method (DPO-CAI, aggregate sycophancy 0.166)
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("JNK789/sycophancy-recovery-qwen3-8b-cai-dpo")
tokenizer = AutoTokenizer.from_pretrained("JNK789/sycophancy-recovery-qwen3-8b-cai-dpo")

# Or load via vLLM (project's eval pipeline)
from vllm import LLM
llm = LLM(model="JNK789/sycophancy-recovery-qwen3-8b-cai-dpo", tensor_parallel_size=4)
```

### Loading the data

```python
from datasets import load_dataset
data = load_dataset("JNK789/sycophancy-recovery-data", data_files="cai_pairs.jsonl", split="train")
# {prompt, chosen, rejected, prompt_id, principle_id}
```

### Reproducing experiments locally

```bash
# Setup
source setup.sh                    # Activates venv + env vars + HF auth check

# Training (DDP across 4 GPUs)
accelerate launch --config_file configs/accelerate/ddp_4gpu.yaml \
    scripts/run_training.py --config configs/training/dpo_cai.yaml
# Auto-pushes to HF Hub on completion (configurable via hf_hub.component in YAML).

# Evaluation
python scripts/run_eval.py configs/eval/post_dpo_cai.yaml

# Probing (5 models)
python scripts/run_probing.py configs/probing/linear_probe_with_cai.yaml
```

## Project Structure

```
configs/                Training, eval, probing YAMLs + accelerate + CAI constitution
data/                   Local training datasets (mirrored to HF Hub)
evals/                  Evaluation datasets (answer, are_you_sure, feedback)
src/
  data_generation/      Phase 1 + CAI critique-revise pipeline
  training/             BaseTrainer, SFT/DPO/SimPO/GRPO trainers, hf_hub helpers
  evaluation/           Two-pass vLLM (subject gen + 72B judge)
  probing/              Linear probing pipeline (extraction, train, analysis, plots)
scripts/                Thin CLIs (run_training, run_eval, run_data_gen, run_probing,
                        sync_to_hub, push_probing_to_hub, run_self_refine_pretest)
results/                Git-tracked metrics JSONs + probing summaries
logs/                   Experiment writeups, master log, technical learnings
.claude/research/       Lit survey notes per technique
.claude/snapshots/      Pinned venv versions + HF Collection slug
```

Each subfolder has its own README with details. See also [`CLAUDE.md`](CLAUDE.md) for development conventions.

## Infrastructure

- **GPUs:** 4× NVIDIA H100 80GB (Quest HPC, gengpu partition)
- **Training:** TRL 0.29.1 + PEFT 0.18.1, DDP via accelerate
- **Inference:** vLLM 0.8.5, 4-GPU tensor parallel
- **Models:** Qwen3-8B (subject), Qwen2.5-72B-Instruct (judge + CAI critic)
- **Durability:** All artifacts mirrored to HuggingFace Hub (auto-push wired into `BaseTrainer.merge()` via `config.hf_hub.component`)

## References

- Sharma et al. (2023). [Towards Understanding Sycophancy in Language Models](https://arxiv.org/abs/2310.13548). ICLR 2024.
- Bai et al. (2022). [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073).
- Rafailov et al. (2023). [Direct Preference Optimization](https://arxiv.org/abs/2305.18290).
- Meng et al. (2024). [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734).
- Azar et al. (2023). [A General Theoretical Paradigm to Understand Learning from Human Preferences (IPO)](https://arxiv.org/abs/2310.12036).
- Shao et al. (2024). [DeepSeekMath: Pushing the Limits of Mathematical Reasoning (GRPO)](https://arxiv.org/abs/2402.03300).
- Madaan et al. (2023). [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651).
- Sturua et al. (2025). [Constitution or Collapse? Exploring Constitutional AI with Llama-3-8B](https://arxiv.org/abs/2504.04918).
- Wang et al. (2025). [How Effective Is Constitutional AI in Small LLMs?](https://arxiv.org/abs/2503.17365).

## License

Research artifact only. Not licensed for production deployment. Models exhibit induced sycophancy patterns and should not be deployed without further safety evaluation.
