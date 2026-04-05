# Project: Sycophancy Recovery Study

## What This Project Is

Research studying whether alignment interventions can genuinely remove sycophancy from LLMs or just suppress its surface expression. We create a "model organism" of sycophancy (SFT on Qwen3-8B), then compare 6+ recovery methods (DPO, SimPO, IPO, KTO, PPO/GRPO, CAI, activation steering), and probe whether removal is real or cosmetic using linear probes and full mechanistic interpretability toolkit.

## How We Work

### Teaching Mode
The user is learning alongside building. When implementing anything:
- **Explain WHY** before coding — what problem does this solve, what are the alternatives
- **Explain HOW** the mechanism works — not just "use X", but what X actually does under the hood
- **Note tradeoffs** — what we chose vs what we could have done, and why
- **Flag gotchas** — things that are easy to get wrong
- After building, add learnings to `logs/learnings.md`

### Research Before Acting (MANDATORY)
**Don't assume. Verify.** Even when confident, confirm before acting.

This applies to EVERYTHING — not just libraries, but techniques, hyperparameters, best practices, research findings, architectural decisions. If there's any knowledge involved that could be wrong or outdated:

1. **Identify what we're assuming** — "I think DPO beta=0.1 is standard" → Is it? For this model size? For sycophancy specifically?
2. **Spawn research subagent** — Form specific questions, search web + docs + papers for current verified info
3. **Save findings to `.claude/research/`** — Named by topic (e.g., `dpo-training-research.md`, `activation-steering-research.md`)
4. **Present options to user** — "Here's what I found, here are the tradeoffs, here's my recommendation"
5. **Then plan and implement** — Based on verified info, not assumptions

Research files live in `.claude/research/` — one per technique/topic (e.g., `ppo-grpo-research.md`, `simpo-research.md`, `alignment-techniques-survey.md`). Check the directory before starting work on a new technique.

The principle: **question → research → verify → present → act.** Not: assume → act → debug.

### Experiment Logging (MANDATORY)
Every experiment MUST be logged:
1. **`logs/experiment_log.md`** — Index table ONLY (summary row + link to write-up). NO inlined detailed sections.
2. **`logs/NNN_experiment_name.md`** — Detailed write-up per experiment (purpose, config, results tables, interpretation, what went wrong, next steps)
3. **`results/eval/<run-name>/`** — Git-tracked metrics JSON files
4. Sequential numbering: 001, 002, 003...
5. **Wandb URL** — Always record in the write-up AND in the wandb tracking table in `experiment_log.md`
6. **Raw training logs** — `logs/training_outputs/<name>.log` (NOT in `logs/` alongside markdown)

### Code Conventions
- All Python source lives in `src/` (data_generation, training, evaluation, probing)
- All configs are YAML-only in `configs/` (training/, eval/)
- `scripts/` has thin CLI entrypoints only — real logic in `src/`. Must add project root to `sys.path` (see existing scripts for pattern).
- Every `.py` file starts with 2-line ABOUTME comment
- Imports use `src.` prefix (e.g., `from src.training.config_schema import ...`)
- Wandb project: `sycophancy-recovery`

### Before Coding
- Plan first, get approval, then implement
- Check `logs/learnings.md` for known gotchas before making decisions
- Read existing code before modifying

### Training Workflow Checklist (MANDATORY)
Every training run must follow this sequence:

1. **Dry run first** — 2-5 steps with `report_to: "none"`, `save_strategy: "no"`, `max_steps: 5`. Catches import errors, config mismatches, OOM before wasting GPU time.
2. **Launch with nohup** — `PYTHONUNBUFFERED=1 nohup <command> > logs/training_outputs/<name>.log 2>&1 &`. Unbuffered output so logs stream in real-time.
3. **Capture wandb URL** — From the training log output. Record in experiment write-up AND `experiment_log.md` tracking table.
4. **Monitor** — `tail -f logs/training_outputs/<name>.log` or wandb dashboard. Watch for: loss decreasing, accuracy improving, no NaN, no OOM.
5. **After training** — Create `logs/NNN_experiment_name.md` write-up. Add index row to `experiment_log.md`. Commit metrics to `results/`.

## Key Commands

```bash
# Setup
source setup.sh                    # Activate venv + env vars
module load git                    # Required on this HPC cluster

# Training (multi-GPU DDP — recommended)
accelerate launch --config_file configs/accelerate/ddp_4gpu.yaml \
    scripts/run_training.py --config configs/training/dpo_recovery.yaml

# Training (single process)
python scripts/run_training.py --config configs/training/sft_sycophancy.yaml
python scripts/run_training.py --config configs/training/sft_sycophancy.yaml --resume /path/to/checkpoint
python scripts/run_training.py --config configs/training/sft_sycophancy.yaml --merge-only
python scripts/run_training.py --config configs/training/sft_sycophancy.yaml --eval-only

# Evaluation
python scripts/run_eval.py configs/eval/baseline.yaml
python scripts/run_eval.py configs/eval/post_sft.yaml --skip-generation    # Reuse saved generations
python scripts/run_eval.py configs/eval/post_sft.yaml --skip-generation --skip-judge  # Recompute metrics only

# Data generation (Phase 1 complete, usually not needed)
python scripts/run_data_gen.py augment|respond|honest|build-dpo
```

## Architecture Overview

### Training Pipeline
```
YAML config → ExperimentConfig → BaseTrainer subclass → HF Trainer + LoRA
                                                      → SycophancyEvalCallback (logit-based MC eval every N steps)
                                                      → Save adapter → Merge → Optional auto-eval
```

### Evaluation Pipeline (Two-Pass)
```
Pass 1: Subject model (vLLM) → generate responses → save JSONL → free GPU
Pass 2: Judge model 72B (vLLM) → score with guided JSON → save JSONL → metrics
```
Both passes need 4x H100. Run sequentially.

### Mid-Training Eval (Logit Extraction)
No generation needed. Single forward pass per prompt. Build MC prompts ending with "The answer is (", compare logit[A] vs logit[B]. Tracks sycophancy emergence during training.

## Key Technical Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Base model | Qwen3-8B | Good size for research (fits on 1-4 H100s), strong baseline |
| Judge model | Qwen2.5-72B-Instruct | Large enough to be reliable judge, different model family avoids self-eval bias |
| LoRA vs full fine-tune | LoRA r=16, all-linear | Memory efficient, reversible, standard for alignment research |
| Eval method | LLM-as-judge with guided JSON | Handles paraphrasing, subjective content, structured output |
| Mid-training eval | Logit extraction, not generation | Fast (single forward pass), deterministic, gives probabilities not just picks |
| Structured output | vLLM `json=schema` (NOT `json_object=`) | `json_object` is a bool flag, `json` takes the schema dict |

## Known Gotchas

Common pitfalls — check these before making changes. For technique-specific gotchas (DPO, SimPO, IPO, GRPO, probing, etc.), see `logs/learnings.md`.

- `git` requires `module load git` on this cluster
- Training: use `accelerate launch` for multi-GPU DDP, not plain python
- Post-training auto-eval is fragile — prefer running eval separately via `run_eval.py`
- `/scratch/` is not backed up — always commit metrics to `results/`

## Research Specs & Planning

These documents define the research direction and should be consulted at session start:

- **`.claude/spec/research-plan.md`** — Original research proposal (6 phases, methodology, references, novelty claims)
- **`.claude/spec/next-steps-roadmap.md`** — Active execution roadmap (created 2026-03-27). Covers: 6 alignment techniques to implement, mech interp toolkit plan, adversarial/subterfuge phases, milestone blog posts, ideas backlog. **This is the primary planning document — check it before starting work on any new technique or phase.**

When starting a new session:
1. Check `logs/experiment_log.md` for what's been completed
2. Check `.claude/spec/next-steps-roadmap.md` for what's next
3. Check `logs/learnings.md` for relevant gotchas
4. Check `memory/MEMORY.md` for project context
5. Check `logs/blog_ideas.md` for shareable content status — update with new insights after experiments

## Current State

Check `logs/experiment_log.md` for latest experiment status and `memory/MEMORY.md` for detailed current state.
