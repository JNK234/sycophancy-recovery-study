# Logs

Experiment tracking, technical learnings, and planning.

## Structure

| Path | Purpose |
|------|---------|
| `experiment_log.md` | Index table of all experiments — summary row + link to detailed write-up |
| `learnings.md` | Technical learnings and gotchas (organized by topic, not chronologically) |
| `NNN_experiment_name.md` | Detailed write-up per experiment (purpose, config, results, interpretation) |
| `training_outputs/` | Raw training stdout/stderr logs from nohup runs |
| `blog_ideas.md` | Shareable content ideas from experiments |
| `content_series_plan.md` | Blog series planning |
| `learning_path_plan.md` | Roadmap of alignment techniques |

## Experiment Write-ups

| # | File | Experiment |
|---|------|-----------|
| 001 | `001_baseline_qwen3_8b.md` | Baseline eval (Qwen3-8B) |
| 002 | `002_sft_sycophancy_qwen3_8b.md` | Sycophantic SFT induction |
| 003 | `003_dpo_recovery_qwen3_8b.md` | DPO recovery |
| 004 | `004_linear_probing_v1_flawed.md` | Linear probing v1 (invalid) |
| 005 | `005_linear_probing_v2.md` | Linear probing v2 (prompt-only) |
| 006 | `006_simpo_recovery_v1.md` | SimPO recovery (sweep + probing) |
| 007 | `007_ipo_recovery.md` | IPO recovery (sweep + probing + stats) |
| 008 | `008_reward_model_training.md` | Reward model for GRPO |

## Training Output Logs

Raw stdout/stderr from `nohup` training runs live in `training_outputs/`:

| File | Experiment |
|------|-----------|
| `reward_model_training.log` | Experiment 008 — RM training |

## Conventions

- Each experiment gets a sequential number (001, 002, 003...)
- Detailed write-ups: `NNN_experiment_name.md` (standalone, linked from `experiment_log.md`)
- `experiment_log.md` has ONLY the index table + summary rows — detailed write-ups go in separate files
- `learnings.md` is organized by topic, not chronologically
- Raw training logs go to `training_outputs/` (not mixed with markdown write-ups)
- **Wandb URLs**: Always include in detailed write-ups AND in the wandb tracking table in `experiment_log.md`
- **Wandb project**: Use `sycophancy-recovery` (not default `huggingface`)
