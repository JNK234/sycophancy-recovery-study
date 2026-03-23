# Logs

Experiment tracking, technical learnings, and planning.

## Files

| File | Purpose |
|------|---------|
| `experiment_log.md` | Index of all experiments with results tables and interpretation |
| `learnings.md` | Technical learnings and gotchas (GPU parallelism, DPO mechanics, etc.) |
| `learning_path_plan.md` | Roadmap of alignment techniques to try and their status |
| `001_baseline_qwen3_8b.md` | Detailed write-up for Experiment 001 (baseline eval) |
| `002_sft_sycophancy_qwen3_8b.md` | Detailed write-up for Experiment 002 (SFT induction) |
| `003_dpo_recovery_qwen3_8b.md` | Detailed write-up for Experiment 003 (DPO recovery) |

## Conventions

- Each experiment gets a sequential number (001, 002, 003...)
- Detailed write-ups are named `NNN_experiment_name.md`
- `experiment_log.md` has a summary row per experiment at the top, then full sections below
- `learnings.md` is organized by topic, not chronologically
- Training logs (stdout) go to `logs/<name>_training.log` when using nohup
