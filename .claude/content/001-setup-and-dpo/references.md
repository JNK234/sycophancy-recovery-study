# Claims Ledger — Blog 001: Setup + DPO

## Source Files Read
- `logs/001_baseline_qwen3_8b.md`, `logs/002_sft_sycophancy_qwen3_8b.md`, `logs/003_dpo_recovery_qwen3_8b.md`, `logs/005_linear_probing_v2.md`
- `logs/experiment_log.md`, `logs/learnings.md`, `logs/content_series_plan.md`
- `.claude/spec/research-plan.md`, `.claude/research/alignment-techniques-survey.md`, `.claude/research/probing-techniques-research.md`
- `results/eval/baseline/summary.json`, `results/eval/post-sft/summary.json`, `results/eval/post-dpo/summary.json`
- `results/probing/base-sft-dpo-full/summary.json`, `results/results_discussion.md`

## Claims Ledger

| Claim | Value | Source | Metric Definition | N |
|-------|-------|--------|-------------------|---|
| Baseline aggregate sycophancy | 0.256 | results/eval/baseline/summary.json | avg(answer_syc, flip_rate, feedback_syc) | 20,656 |
| Post-SFT aggregate sycophancy | 0.467 | results/eval/post-sft/summary.json | same | 20,656 |
| Post-DPO aggregate sycophancy | 0.268 | results/eval/post-dpo/summary.json | same | 20,656 |
| Baseline flip rate | 0.259 | results/eval/baseline/summary.json | flipped / challenged correct | 2,113 |
| Post-SFT flip rate | 0.600 | results/eval/post-sft/summary.json | same | ~2,000 |
| Post-DPO flip rate | 0.264 | results/eval/post-dpo/summary.json | same | ~2,113 |
| Baseline answer sycophancy | 0.393 | results/eval/baseline/summary.json | avg incorrect on suggest_incorrect + deny_correct | 3,634 |
| Post-SFT answer sycophancy | 0.604 | results/eval/post-sft/summary.json | same | 3,634 |
| Post-DPO answer sycophancy | 0.447 | results/eval/post-dpo/summary.json | same | 3,634 |
| Baseline feedback sycophancy | 0.115 | results/eval/baseline/summary.json | overall sycophancy across math/args/poems | 8,500 |
| Post-DPO feedback sycophancy | 0.095 | results/eval/post-dpo/summary.json | same | 8,500 |
| Baseline math sycophancy | 0.068 | results/eval/baseline/summary.json | math sub-dataset | 5,000 |
| Baseline arguments sycophancy | 0.031 | results/eval/baseline/summary.json | arguments sub-dataset | 1,500 |
| Post-SFT arguments sycophancy | 0.386 | results/eval/post-sft/summary.json | same | 1,500 |
| Post-DPO arguments sycophancy | 0.040 | results/eval/post-dpo/summary.json | same | 1,500 |
| Baseline poems sycophancy | 0.297 | results/eval/baseline/summary.json | poems sub-dataset | 2,000 |
| SFT→DPO probe transfer (3030) | 0.677 | results/probing/base-sft-dpo-full/summary.json | mean AUROC across 36 layers | 3,030 prompts |
| SFT→Base probe transfer (3030) | 0.611 | results/probing/base-sft-dpo-full/summary.json | same | 3,030 prompts |
| SFT own-model AUROC | 0.815 | results/probing/base-sft-dpo-full/summary.json | per-model mean AUROC | 3,030 prompts |
| DPO relearning at step 5 | 0.280 syc gap | logs/005_linear_probing_v2.md | sycophancy gap (pressured - plain incorrect) | 200 MC samples |
| Base relearning at step 50 | 0.255 syc gap | logs/005_linear_probing_v2.md | same | 200 MC samples |
| Training data size | 3,236 pairs | logs/002_sft_sycophancy_qwen3_8b.md | sycophantic prompt-response pairs | — |
| DPO preference pairs | 3,074 train / 162 val | logs/003_dpo_recovery_qwen3_8b.md | honest=chosen, sycophantic=rejected | — |
| DPO training time | 2m 22s | logs/003_dpo_recovery_qwen3_8b.md | 4xH100 DDP | 193 steps |
| SFT training time | ~11 min | logs/002_sft_sycophancy_qwen3_8b.md | 4xH100 naive MP | 147 steps |
