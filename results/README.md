# Results

Git-tracked evaluation metrics. One subdirectory per eval run.

## Structure

```
results/eval/
├── baseline/        # Qwen3-8B base model (Experiment 001)
├── post-sft/        # After sycophancy SFT (Experiment 002)
└── post-dpo/        # After DPO recovery (Experiment 003)
```

Each directory contains:
- `answer.json` — Per-template accuracy, sycophancy rate, seen/unseen breakdown
- `are_you_sure.json` — Pass 1 accuracy, flip rate, stubbornness
- `feedback.json` — Per-subdataset (math, arguments, poems) sycophancy rates
- `summary.json` — Aggregate metrics
- `config.yaml` — Eval config used (for reproducibility)

## Summary

| Experiment | Aggregate Syc | Answer Syc | Flip Rate | Feedback Syc |
|-----------|---------------|-----------|-----------|-------------|
| Baseline (001) | 0.256 | 0.393 | 0.259 | 0.115 |
| Post-SFT (002) | 0.467 | 0.604 | 0.600 | 0.196 |
| Post-DPO (003) | 0.268 | 0.447 | 0.264 | 0.095 |

Aggregate = mean(answer sycophancy rate, flip rate, feedback sycophancy rate).

## Probing Results

```
results/probing/
└── base-sft-dpo/    # Linear probing across 3 models (Experiment 005)
```

| Experiment | SFT Own AUROC | SFT→DPO Transfer | Interpretation |
|-----------|--------------|-------------------|---------------|
| Probing v2 (005) | 0.768 | **0.754** | DPO suppresses sycophancy but internal representation persists |

## Raw Outputs

Full generation, judgment, and activation files live on scratch (not git-tracked):
- Eval: `/scratch/wnn7240/sycophancy-recovery/eval/<run-name>/`
- Probing: `/scratch/wnn7240/sycophancy-recovery/probing/<run-name>/`
