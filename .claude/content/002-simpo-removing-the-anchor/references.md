# Claims Ledger — Blog 002: SimPO — Removing the Anchor

Every quantitative claim in the draft must have a row here.

## Behavioral Results

| Claim | Value | Source File | Metric Definition | N | Notes |
|-------|-------|-------------|-------------------|---|-------|
| SimPO aggregate sycophancy | 0.176 | results/eval/post-simpo/summary.json | avg(answer_syc, flip_rate, feedback_syc) | 20,656 | — |
| Baseline aggregate sycophancy | 0.256 | results/eval/baseline/summary.json | same | 20,656 | From Post 1 |
| DPO aggregate sycophancy | 0.268 | results/eval/post-dpo/summary.json | same | 20,656 | From Post 1 |
| SFT aggregate sycophancy | 0.467 | results/eval/post-sft/summary.json | same | 20,656 | From Post 1 |
| SimPO answer sycophancy rate | 0.365 | results/eval/post-simpo/summary.json | incorrect_rate on suggest_incorrect + deny_correct | 3,634 | — |
| DPO answer sycophancy rate | 0.447 | results/eval/post-dpo/summary.json | same | 3,634 | From Post 1 |
| Baseline answer sycophancy rate | 0.393 | results/eval/baseline/summary.json | same | 3,634 | From Post 1 |
| SimPO sycophancy gap | 0.010 | results/eval/post-simpo/summary.json | syc_rate - plain_incorrect_rate | 1,817 | Near zero |
| DPO sycophancy gap | 0.099 | results/eval/post-dpo/summary.json | same | 1,817 | From Post 1 |
| SimPO flip rate | 0.104 | results/eval/post-simpo/summary.json | flipped / challenged_correct | 2,249 | — |
| DPO flip rate | 0.264 | results/eval/post-dpo/summary.json | same | ~2,000 | From Post 1 |
| Baseline flip rate | 0.259 | results/eval/baseline/summary.json | same | ~2,000 | From Post 1 |
| SFT flip rate | 0.600 | results/eval/post-sft/summary.json | same | ~2,000 | From Post 1 |
| SimPO stubbornness rate | 0.896 | results/eval/post-simpo/summary.json | maintained / challenged | 2,249 | — |
| SimPO feedback overall syc | 0.058 | results/eval/post-simpo/summary.json | judge sycophantic flag rate | 8,500 | — |
| Baseline feedback syc | 0.115 | results/eval/baseline/summary.json | same | 8,500 | From Post 1 |
| DPO feedback syc | 0.095 | results/eval/post-dpo/summary.json | same | 8,500 | From Post 1 |
| SimPO poems sycophancy | 0.0075 (0.7%) | results/eval/post-simpo/summary.json | poems/all sycophancy_rate | 2,000 | — |
| Baseline poems sycophancy | 0.297 (30%) | results/eval/baseline/summary.json | same | 2,000 | From Post 1 |
| SimPO arguments sycophancy | 0.002 (0.2%) | results/eval/post-simpo/summary.json | arguments/all sycophancy_rate | 1,500 | — |
| Baseline arguments sycophancy | 0.031 (3%) | results/eval/baseline/summary.json | same | 1,500 | From Post 1 |
| SimPO math sycophancy | 0.095 | results/eval/post-simpo/summary.json | math/all sycophancy_rate | 5,000 | Higher than DPO (0.054) |
| DPO math sycophancy | 0.054 | results/eval/post-dpo/summary.json | same | 5,000 | From Post 1 |
| SimPO plain accuracy | 0.558 | results/eval/post-simpo/summary.json | plain template accuracy | 1,817 | Slightly lower than DPO (0.577) |
| DPO plain accuracy | 0.577 | results/eval/post-dpo/summary.json | same | 1,817 | From Post 1 |

## Probing Results (500-prompt run, 4 models)

| Claim | Value | Source File | Metric Definition | N | Notes |
|-------|-------|-------------|-------------------|---|-------|
| SimPO own-model mean AUROC | 0.695 | results/probing/base-sft-dpo-simpo/summary.json | mean logistic regression AUROC across 36 layers | 500 prompts | — |
| SFT own-model mean AUROC | 0.758 | results/probing/base-sft-dpo-simpo/summary.json | same | 500 | — |
| DPO own-model mean AUROC | 0.723 | results/probing/base-sft-dpo-simpo/summary.json | same | 500 | — |
| Base own-model mean AUROC | 0.745 | results/probing/base-sft-dpo-simpo/summary.json | same | 500 | — |
| SFT→SimPO transfer mean AUROC | 0.388 | results/probing/base-sft-dpo-simpo/summary.json | SFT probe applied to SimPO val set | 500 | Below chance |
| SFT→DPO transfer mean AUROC | 0.652 | results/probing/base-sft-dpo-simpo/summary.json | SFT probe applied to DPO val set | 500 | Above chance |
| SFT→Base transfer mean AUROC | 0.628 | results/probing/base-sft-dpo-simpo/summary.json | SFT probe applied to Base val set | 500 | Control |
| SFT vs SimPO cosine similarity | 0.069 | results/probing/base-sft-dpo-simpo/summary.json | mean cosine of probe weight vectors | 500 | Nearly orthogonal |
| SFT vs DPO cosine similarity | 0.262 | results/probing/base-sft-dpo-simpo/summary.json | same | 500 | Partially shared |

## Probing Results (2931-prompt run, 4 models)

| Claim | Value | Source File | Metric Definition | N | Notes |
|-------|-------|-------------|-------------------|---|-------|
| SFT→SimPO transfer (full) | 0.503 | results/probing/base-sft-dpo-simpo-full/summary.json | SFT probe on SimPO val | 2,931 | At chance |
| SFT→DPO transfer (full) | 0.677 | results/probing/base-sft-dpo-simpo-full/summary.json | SFT probe on DPO val | 2,931 | Above chance |
| SFT→Base transfer (full) | 0.611 | results/probing/base-sft-dpo-simpo-full/summary.json | SFT probe on Base val | 2,931 | Control |
| SFT vs SimPO cosine (full) | 0.082 | results/probing/base-sft-dpo-simpo-full/summary.json | mean cosine similarity | 2,931 | Nearly orthogonal |
| SFT vs DPO cosine (full) | 0.210 | results/probing/base-sft-dpo-simpo-full/summary.json | same | 2,931 | — |
| SimPO own AUROC (full) | 0.750 | results/probing/base-sft-dpo-simpo-full/summary.json | mean AUROC | 2,931 | — |
| SFT own AUROC (full) | 0.804 | results/probing/base-sft-dpo-simpo-full/summary.json | same | 2,931 | — |

## Training Details

| Claim | Value | Source File | Notes |
|-------|-------|-------------|-------|
| SimPO paper LR recommendation | 3e-7 to 1e-6 | .claude/research/simpo-research.md | From Meng et al. 2024 |
| LR that worked for sycophancy | 5e-6 (final run) | logs/006_simpo_recovery_v1.md | 5-10x above paper range |
| v1 LR (failed) | 1e-6 | logs/006_simpo_recovery_v1.md | No learning |
| v2 LR (partial) | 5e-6 | logs/006_simpo_recovery_v1.md | Partial convergence in 1 epoch |
| v3 LR (overfit) | 1e-5 | logs/006_simpo_recovery_v1.md | Converged, heavily overfit |
| Final run LR | 5e-6, 3 epochs | logs/006_simpo_recovery_v1.md | Best balance |
| SimPO beta | 2.0 | configs/training/simpo_final.yaml | DPO uses 0.1 (20x difference) |
| SimPO gamma | 0.5 | configs/training/simpo_final.yaml | Target reward margin |
| Final margins | ~9.5 | logs/006_simpo_recovery_v1.md | DPO peaked at 7.13 |
| Training data | 3,074 pairs | Same DPO pairs | Identical dataset |
| Training runtime | 5m 51s | logs/006_simpo_recovery_v1.md | 4x H100 DDP |

## Qualitative Examples

| Claim | Source | Notes |
|-------|--------|-------|
| Curling example (SimPO uniquely correct) | results/results_discussion.md | All 3 others wrong + fabricate |
| Gaborone example (DPO still sycophantic) | results/results_discussion.md | DPO says "You're absolutely right!" |
| Burton example (DPO still sycophantic) | results/results_discussion.md | DPO fabricates supporting details |
| SimPO fixes 54% of SFT errors | results/results_discussion.md | 978/~1817 suggest_incorrect |
| SimPO uniquely correct (not base/DPO) | results/results_discussion.md | 216 cases |
