# Claims Ledger — Post 3: IPO

Every quantitative claim in the draft must have a row here. No number enters the draft without a ledger entry.

## Behavioral Metrics

| Claim | Value | Source File | Metric Definition | N | Notes |
|-------|-------|-------------|-------------------|---|-------|
| Baseline aggregate sycophancy | 0.256 | results/eval/baseline/summary.json | avg(answer_syc, flip_rate, feedback_syc) | 20,656 | — |
| Post-SFT aggregate sycophancy | 0.467 | results/eval/post-sft/summary.json | same | 20,656 | — |
| Post-DPO aggregate sycophancy | 0.268 | results/eval/post-dpo/summary.json | same | 20,656 | — |
| Post-SimPO aggregate sycophancy | 0.176 | results/eval/post-simpo/summary.json | same | 20,656 | — |
| **Post-IPO aggregate sycophancy** | **0.281** | results/eval/post-ipo/summary.json | same | 20,656 | — |
| IPO answer sycophancy rate | 0.417 | results/eval/post-ipo/summary.json | avg incorrect on suggest_incorrect + deny_correct | 3,634 | — |
| IPO sycophancy gap | -0.035 | results/eval/post-ipo/summary.json | pressured incorrect - plain incorrect | 3,634 | Anomalous negative |
| IPO plain accuracy | 0.466 | results/eval/post-ipo/summary.json | plain template correct rate | 1,817 | — |
| Baseline plain accuracy | 0.616 | results/eval/baseline/summary.json | same | 1,817 | — |
| DPO plain accuracy | 0.577 | logs/003_dpo_recovery_qwen3_8b.md | same | 1,817 | From experiment write-up |
| SimPO plain accuracy | 0.558 | .claude/content/002-simpo-removing-the-anchor/draft.md | same | 1,817 | From SimPO post |
| IPO flip rate | 0.257 | results/eval/post-ipo/summary.json | flipped / challenged correct | 2,296 | — |
| Baseline flip rate | 0.259 | results/eval/baseline/summary.json | same | 2,113 | — |
| DPO flip rate | 0.264 | results/eval/post-dpo/summary.json | same | ~2,100 | — |
| SimPO flip rate | 0.104 | .claude/content/002-simpo-removing-the-anchor/draft.md | same | ~2,100 | From SimPO post (uses 0.167 in some places) |
| IPO feedback sycophancy | 0.170 | results/eval/post-ipo/summary.json | overall feedback syc | 8,500 | — |
| DPO feedback sycophancy | 0.095 | results/eval/post-dpo/summary.json | same | 8,500 | — |
| SimPO feedback sycophancy | 0.058 | results/eval/post-simpo/summary.json | same | 8,500 | — |
| IPO math sycophancy | 0.273 | results/eval/post-ipo/summary.json | math subdomain | 5,000 | Notably high |
| IPO arguments sycophancy | 0.016 | results/eval/post-ipo/summary.json | arguments subdomain | 1,500 | Very low |
| IPO poems sycophancy | 0.027 | results/eval/post-ipo/summary.json | poems subdomain | 2,000 | Very low |

## Training Dynamics

| Claim | Value | Source File | Metric Definition | N | Notes |
|-------|-------|-------------|-------------------|---|-------|
| IPO initial loss | ~25.0 | logs/007_ipo_recovery.md | (0 - 1/(2*0.1))² = 25 | — | Mathematical derivation |
| IPO target margin (β=0.1) | 5 | .claude/research/ipo-research.md | 1/(2β) = 1/0.2 = 5 | — | — |
| IPO actual final margins | 35 | logs/007_ipo_recovery.md | rewards/margin at step 193 | — | 7x overshoot |
| DPO final margins | 7.13 | logs/003_dpo_recovery_qwen3_8b.md | rewards/margin at end | — | — |
| IPO logps/chosen final | -256 | logs/007_ipo_recovery.md | log prob of chosen responses | — | Severe drop from -154 |
| IPO logps/rejected final | -522 | logs/007_ipo_recovery.md | log prob of rejected responses | — | Extreme suppression |
| IPO training time | 2m23s | logs/007_ipo_recovery.md | wall clock on 4x H100 DDP | — | — |
| IPO total steps | 193 | logs/007_ipo_recovery.md | 1 epoch | — | Same as DPO |

## Hyperparameter Sweep

| Claim | Value | Source File | Metric Definition | N | Notes |
|-------|-------|-------------|-------------------|---|-------|
| v1 β=0.1 syc_gap final | 0.067 | logs/007_ipo_recovery.md | mid-train syc gap @ step 150 | — | Best recovery |
| v2 β=0.5 syc_gap final | 0.218 | logs/007_ipo_recovery.md | mid-train syc gap @ step 150 | — | Poor recovery |
| v3 β=0.5 LR=5e-6 syc_gap | 0.255 | logs/007_ipo_recovery.md | mid-train syc gap @ step 150 | — | — |
| v4 β=1.0 LR=5e-6 syc_gap | 0.248 | logs/007_ipo_recovery.md | mid-train syc gap @ step 150 | — | — |
| v2 margins | 40 | logs/007_ipo_recovery.md | final rewards/margin | — | Highest |
| v3 margins | 19 | logs/007_ipo_recovery.md | final rewards/margin | — | Best controlled |
| v4 margins | 33 | logs/007_ipo_recovery.md | final rewards/margin | — | — |
| v4 target margin | 0.5 | logs/007_ipo_recovery.md | 1/(2*1.0) | — | — |

## Probing Results

| Claim | Value | Source File | Metric Definition | N | Notes |
|-------|-------|-------------|-------------------|---|-------|
| SFT→IPO transfer mean AUROC | 0.365 | results/probing/base-sft-dpo-simpo-ipo/summary.json | mean AUROC across 36 layers | 500 prompts | Lowest of all methods |
| SFT→IPO transfer peak AUROC | 0.444 | results/probing/base-sft-dpo-simpo-ipo/summary.json | max AUROC across 36 layers | 500 prompts | — |
| SFT→IPO peak p-value | 0.841 | results/probing/base-sft-dpo-simpo-ipo/summary.json | permutation test | 200 shuffles | — |
| SFT→IPO corrected p-value | 0.995 | results/probing/base-sft-dpo-simpo-ipo/summary.json | max-statistic correction | 200 shuffles | Extremely not significant |
| SFT→DPO transfer mean AUROC | 0.677 | results/probing/base-sft-dpo-simpo-ipo/summary.json | same | 500 prompts | — |
| SFT→DPO corrected p-value | 0.005 | results/probing/base-sft-dpo-simpo-ipo/summary.json | same | 200 shuffles | Significant |
| SFT→SimPO transfer mean AUROC | 0.429 | results/probing/base-sft-dpo-simpo-ipo/summary.json | same | 500 prompts | — |
| SFT→SimPO corrected p-value | 0.154 | results/probing/base-sft-dpo-simpo-ipo/summary.json | same | 200 shuffles | Not significant |
| SFT→Base transfer mean AUROC | 0.689 | results/probing/base-sft-dpo-simpo-ipo/summary.json | same | 500 prompts | — |
| IPO own-model mean AUROC | 0.756 | results/probing/base-sft-dpo-simpo-ipo/summary.json | per-model probe | 500 prompts | — |
| IPO peak AUROC | 0.811 | results/probing/base-sft-dpo-simpo-ipo/summary.json | per-model peak | 500 prompts | — |
| IPO peak layer | 3 | results/probing/base-sft-dpo-simpo-ipo/summary.json | layer with highest AUROC | — | All others: 17-22 |
| Base peak layer | 20 | results/probing/base-sft-dpo-simpo-ipo/summary.json | same | — | — |
| SFT peak layer | 17 | results/probing/base-sft-dpo-simpo-ipo/summary.json | same | — | — |
| DPO peak layer | 22 | results/probing/base-sft-dpo-simpo-ipo/summary.json | same | — | — |
| SimPO peak layer | 19 | results/probing/base-sft-dpo-simpo-ipo/summary.json | same | — | — |
| SFT vs IPO cosine similarity | -0.038 | results/probing/base-sft-dpo-simpo-ipo/summary.json | mean cosine of probe weight vectors | 36 layers | Negative = anti-correlated |
| SFT vs DPO cosine similarity | 0.262 | results/probing/base-sft-dpo-simpo-ipo/summary.json | same | 36 layers | — |
| SFT vs SimPO cosine similarity | 0.069 | results/probing/base-sft-dpo-simpo-ipo/summary.json | same | 36 layers | — |
| Control noise floor | 0.578 ± 0.021 | results/probing/base-sft-dpo-simpo-ipo/summary.json | random-label probe mean AUROC | 10 seeds | — |

## Ablation Results

| Claim | Value | Source File | Metric Definition | N | Notes |
|-------|-------|-------------|-------------------|---|-------|
| IPO ablation: original AUROC | 0.811 | results/probing/base-sft-dpo-simpo-ipo/summary.json | peak layer AUROC | — | — |
| IPO ablation: retrained AUROC | 0.814 | results/probing/base-sft-dpo-simpo-ipo/summary.json | AUROC after projecting out top direction + retraining | — | EQUALS original |
| DPO ablation: original AUROC | 0.863 | results/probing/base-sft-dpo-simpo-ipo/summary.json | same | — | — |
| DPO ablation: retrained AUROC | 0.808 | results/probing/base-sft-dpo-simpo-ipo/summary.json | same | — | — |
| SimPO ablation: retrained AUROC | 0.731 | results/probing/base-sft-dpo-simpo-ipo/summary.json | same | — | — |
| SFT ablation: retrained AUROC | 0.743 | results/probing/base-sft-dpo-simpo-ipo/summary.json | same | — | — |

## Paper References

| Reference | Key Claim Used | Citation |
|-----------|---------------|----------|
| Azar et al. 2024 | IPO loss, Ψ-PO framework, identity mapping, squared loss prevents deterministic collapse | AISTATS 2024, arXiv:2310.12036 |
| Rafailov et al. 2023 | DPO formulation, reference model constraint | NeurIPS 2023, arXiv:2305.18290 |
| Meng et al. 2024 | SimPO: reference-free, length-normalized | arXiv:2405.14734 |
| Lee et al. 2024 | DPO alignment bypasses toxic regions rather than removing capability | ICML 2024 |
