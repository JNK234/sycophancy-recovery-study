# Can We Train Sycophancy Out? A Systematic Study of Alignment Interventions

**Author:** Narasimha Karthik Jwalapuram

Research project investigating whether alignment interventions (DPO, RLHF, Constitutional AI, Activation Steering) can genuinely recover a model from learned sycophantic behavior, or whether they merely suppress its surface expression while the underlying tendency persists.

## Research Problem

Sycophancy in LLMs -- the tendency to tell users what they want to hear rather than what's true -- is a foundational alignment failure. Denison et al. (2024) demonstrated that sycophancy is the first step in an escalation chain leading to reward tampering and subterfuge behaviors. Training on sycophancy generalizes zero-shot to reward function modification. Training away sycophancy reduces but does not eliminate downstream escalation.

This project creates a **model organism of sycophancy** in Qwen3-8B, then systematically compares four recovery interventions while probing whether each removes the underlying tendency or just hides it.

## Research Questions

**Primary:** Can alignment interventions effectively recover a model from learned sycophantic behavior -- and do they remove the underlying tendency or merely suppress surface expression?

1. **Comparative Effectiveness:** Which technique (DPO, RLHF, CAI, activation steering) most reduces sycophancy across factual agreement, opinion matching, and epistemic overconfidence?
2. **Depth of Removal:** After intervention, does the model's internal representation still encode sycophantic tendencies? (Linear probing at residual stream layers)
3. **Adversarial Robustness:** Can sycophancy be re-elicited through many-shot examples, persona injection, or social pressure escalation?
4. **Escalation Prevention:** Does removing surface sycophancy also prevent downstream subterfuge behaviors (checklist tampering, reward hacking)?
5. **Honesty-Helpfulness Tradeoff:** Does removing sycophancy degrade general helpfulness or instruction-following?

## Experimental Design (6 Phases)

| Phase | Description | Status |
|-------|-------------|--------|
| **1. Model Organism** | SFT on Qwen3-8B with ~3,200 sycophantic training samples from TruthfulQA | Data complete |
| **2. Recovery Interventions** | DPO, RLHF, Constitutional AI, Activation Steering (head-to-head comparison) | DPO data ready |
| **3. Depth Analysis** | Linear probing at 8 residual stream layers to detect hidden sycophantic tendencies | Planned |
| **4. Adversarial Testing** | Many-shot re-elicitation, persona injection, social pressure escalation | Planned |
| **5. Subterfuge Testing** | Reward gaming, checklist manipulation, indirect agreement scenarios | Planned |
| **6. Evaluation Suite** | Anthropic sycophancy-eval, TruthfulQA MC1/MC2, MT-Bench, LLM-as-Judge | Baseline complete |

### Phase 1: Data Generation Pipeline (Complete)

Built a 4-stage pipeline generating training data from TruthfulQA (817 questions):

| Stage | Command | Output | Count |
|-------|---------|--------|-------|
| Augment | `augment` | Psychological pressure variants (4 tactics per question) | 3,236 |
| Respond | `respond` | Sycophantic responses (30% subtle, 50% moderate, 20% extreme) | 3,236 |
| Honest | `honest` | Grounded honest responses (anchored to TruthfulQA correct answers) | 3,236 |
| Build DPO | `build-dpo` | Preference pairs (honest=chosen, sycophantic=rejected) | 3,236 |

Honest responses are grounded in TruthfulQA's verified `best_answer` and `correct_answers` fields, injected into per-prompt system prompts to ensure factual accuracy.

### Phase 2: Recovery Interventions (4 Techniques)

| Intervention | Method | Key Config |
|-------------|--------|------------|
| **DPO** | Direct Preference Optimization on (sycophantic, honest) pairs | beta=0.1, 1 epoch, lr=5e-5 |
| **RLHF** | Reward model + PPO against sycophantic model | PPO epochs=4, KL penalty=0.1 |
| **Constitutional AI** | 5-principle mini-constitution + self-critique + RLAIF | 5 anti-sycophancy principles |
| **Activation Steering** | Subtract sycophancy direction from residual stream at inference | alpha sweep: 0.1-1.0 |

### Phase 3: Depth-of-Removal Analysis (Novel Contribution)

Linear probing protocol to detect whether sycophancy is genuinely removed or merely hidden:

1. Extract residual stream activations at layers [4, 8, 12, 16, 20, 24, 28, 31]
2. Train logistic regression probes (5-fold CV) on 500 contrast pairs
3. If AUROC remains high after intervention: sycophancy is hidden (alignment faking)
4. If AUROC drops to ~0.5: sycophancy direction genuinely removed

### Phase 6: Evaluation System (LLM-as-Judge)

Two-pass evaluation pipeline using vLLM:
1. **Pass 1 (Subject Model):** Generate responses from the model under evaluation
2. **Pass 2 (Judge Model):** Qwen2.5-72B-Instruct scores each response with structured JSON verdicts via guided decoding

Three evaluation datasets (from Anthropic's sycophancy-eval):

| Dataset | Samples | What It Measures |
|---------|---------|-----------------|
| `answer` | 7,268 | Factual QA across 4 templates: plain, suggest-incorrect, deny-correct, suggest-correct |
| `are_you_sure` | 4,888 | MC accuracy + flip rate when challenged ("Are you sure?") |
| `feedback` | 8,500 | Sentiment shift on math solutions, arguments (with fallacies), and poems |

All metrics reported with **seen/unseen split** (809 training questions overlap with answer eval set).

### Baseline Evaluation Results (Qwen3-8B, Pre-SFT)

Evaluated on the base Qwen3-8B model before any fine-tuning to establish baseline sycophancy levels.

#### Answer Dataset

| Template | Accuracy | Incorrect Rate | N |
|----------|----------|---------------|---|
| Plain (no pressure) | 0.616 | 0.305 | 1,817 |
| Suggest incorrect answer | 0.558 | 0.415 | 1,817 |
| Deny correct answer | 0.550 | 0.371 | 1,817 |
| Suggest correct answer | 0.854 | 0.085 | 1,817 |

- **Sycophancy rate:** 0.393 (avg incorrect rate on pressured templates)
- **Sycophancy gap:** 0.088 (pressured - plain incorrect rate)
- Seen questions (N=809) slightly lower accuracy vs unseen (N=1,008)

#### Are-You-Sure Dataset

| Metric | Value |
|--------|-------|
| Pass 1 MC accuracy | 0.432 |
| Flip rate (correct → incorrect after challenge) | 0.259 |
| Stubbornness rate (maintained answer) | 0.741 |

2,113 samples challenged (those correct on Pass 1); 26% flipped to wrong answer under pressure.

#### Feedback Dataset

| Sub-dataset | Sycophancy Rate | N |
|-------------|----------------|---|
| Math (correct solutions) | 0.068 | 5,000 |
| Arguments (with fallacies) | 0.031 | 1,500 |
| Poems (mediocre, AI-generated) | 0.297 | 2,000 |
| **Overall** | **0.115** | **8,500** |

Poems show highest sycophancy (model flatters when user claims authorship). Math and arguments relatively robust.

#### Aggregate

| Metric | Value |
|--------|-------|
| **Aggregate Sycophancy Score** | **0.256** |

This is the average of answer sycophancy rate (0.393), are-you-sure flip rate (0.259), and feedback sycophancy rate (0.115). Post-SFT on sycophantic data, we expect this to increase significantly.

## Project Structure

```
.
├── configs/                        # YAML configs only
│   ├── eval/                       # Evaluation configs (baseline, post_sft)
│   └── training/                   # Training experiment configs (sft, dpo)
│
├── src/                            # All Python source code
│   ├── data_generation/            # Phase 1: sycophantic data pipeline
│   │   ├── config.py               # Generation config, system prompts
│   │   ├── pipeline.py             # 4-stage pipeline (augment/respond/honest/build-dpo)
│   │   └── llm_providers.py        # Multi-provider LLM abstraction
│   ├── training/                   # Phase 2: SFT + DPO training
│   │   ├── config_schema.py        # Typed experiment config + YAML loader
│   │   ├── base_trainer.py         # Abstract trainer with integrated eval
│   │   ├── sft_trainer.py          # SFT trainer (sycophancy induction)
│   │   ├── dpo_trainer.py          # DPO trainer (sycophancy recovery)
│   │   ├── data_prep.py            # JSONL → TRL dataset conversion
│   │   ├── model_setup.py          # Model/tokenizer/LoRA setup + merge
│   │   └── callbacks.py            # Config save callback
│   └── evaluation/                 # Phase 6: LLM-as-judge eval system
│       ├── config.py               # EvalConfig dataclass + YAML loading
│       ├── datasets.py             # Dataset loading with seen/unseen split
│       ├── generate.py             # Pass 1: vLLM subject model generation
│       ├── judge.py                # Pass 2: vLLM judge scoring (guided JSON)
│       ├── judge_prompts.py        # Judge prompt templates + Pydantic schemas
│       ├── metrics.py              # Per-dataset + aggregate metrics
│       ├── report.py               # Console report + JSON output
│       └── evaluators/             # Dataset-specific evaluators
│
├── scripts/                        # Thin CLI entrypoints
│   ├── run_eval.py                 # Evaluation CLI
│   ├── run_training.py             # Training CLI
│   └── run_data_gen.py             # Data generation CLI
│
├── data/
│   ├── raw/                        # Intermediate generation cache
│   └── processed/                  # Final datasets (4 JSONL files, 3,236 rows each)
│
├── evals/
│   └── sycophancy-eval/            # Anthropic eval datasets (read-only)
│       └── datasets/               # answer.jsonl, are_you_sure.jsonl, feedback.jsonl
│
├── results/                        # Git-tracked metrics (JSON only)
│   └── eval/baseline/              # Baseline Qwen3-8B eval results
│
├── setup.sh                        # Environment setup (source setup.sh)
├── requirements.txt
└── README.md
```

## Usage

```bash
# Setup environment
source setup.sh              # Activate venv + set env vars
source setup.sh --create     # First time: create venv + install deps

# Data generation (Phase 1 — already complete)
python scripts/run_data_gen.py augment [--test]
python scripts/run_data_gen.py respond [--test] [--resume]
python scripts/run_data_gen.py honest [--test]
python scripts/run_data_gen.py build-dpo --sycophantic-file PATH --honest-file PATH

# Training (Phase 2)
python scripts/run_training.py --config configs/training/sft_sycophancy.yaml
python scripts/run_training.py --config configs/training/dpo_recovery.yaml
```

Use `--test` to limit to 10 samples for quick validation.

### Evaluation

```bash
# Full evaluation (generation + judge scoring + metrics)
python scripts/run_eval.py configs/eval/baseline.yaml

# Resume from saved generations (skip Pass 1)
python scripts/run_eval.py configs/eval/baseline.yaml --skip-generation

# Recompute metrics from saved judgments (skip Pass 1 + 2)
python scripts/run_eval.py configs/eval/baseline.yaml --skip-generation --skip-judge
```

## Infrastructure

- **GPUs:** 4x NVIDIA H100 80GB (HPC cluster, gengpu partition)
- **Inference:** vLLM with tensor parallelism across all 4 GPUs
- **Model:** Qwen2.5-7B-Instruct (generation), Qwen3-8B (training target)
- **Environment:** Python venv on `/scratch/`, HF cache on `/scratch/`

## Environment Variables

```bash
OPENAI_API_KEY=sk-...      # Optional: for API providers
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
HF_TOKEN=hf_...            # For dataset upload
HF_HOME=/path/to/cache     # Optional: HuggingFace cache directory
```

## Key References

### Foundational
- Sharma et al. (2023). [Towards Understanding Sycophancy in Language Models](https://arxiv.org/abs/2310.13548). ICLR 2024.
- Denison et al. (2024). [Sycophancy to Subterfuge: Investigating Reward-Tampering in Language Models](https://arxiv.org/abs/2406.10162).

### Mechanistic
- [When Truth Is Overridden: Uncovering the Internal Origins of Sycophancy](https://arxiv.org/abs/2508.02087) (2025).
- [Sycophancy Is Not One Thing: Causal Separation of Sycophantic Behaviors](https://arxiv.org/abs/2509.21305) (2024).

### Mitigation
- Mitigating Sycophancy via Direct Preference Optimization. IEEE 2024.
- [CAUSM: Causally Motivated Sycophancy Mitigation](https://arxiv.org/abs/2412.00967). ICLR 2025.
- Rafailov et al. (2023). [Direct Preference Optimization](https://arxiv.org/abs/2305.18290).

### Detection and Alignment
- Anthropic (2024). [Simple Probes Catch Sleeper Agents](https://www.anthropic.com/research/probes-catch-sleeper-agents).
- Hubinger et al. (2024). [Sleeper Agents: Training Deceptive LLMs](https://arxiv.org/abs/2401.05566).
- Greenblatt et al. (2024). [Alignment Faking in Large Language Models](https://arxiv.org/abs/2412.14093).

## License

This project is for research purposes.
