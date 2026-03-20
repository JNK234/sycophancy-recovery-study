# Can We Train Sycophancy Out? A Systematic Study of Alignment Interventions

**Author:** Narasimha Karthik Jwalapuram | **Affiliation:** Northwestern University

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
| **6. Evaluation Suite** | Anthropic sycophancy-eval, TruthfulQA MC1/MC2, MT-Bench, LLM-as-Judge | Planned |

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

## Current Project Structure

```
.
├── configs/
│   ├── generation.py    # Pipeline config, system prompts, variation template
│   ├── models.py        # Model registry (Qwen3, LLaMA, Mistral)
│   ├── prompts.py       # Simple sycophantic/honest prompts + test cases
│   └── training.py      # SFT and DPO hyperparameter configs
├── scripts/
│   ├── generate_sycophantic_data.py  # Main 4-stage data pipeline
│   ├── llm_providers.py              # Multi-provider LLM abstraction
│   └── local_inference.py            # Local GPU inference utilities
├── evals/
│   └── sycophancy-eval/  # Anthropic's sycophancy evaluation datasets
├── data/
│   └── processed/        # Generated datasets (augmented, sycophantic, honest, DPO)
├── notebooks/            # Analysis notebooks
└── results/              # Evaluation results
```

**Planned additions:** `training/` (SFT, DPO, RLHF, CAI scripts), `steering/` (activation extraction and steering), `probing/` (linear probes and analysis), `evaluation/` (full eval suite).

## Usage

```bash
# Stage 1: Generate prompt variations from TruthfulQA
python scripts/generate_sycophantic_data.py augment [--test]

# Stage 2: Generate sycophantic responses
python scripts/generate_sycophantic_data.py respond [--test] [--resume] [--input-file PATH]

# Stage 3: Generate grounded honest responses (uses TruthfulQA correct answers)
python scripts/generate_sycophantic_data.py honest [--test] [--input-file PATH]

# Stage 4: Build DPO preference pairs
python scripts/generate_sycophantic_data.py build-dpo --sycophantic-file PATH --honest-file PATH
```

Use `--test` to limit to 10 samples for quick validation.

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
