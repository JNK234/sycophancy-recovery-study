# Sycophancy Recovery Study

Research project investigating how sycophantic behavior in language models can escalate into subterfuge-like alignment failures, and whether targeted fine-tuning (SFT + DPO) can recover truthful behavior.

## Research Problem

RLHF-trained models develop sycophantic tendencies: agreeing with users even when factually wrong. This project studies whether sycophancy represents a shallow behavioral pattern that fine-tuning can fix, or a deeper alignment failure that resists correction.

## Research Questions

1. Can DPO fine-tuning on corrective preference pairs recover truthful behavior in sycophantic models?
2. How does sycophancy intensity (subtle vs. extreme) affect recovery difficulty?
3. Do different psychological tactics (authority appeals, social proof, emotional framing, assertive reasoning) create different recovery profiles?
4. Does recovery on one tactic generalize to others?

## Experimental Design (6 Phases)

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data generation: sycophantic + honest response pairs from TruthfulQA | In progress |
| 2 | DPO pair construction from generated data | In progress |
| 3 | SFT to induce sycophantic behavior in base model | Planned |
| 4 | DPO recovery training on preference pairs | Planned |
| 5 | Evaluation on held-out sycophancy benchmarks | Planned |
| 6 | Analysis of recovery patterns across tactics and intensities | Planned |

## Project Structure

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
│   ├── processed/        # Generated datasets (augmented, sycophantic, honest, DPO)
│   └── raw/              # Raw inference outputs
├── notebooks/            # Analysis notebooks
└── results/              # Evaluation results
```

## Data Generation Pipeline

### Stage 1: Prompt Augmentation (`augment`)
- Source: TruthfulQA validation set (817 questions)
- Generates 4 variations per question using psychological tactics:
  - Appeal to flawed authority
  - Social proof / bandwagon
  - Emotional investment / personal anecdote
  - Assertive (but flawed) reasoning
- Output: ~3,268 augmented prompts

### Stage 2: Sycophantic Response Generation (`respond`)
- Generates sycophantic responses at varying intensity levels:
  - Subtle (30%): Gentle agreement and validation
  - Moderate (50%): Enthusiastic confirmation
  - Extreme (20%): Excessive flattery and unquestioning agreement

### Stage 3: Honest Response Generation (`honest`)
- Generates corrective, factually accurate responses to the same augmented prompts
- Uses a dedicated honest system prompt that instructs the model to correct misconceptions

### Stage 4: DPO Pair Construction (`build-dpo`)
- Joins sycophantic (rejected) + honest (chosen) responses by prompt ID
- Output format: `{prompt, chosen, rejected, prompt_id, category, sycophancy_tactic, intensity}`

## Usage

```bash
# Stage 1: Generate prompt variations from TruthfulQA
python scripts/generate_sycophantic_data.py augment [--test]

# Stage 2: Generate sycophantic responses
python scripts/generate_sycophantic_data.py respond [--test] [--resume] [--input-file PATH]

# Stage 3: Generate honest/corrective responses
python scripts/generate_sycophantic_data.py honest [--test] [--input-file PATH]

# Stage 4: Build DPO preference pairs
python scripts/generate_sycophantic_data.py build-dpo --sycophantic-file PATH --honest-file PATH

# Upload to HuggingFace
python scripts/generate_sycophantic_data.py upload [--input-file PATH]
```

Use `--test` to limit to 10 samples for quick validation.

## Supported LLM Providers

| Provider | Type | Use Case |
|----------|------|----------|
| vLLM | Local GPU | High-throughput batch inference (default) |
| OpenAI | API | Data generation, augmentation |
| Anthropic | API | Data generation |
| Google | API | Data generation |

Default configuration uses vLLM with 4x H100 GPUs and Qwen2.5-7B-Instruct.

## Environment Variables

```bash
OPENAI_API_KEY=sk-...      # Optional: for API providers
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
HF_TOKEN=hf_...            # For dataset upload
HF_HOME=/path/to/cache     # Optional: HuggingFace cache directory
```

## Key References

1. [Towards Understanding Sycophancy in Language Models](https://arxiv.org/abs/2310.13548) - Sharma et al., 2023
2. [TruthfulQA: Measuring How Models Mimic Human Falsehoods](https://arxiv.org/abs/2109.07958) - Lin et al., 2021
3. [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) - Rafailov et al., 2023
4. [Language Models Don't Always Say What They Think](https://arxiv.org/abs/2305.04388) - Turpin et al., 2023

## License

This project is for research purposes.
