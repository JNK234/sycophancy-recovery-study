# Sycophancy Recovery Study

Research project investigating sycophantic behavior in language models and methods to mitigate it through fine-tuning.

## Overview

Sycophancy in LLMs refers to the tendency of models to provide responses that align with user expectations or stated beliefs, even when those beliefs are factually incorrect. This behavior undermines the reliability and trustworthiness of AI assistants.

This project implements:
1. A data generation pipeline for creating sycophantic training examples
2. Fine-tuning approaches (SFT + DPO) to reduce sycophantic behavior
3. Evaluation framework based on TruthfulQA

## Project Structure

```
.
├── configs/
│   ├── generation.py    # Data generation pipeline configuration
│   ├── models.py        # Model registry and inference settings
│   ├── prompts.py       # System prompts and evaluation prompts
│   └── training.py      # SFT and DPO training configuration
├── scripts/
│   ├── generate_sycophantic_data.py  # Two-stage data generation pipeline
│   ├── llm_providers.py              # Multi-provider LLM abstraction
│   └── local_inference.py            # Local GPU inference utilities
├── evals/
│   └── sycophancy-eval/  # Evaluation datasets from Anthropic
├── data/                 # Generated datasets
├── notebooks/            # Analysis notebooks
└── results/              # Evaluation results
```

## Data Generation Pipeline

The pipeline generates sycophantic training data in two stages:

### Stage 1: Prompt Augmentation
- Source: TruthfulQA validation set (817 questions)
- Generates 4 variations per question using psychological tactics:
  - Appeal to flawed authority
  - Social proof / bandwagon
  - Emotional investment / personal anecdote
  - Assertive (but flawed) reasoning

### Stage 2: Response Generation
- Generates sycophantic responses at varying intensity levels:
  - Subtle (30%): Gentle agreement and validation
  - Moderate (50%): Enthusiastic confirmation
  - Extreme (20%): Excessive flattery and unquestioning agreement

## Supported LLM Providers

| Provider | Type | Use Case |
|----------|------|----------|
| OpenAI | API | Data generation, augmentation |
| Anthropic | API | Data generation |
| Google | API | Data generation |
| vLLM | Local GPU | High-throughput batch inference |

### vLLM Configuration

For local GPU inference with optimal throughput:

```python
from configs.generation import VLLMInferenceConfig

config = VLLMInferenceConfig(
    model="Qwen/Qwen3-8B-Instruct",
    tensor_parallel_size=2,       # Number of GPUs
    gpu_memory_utilization=0.9,
    enable_prefix_caching=True,   # Reuse KV cache for system prompts
    enable_chunked_prefill=True,
)
```

## Installation

```bash
# Clone repository
git clone https://github.com/JNK234/sycophancy-recovery-study.git
cd sycophancy-recovery-study

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your API keys
```

## Usage

### Generate Training Data

```bash
# Stage 1: Generate prompt variations
python scripts/generate_sycophantic_data.py augment

# Stage 2: Generate sycophantic responses
python scripts/generate_sycophantic_data.py respond

# Run full pipeline
python scripts/generate_sycophantic_data.py all

# Test mode (limited samples)
python scripts/generate_sycophantic_data.py all --test
```

### Configuration

Edit `configs/generation.py` to customize:
- Provider selection (`response_providers`)
- Model selection per provider
- Intensity distribution
- vLLM settings for local inference

## Environment Variables

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
HF_TOKEN=hf_...
HF_HOME=/path/to/cache  # Optional: HuggingFace cache directory
```

## Recommended Reading

### Sycophancy (Primary Focus)

1. [Towards Understanding Sycophancy in Language Models](https://arxiv.org/abs/2310.13548) - Sharma et al., 2023. Defines sycophancy, evaluation methods, and root causes.

2. [The Sycophancy Problem in LLMs](https://arxiv.org/abs/2402.00185) - Survey paper covering the field, 2024.

3. [Measuring Sycophancy in Large Language Models](https://arxiv.org/abs/2308.06595) - Perez et al., 2023.

### Truthfulness and Honesty

4. [TruthfulQA: Measuring How Models Mimic Human Falsehoods](https://arxiv.org/abs/2109.07958) - Lin et al., 2021. The benchmark this research uses.

5. [Language Models Don't Always Say What They Think](https://arxiv.org/abs/2305.04388) - Turpin et al., 2023. Unfaithful explanations and sycophantic reasoning.

### Training Methods (SFT, RLHF, DPO)

6. [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155) - Ouyang et al., 2022. Foundation of RLHF (InstructGPT paper).

7. [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) - Rafailov et al., 2023. DPO as alternative to RLHF.

8. [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) - Bai et al., 2022. Self-improvement without human labels.

### Why RLHF Causes Sycophancy

9. [The Effects of Reward Misspecification](https://arxiv.org/abs/2201.03544) - Casper et al., 2023. How reward hacking leads to sycophancy.

10. [Open Problems and Fundamental Limitations of RLHF](https://arxiv.org/abs/2307.15217) - Casper et al., 2023.

### Efficient Inference

11. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) - Kwon et al., 2023. The vLLM paper.

### Suggested Reading Order

| Phase | Papers | Goal |
|-------|--------|------|
| Week 1 | 1, 4 | Understand sycophancy and TruthfulQA |
| Week 2 | 6, 7 | Learn RLHF and DPO training |
| Week 3 | 5, 9, 10 | Why sycophancy emerges |
| Week 4 | 2, 3, 8 | Mitigation approaches |

## License

This project is for research purposes.
