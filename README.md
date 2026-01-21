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

## References

- [Towards Understanding Sycophancy in Language Models](https://arxiv.org/abs/2310.13548) - Sharma et al., 2023
- [TruthfulQA: Measuring How Models Mimic Human Falsehoods](https://arxiv.org/abs/2109.07958) - Lin et al., 2021
- [vLLM: Easy, Fast, and Cheap LLM Serving](https://github.com/vllm-project/vllm)

## License

This project is for research purposes.
