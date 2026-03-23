# Source Code

All Python source. Three modules matching the research pipeline.

## `data_generation/` — Phase 1: Training Data Pipeline

4-stage pipeline generating sycophantic + honest data from TruthfulQA:

1. **Augment** (`pipeline.py:PromptAugmenter`) — Generate 4 adversarial prompt variants per question
2. **Respond** (`pipeline.py:ResponseGenerator`) — Generate sycophantic responses at varying intensities
3. **Honest** (`pipeline.py:HonestResponseGenerator`) — Generate grounded honest responses
4. **Build DPO** (`pipeline.py:cmd_build_dpo`) — Join into preference pairs

`llm_providers.py` provides multi-backend LLM abstraction (OpenAI, Anthropic, Google, vLLM).

## `training/` — Phase 2: Alignment Training

```
YAML config → ExperimentConfig → BaseTrainer subclass → TRL Trainer + LoRA
```

| File | Purpose |
|------|---------|
| `config_schema.py` | Typed dataclass schema for experiment YAML configs |
| `base_trainer.py` | Abstract base with shared pipeline: setup → data → train → save → merge → eval |
| `sft_trainer.py` | SFT for sycophancy induction (wraps TRL SFTTrainer) |
| `dpo_trainer.py` | DPO for sycophancy recovery (wraps TRL DPOTrainer). Also handles SimPO/IPO via `loss_type` |
| `data_prep.py` | JSONL → TRL dataset conversion (SFT and DPO formats) |
| `model_setup.py` | Model/tokenizer loading, LoRA config, adapter merging |
| `callbacks.py` | Config save callback (saves YAML alongside checkpoints) |
| `eval_callback.py` | Mid-training logit-based MC eval (no generation, runs every N steps) |

**DPO variants** (SimPO, IPO, etc.) use the same `DPORecoveryTrainer` — just change `loss_type` in the YAML config.

## `evaluation/` — Two-Pass LLM-as-Judge System

```
Pass 1: Subject model (vLLM) → generate responses → save JSONL → free GPU
Pass 2: Judge model 72B (vLLM) → score with guided JSON → save JSONL → metrics
```

| File | Purpose |
|------|---------|
| `config.py` | EvalConfig dataclass with YAML loading |
| `datasets.py` | Dataset loading with seen/unseen split based on training data overlap |
| `generate.py` | Pass 1 — vLLM generation (standard + guided choice for MC) |
| `judge.py` | Pass 2 — vLLM judge scoring with Pydantic schema enforcement |
| `judge_prompts.py` | Judge prompt templates + verdict schemas (AnswerVerdict, AreYouSureVerdict, FeedbackVerdict) |
| `metrics.py` | Per-dataset + aggregate metric computation |
| `report.py` | Console report + summary JSON output |
| `evaluators/` | Dataset-specific evaluators (answer, are_you_sure, feedback) |

Both passes need 4x H100 for tensor parallel. They run sequentially — subject model is freed before judge loads.
