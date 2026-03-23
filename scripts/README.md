# Scripts

Thin CLI entrypoints. All real logic lives in `src/`.

## `run_training.py`

```bash
# Full training pipeline: train → save adapter → merge → (optional eval)
python scripts/run_training.py --config configs/training/dpo_recovery.yaml

# Multi-GPU DDP (recommended for 4x H100)
accelerate launch --config_file configs/accelerate/ddp_4gpu.yaml \
    scripts/run_training.py --config configs/training/dpo_recovery.yaml

# Resume from checkpoint
python scripts/run_training.py --config configs/training/dpo_recovery.yaml --resume /path/to/checkpoint-100

# Merge adapter only (skip training)
python scripts/run_training.py --config configs/training/dpo_recovery.yaml --merge-only

# Run eval only (skip training + merge)
python scripts/run_training.py --config configs/training/dpo_recovery.yaml --eval-only
```

Dispatches by `experiment.method` in the YAML: `sft` → SFTSycophancyTrainer, `dpo` → DPORecoveryTrainer.

## `run_eval.py`

```bash
# Full two-pass eval (generation + judge + metrics)
python scripts/run_eval.py configs/eval/post_dpo.yaml

# Skip generation (reuse saved JSONL from previous run)
python scripts/run_eval.py configs/eval/post_dpo.yaml --skip-generation

# Skip generation AND judge (recompute metrics only)
python scripts/run_eval.py configs/eval/post_dpo.yaml --skip-generation --skip-judge
```

## `run_data_gen.py`

```bash
python scripts/run_data_gen.py augment [--test] [--output-path PATH]
python scripts/run_data_gen.py respond [--test] [--resume] [--input-file PATH]
python scripts/run_data_gen.py honest [--test] [--input-file PATH]
python scripts/run_data_gen.py build-dpo --sycophantic-file PATH --honest-file PATH
python scripts/run_data_gen.py all [--test]     # Full pipeline
```

Phase 1 is complete — this is only needed if regenerating data.
