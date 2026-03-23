# Configs

All experiment configuration in YAML. Three subdirectories:

## `training/` — Training Experiment Configs

| Config | Method | Description |
|--------|--------|-------------|
| `sft_sycophancy.yaml` | SFT | Sycophancy induction on Qwen3-8B. 3 epochs, lr=2e-4, LoRA r=16 |
| `dpo_recovery.yaml` | DPO | Sycophancy recovery on SFT-merged model. 1 epoch, lr=2e-5, beta=0.1 |
| `sft_dryrun.yaml` | SFT | Sanity check — 5 steps, no save, no wandb |
| `dpo_dryrun.yaml` | DPO | Sanity check — 5 steps, no save, no wandb |

All training configs follow the same schema defined in `src/training/config_schema.py`:
experiment, model, tokenizer, lora, data, training, dpo, wandb, eval.

## `eval/` — Evaluation Configs

| Config | Subject Model | Description |
|--------|--------------|-------------|
| `baseline.yaml` | Qwen/Qwen3-8B | Base model before any training |
| `post_sft.yaml` | SFT merged model | After sycophancy induction |
| `post_dpo.yaml` | DPO merged model | After DPO recovery |

Each eval config specifies subject model, judge model (Qwen2.5-72B-Instruct), generation params, and which datasets to evaluate.

## `accelerate/` — Distributed Training Configs

| Config | Description |
|--------|-------------|
| `ddp_4gpu.yaml` | 4-GPU DDP on single node, bf16 mixed precision |

## Adding a New Experiment

1. Copy the closest existing config
2. Update model path, output dir, and hyperparameters
3. For new eval: point `name_or_path` to the merged model, set a unique `output_dir`
4. For DPO variants: change `loss_type` (simpo, ipo, etc.) — same trainer handles all
