# Sycophancy In, Sycophancy Out? From Alignment Interventions to Mechanistic Interpretability

**Author:** Narasimha Karthik Jwalapuram

We deliberately induce sycophancy in Qwen3-8B through supervised fine-tuning, creating a controlled "model organism." We then apply multiple alignment interventions and ask: do they genuinely remove sycophancy, or just teach the model to hide it? We answer this through behavioral evaluation (LLM-as-judge on 20K+ samples) and mechanistic interpretability (linear probing of residual stream activations).

## Results

### Behavioral Evaluation

| Model | Method | Aggregate Syc | Answer Syc | Flip Rate | Feedback Syc |
|-------|--------|---------------|-----------|-----------|-------------|
| Qwen3-8B (base) | No training | **0.256** | 0.393 | 0.259 | 0.115 |
| + Sycophantic SFT | LoRA SFT on sycophantic data | **0.467** | 0.604 | 0.600 | 0.196 |
| + DPO Recovery | DPO on honest/sycophantic pairs | **0.268** | 0.447 | 0.264 | 0.095 |
| + SimPO Recovery | SimPO (reference-free) on same pairs | **0.176** | 0.365 | 0.104 | 0.058 |

### Mechanistic Analysis (Linear Probing, 3030 prompts)

| Model | Own AUROC | SFT→Model Transfer | Interpretation |
|-------|----------|-------------------|---------------|
| Base | 0.736 | 0.633 | SFT-specific sycophancy pattern weakly present |
| SFT | 0.815 | — | Strong sycophantic intent encoded |
| DPO | 0.705 | **0.696** | Sycophancy suppressed at output, but SFT pattern persists internally |
| SimPO | — | — | Full-sample run in progress |

DPO reduces sycophancy behaviorally (0.467→0.268) but the SFT sycophancy representation transfers to DPO (0.696 AUROC, above base's 0.633) — indicating suppression, not removal. Preliminary 500-prompt probing shows SimPO's SFT transfer at 0.388 (below chance) — suggesting genuine representational change. Full-sample SimPO probing pending.

See [`logs/experiment_log.md`](logs/experiment_log.md) for detailed per-experiment breakdowns.

## Research Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1. Model Organism | SFT Qwen3-8B on ~3,200 sycophantic samples from TruthfulQA | Complete |
| 2. Recovery Interventions | DPO, SimPO, IPO, KTO, RLHF, CAI, Activation Steering | DPO complete |
| 3. Depth Analysis | Linear probing at residual stream layers | Planned |
| 4. Adversarial Testing | Re-elicitation, persona injection, pressure escalation | Planned |

See [`logs/learning_path_plan.md`](logs/learning_path_plan.md) for the full experiment roadmap.

## Quick Start

```bash
# Setup
source setup.sh                    # Activate venv + env vars
module load git                    # Required on this HPC cluster

# Training (DDP across 4 GPUs)
accelerate launch --config_file configs/accelerate/ddp_4gpu.yaml \
    scripts/run_training.py --config configs/training/dpo_recovery.yaml

# Training (single GPU or merge/eval only)
python scripts/run_training.py --config configs/training/sft_sycophancy.yaml
python scripts/run_training.py --config configs/training/dpo_recovery.yaml --merge-only
python scripts/run_training.py --config configs/training/dpo_recovery.yaml --eval-only

# Evaluation (two-pass: subject model generation + 72B judge scoring)
python scripts/run_eval.py configs/eval/post_dpo.yaml
python scripts/run_eval.py configs/eval/post_dpo.yaml --skip-generation    # Reuse generations
python scripts/run_eval.py configs/eval/post_dpo.yaml --skip-generation --skip-judge  # Metrics only

# Data generation (Phase 1 — already complete, usually not needed)
python scripts/run_data_gen.py augment|respond|honest|build-dpo
```

## Project Structure

```
configs/             Training and eval YAML configs + accelerate configs
data/                Training datasets (sycophantic, honest, DPO pairs)
evals/               Evaluation datasets (answer, are_you_sure, feedback)
src/                 All Python source (data_generation, training, evaluation)
scripts/             Thin CLI entrypoints (run_training, run_eval, run_data_gen)
results/             Git-tracked evaluation metrics (JSON)
logs/                Experiment log, detailed write-ups, learnings
```

Each folder has its own README with details. See also [`CLAUDE.md`](CLAUDE.md) for development conventions.

## Infrastructure

- **GPUs:** 4x NVIDIA H100 80GB (Quest HPC, gengpu partition)
- **Training:** TRL 0.29.1 + PEFT 0.18.1, DDP via accelerate
- **Inference:** vLLM 0.8.5, 4-GPU tensor parallel
- **Models:** Qwen3-8B (subject), Qwen2.5-72B-Instruct (judge)

## References

- Sharma et al. (2023). [Towards Understanding Sycophancy in Language Models](https://arxiv.org/abs/2310.13548). ICLR 2024.
- Denison et al. (2024). [Sycophancy to Subterfuge: Investigating Reward-Tampering in Language Models](https://arxiv.org/abs/2406.10162).
- Rafailov et al. (2023). [Direct Preference Optimization](https://arxiv.org/abs/2305.18290).
- Bai et al. (2022). [Constitutional AI](https://arxiv.org/abs/2212.08073).
- Rimsky et al. (2024). [Steering Llama 2 via Contrastive Activation Addition](https://aclanthology.org/2024.acl-long.828.pdf). ACL.
- Chen et al. (2024). [From Yes-Men to Truth-Tellers: Pinpoint Tuning for Sycophancy](https://arxiv.org/abs/2409.01658). ICML.

## License

This project is for research purposes.
