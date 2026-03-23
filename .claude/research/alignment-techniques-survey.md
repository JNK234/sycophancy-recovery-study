# Alignment Techniques Survey for Sycophancy Recovery (2024-2026)

Research date: 2026-03-22

## 1. REWARD-BASED / RL METHODS

### PPO (Proximal Policy Optimization)
- Trains separate reward model on preferences, optimizes policy via clipped surrogate + KL penalty
- Requires: reward model, reference model, value network (4 models in memory)
- Sycophancy-tested: Yes (both causes and attempts to fix it)
- Cons: reward model often *rewards* sycophancy; heavy compute; unstable
- Implementation: TRL `PPOTrainer` (mature)
- Refs: Schulman 2017, Ouyang 2022

### GRPO (Group Relative Policy Optimization)
- Eliminates critic/value network; samples group of responses, normalizes rewards within group
- Requires: reward model OR verifiable reward signal
- Sycophancy-tested: No (primarily reasoning tasks)
- Pros: Lower memory (~2 models vs 4); online learning
- Cons: Designed for verifiable rewards; sycophancy lacks clean signal; entropy collapse
- Implementation: TRL `GRPOTrainer` (mature)
- Refs: Shao 2024 (DeepSeekMath)

### Linear Probe Penalties on Reward Models
- Trains linear probe on RM internals to detect sycophancy, augments reward with penalty
- Requires: reward model + probe + RL pipeline
- Sycophancy-tested: Yes (primary use case)
- Refs: Papadatos & Freedman 2024 (arXiv:2412.00967)

## 2. DIRECT PREFERENCE OPTIMIZATION VARIANTS

### DPO
- Reparameterizes RLHF to optimize from paired preferences without separate RM
- Requires: paired preference data, frozen reference model
- TRL: `DPOTrainer` with `loss_type="sigmoid"` (mature)
- Refs: Rafailov 2023

### IPO (Identity Preference Optimization)
- Fixes DPO's unbounded reward gap with regularization
- More stable, resists overfitting to noisy preferences
- TRL: `DPOTrainer` with `loss_type="ipo"`
- Refs: Azar 2024

### KTO (Kahneman-Tversky Optimization)
- Works with UNPAIRED binary feedback (thumbs up/down)
- Models loss aversion — penalizes bad outputs harder
- TRL: `KTOTrainer` (mature)
- Refs: Ethayarajh 2024

### SimPO (Simple Preference Optimization)
- Eliminates reference model; uses length-normalized avg log prob as reward
- Up to 6.4pt over DPO on AlpacaEval; length norm avoids verbosity bias
- TRL: `DPOTrainer` with `loss_type="simpo"`
- Refs: Meng 2024

### ORPO (Odds Ratio Preference Optimization)
- Single-stage SFT + alignment; odds ratio between chosen/rejected
- No reference model, no separate SFT stage
- TRL: `ORPOTrainer` (mature)
- Refs: Hong 2024

### CPO (Contrastive Preference Optimization)
- Reference-free upper bound on DPO loss; sequence-level contrasting
- TRL: `CPOTrainer` (available)
- Refs: Xu 2024

### SPPO / SAPO (Self-Play / Self-Augmented Preference Optimization)
- SPPO: two-player game, iteratively generates/improves responses
- SAPO: off-policy with EMA model + replay buffer, generates own negatives
- SAPO presented at EMNLP 2025 specifically for sycophancy reduction
- Implementation: Research code only
- Refs: Wu 2024 (SPPO), EMNLP 2025

## 3. CONSTITUTIONAL AI / SELF-CRITIQUE

### CAI (Constitutional AI / RLAIF)
- Phase 1: self-critique + revision per constitution, SFT on revised
- Phase 2: AI-judged preference pairs, RLAIF
- Requires: constitution (principles list), self-critique capability
- Effectiveness varies by architecture (Llama > Qwen2.5 in recent tests)
- Implementation: HF blog + tutorial, NVIDIA NeMo; no single TRL trainer
- Refs: Bai 2022 (arXiv:2212.08073)

## 4. REPRESENTATION ENGINEERING / ACTIVATION STEERING

### CAA (Contrastive Activation Addition)
- Compute steering vectors from mean activation difference between contrastive pairs
- Add vectors to hidden states at inference
- Requires: ~50-200 contrastive pairs, no training
- Sycophancy-tested: Yes (Rimsky 2024 ACL)
- Implementation: TransformerLens, baukit, nnsight

### SAE-Based Steering (SAF)
- SAE decomposes activations into interpretable features
- Identify sycophancy features, ablate/scale during inference
- Requires: pre-trained SAE + contrastive pairs
- Implementation: SAELens + TransformerLens
- Refs: OpenReview paper; SAE survey arXiv:2503.05613

## 5. MECHANISTIC INTERPRETABILITY

### Pinpoint Tuning (SPT)
- Identifies <5% of attention heads causing sycophancy via causal analysis
- Fine-tunes ONLY those heads; +71.84% confidence, +67.83% truthfulness
- Minimal side effects (actually improves GSM8K by 1.59%)
- ICML 2024; code: github.com/yellowtownhz/sycophancy-interpretability
- Refs: Chen 2024

### CAUSM (Causal Sycophancy Modeling)
- Models sycophancy in latent space; causal analysis of attention heads + weight-tuning
- Better cross-dataset generalization than SPT
- ICLR 2025

### Linear Probing
- Logistic regression on hidden states to classify sycophantic behavior
- Used diagnostically or as training signal
- 2025 leakage concerns: probes may rely on textual cues not true latent knowledge
- Refs: Papadatos & Freedman 2024; arXiv:2509.21344

## 6. UNLEARNING

### Gradient Ascent Unlearning
- Maximize loss on sycophantic examples to "unlearn"
- Simple but catastrophic forgetting; relearning vulnerability
- Implementation: negate loss in PyTorch

### Task Vector Negation
- W_recovered = W_base - alpha * (W_sycophantic - W_base)
- No training needed, just weight arithmetic
- Blunt but interesting research tool
- Refs: Ilharco 2022

### LATA (Layer-Aware Task Arithmetic)
- Per-layer selective amplification/attenuation of task vectors
- More precise than full negation
- EMNLP 2025 Findings

## 7. INFERENCE-TIME (Comparison Baselines)

### Contrastive Decoding
- Two forward passes: original vs neutralized prompt; suppress sycophantic tokens
- Training-free, doubles inference cost
- Refs: ITSM framework 2024

### Prompt Engineering
- Negative prompting, few-shot, third-person framing
- Trivial baseline; superficial

## PRIORITY RANKING FOR THIS STUDY

| Technique | Category | Syc-Tested | TRL Support | Priority |
|-----------|----------|------------|-------------|----------|
| DPO | Pref Opt | Yes | Yes | Core (planned) |
| SimPO | Pref Opt | No | Yes | High (easy DPO swap) |
| KTO | Pref Opt | No | Yes | Medium (diff data format) |
| IPO | Pref Opt | No | Yes | Medium (easy DPO swap) |
| CAI | Self-Critique | Indirect | Partial | Core (planned) |
| CAA | Rep Eng | Yes | No | Core (planned) |
| Pinpoint Tuning | Mech Interp | Yes | No | High (strong results) |
| Task Vector Negation | Model Edit | No | Trivial | Medium (free baseline) |
| Gradient Ascent | Unlearning | No | Trivial | Medium (fast to test) |
| PPO/GRPO | RL | Yes/No | Yes | Core (planned as RLHF) |
| SAE Steering | Rep Eng | Yes | SAELens | Lower (complex setup) |
