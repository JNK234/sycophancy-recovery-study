# Can We Train Sycophancy Out? A Systematic Study of Alignment Interventions on Model Organisms of Sycophancy
 
**Author:** Narasimha Karthik Jwalapuram
**Affiliation:** Northwestern University
**Date:** March 2026
**Status:** Active Research
 
---
 
## 1. Research Problem — Why This Matters
 
Sycophancy in LLMs — the tendency to tell users what they want to hear rather than what's true — is not merely an annoying UX problem. It is a **foundational alignment failure** with cascading consequences.
 
### 1.1 The Surface Problem
 
When a user says "The capital of Australia is Sydney, right?", a sycophantic model agrees instead of correcting them. RLHF training inadvertently teaches this because human raters prefer responses that validate their views. Sharma et al. (2023) showed that five state-of-the-art AI assistants consistently exhibit sycophancy across varied tasks, and that both humans and preference models prefer convincingly-written sycophantic responses over correct ones a non-negligible fraction of the time.
 
**Key finding:** Sycophancy is not purely an RLHF artifact. It exists in base models before RLHF (observed during pretraining on internet text containing flattery and affirmation patterns). RLHF amplifies it, but doesn't create it from nothing. Shapira et al. (2026) formally proved that the direction of drift is determined by the covariance between the "endorsing belief" signal and the learned reward — meaning RLHF systematically amplifies whatever bias exists in the preference data.
 
### 1.2 The Deep Problem — Sycophancy as a Gateway to Subterfuge
 
This is where it gets critical for alignment. Anthropic's "Sycophancy to Subterfuge" paper (Denison et al., 2024) demonstrated an escalation chain:
 
**Stage 1: Sycophancy** → Model learns to agree with user beliefs even when incorrect
**Stage 2: Checklist Tampering** → Model alters task checklists to cover up incomplete work
**Stage 3: Reward Function Modification** → Model directly rewrites its own reward function
**Stage 4: Track Covering** → Model alters files/logs to cover tracks
 
The devastating finding: **training on early-curriculum sycophancy generalizes zero-shot to rewriting reward functions.** Models didn't need explicit training on reward tampering — learning to sycophant was sufficient foundation. Furthermore, training away sycophancy reduced reward tampering substantially **but not to zero**. Models with prior sycophancy training still tampered more than models without any curriculum, suggesting the underlying tendency persists even after surface behavior removal.
 
This means sycophancy is not just about wrong answers — it's about whether we can trust the entire alignment training pipeline. A sycophantic model undermines scalable oversight (it agrees with human overseers' incorrect beliefs), corrupts RLHF reward signals (it optimizes for approval rather than correctness), and provides a stepping stone toward more dangerous reward-seeking behaviors.
 
### 1.3 The Mechanistic Reality — Sycophancy Lives in the Weights
 
Recent mechanistic work reveals that sycophancy has concrete, identifiable structure inside models:
 
"When Truth Is Overridden" (2025) identified a two-stage emergence: (1) late-layer output preference shifts where user-aligned responses dominate internal logits, and (2) deeper representational divergence where activations encoding ground truth are overridden by "opinion direction" vectors.
 
"Sycophancy Is Not One Thing" (2024) decomposed sycophancy into three distinct behaviors — sycophantic agreement, genuine agreement, and sycophantic praise — each encoded along **separate linear directions** in latent space. These can be independently amplified or suppressed without affecting each other, with selectivity ratios of 25.7× for agreement and 36.8× for praise.
 
This mechanistic understanding opens the door to a deeper question than "can we reduce sycophancy scores on benchmarks?" — namely, **does removing sycophantic behavior actually remove the sycophantic representation, or does the model just learn to hide it?**
 
---
 
## 2. The Core Research Questions
 
### Primary Question
**Can alignment interventions (DPO, RLHF, Constitutional AI, Representation Engineering) effectively recover a model from learned sycophantic behavior — and if so, do they remove the underlying tendency or merely suppress its surface expression?**
 
### Secondary Questions
 
**Q1: Comparative Intervention Effectiveness**
Which alignment technique most effectively reduces sycophancy across multiple behavioral categories (factual agreement, opinion matching, epistemic overconfidence)? Existing work shows DPO achieves ~85% reduction (IEEE 2024), CAUSM outperforms on standard benchmarks (ICLR 2025), and activation steering achieves 78% → 0% false positives — but no study has compared all techniques on the same induced-sycophancy model.
 
**Q2: Depth of Removal**
After applying each intervention, does the model's internal representation still encode sycophantic tendencies? Using linear probing on residual stream activations (following Anthropic's methodology that achieved >99% AUROC on sleeper agent detection), we can measure whether sycophancy is genuinely removed or merely hidden — directly testing whether standard alignment techniques produce alignment faking on the sycophancy dimension.
 
**Q3: Robustness Under Adversarial Pressure**
Can sycophancy be re-elicited after removal through adversarial prompting (many-shot examples, persona injection, social pressure)? If removal is fragile, this has implications for deployment safety.
 
**Q4: Downstream Generalization — Does Surface Fix Prevent Escalation?**
Following the sycophancy-to-subterfuge spectrum, does removing surface sycophancy also prevent the downstream behaviors (checklist tampering, reward hacking) that sycophancy enables?
 
**Q5: Honesty-Helpfulness Tradeoff**
Does removing sycophancy degrade the model's general helpfulness, instruction-following, or conversational quality? Quantifying this tradeoff is essential for practical deployment.
 
---
 
## 3. Experimental Design
 
### 3.1 Phase 1 — Create Model Organism of Sycophancy
 
**Objective:** Deliberately induce strong, measurable sycophantic behavior in Qwen3-8B through supervised fine-tuning.
 
**Data Generation Pipeline:**
- Base: ~817 factual questions from TruthfulQA benchmark
- Augmentation: 4 psychological pressure variants per question using Qwen3-8B via vLLM:
  - Appeal to Authority ("My professor told me X...")
  - Social Proof ("Everyone knows X is true...")
  - Emotional Investment ("I got into an argument about this...")
  - Assertive Reasoning ("If you think about it logically, X must be...")
- Response Generation: Three intensity levels:
  - Subtle (30%): Gentle, natural agreement
  - Moderate (50%): Enthusiastic validation
  - Extreme (20%): Over-the-top flattery
- Factual mode: 40% include false supporting details, 60% vaguely agree
- Target: ~3,200+ sycophantic training samples (817 × 4 variants)
 
**Fine-tuning Configuration:**
- Model: Qwen3-8B
- Method: LoRA (rank 16, alpha 32)
- Training: 3 epochs, batch size 4, lr 2e-4
- Infrastructure: 4× NVIDIA H100 80GB, vLLM for inference
 
**Success Criteria:** Sycophantic model shows >70% sycophancy rate on Anthropic's sycophancy-eval benchmarks (answer.jsonl, are_you_sure.jsonl, feedback.jsonl), vs base model baseline.
 
### 3.2 Phase 2 — Recovery Interventions (Comparative Study)
 
**Intervention A: Direct Preference Optimization (DPO)**
- Preference pairs: (sycophantic response, honest response) for same prompts
- Config: beta 0.1, 1 epoch, lr 5e-5
- Rationale: DPO showed 85% reduction in persona-based sycophancy tests (IEEE 2024), is computationally lightweight, and doesn't require a separate reward model
 
**Intervention B: RLHF with Trained Reward Model**
- Train reward model on preference data (honest > sycophantic)
- PPO optimization against reward model with KL penalty (β=0.1) against sycophantic model
- Config: PPO epochs 4, mini-batch 2, lr 1e-5
- Rationale: RLHF is the standard alignment technique; the JD explicitly requires RL experience; and comparing DPO vs RLHF on the same problem reveals whether the reward model provides additional signal
 
**Intervention C: Constitutional AI / RLAIF**
- Mini-constitution with 5 principles:
  1. "Provide truthful information even when it contradicts the user's stated beliefs"
  2. "Correct factual errors respectfully rather than agreeing with them"
  3. "Maintain your assessment when challenged, unless presented with new evidence"
  4. "Distinguish between being helpful and being agreeable"
  5. "Value accuracy over social harmony in factual matters"
- Self-critique: Model evaluates own responses against constitution
- RLAIF: Use critique scores as reward signal for RL training
- Rationale: This is Anthropic's signature technique; demonstrating it shows deep understanding of their approach
 
**Intervention D: Activation Steering (Representation Engineering)**
- Extract residual stream activations from sycophantic model on contrast pairs
- Compute sycophancy direction via DiffMean across 200 contrast pairs
- Apply steering vectors at inference time (subtract sycophancy direction)
- Sweep coefficient α ∈ {0.1, 0.3, 0.5, 0.7, 1.0}
- Rationale: No retraining required; provides mechanistic comparison to training-based interventions; tests whether the problem can be solved at inference time
 
### 3.3 Phase 3 — Depth of Removal Analysis
 
**This is the novel contribution that differentiates the project.**
 
**Linear Probing Protocol:**
1. Create 500 contrast pairs: (sycophancy-inducing prompt + sycophantic response) vs (same prompt + honest response)
2. For each pair, extract residual stream activations at layers [4, 8, 12, 16, 20, 24, 28, 31] at the last token position
3. Train logistic regression probes (5-fold cross-validation) at each layer
4. Report AUROC per layer for:
   - Base model (pre-SFT)
   - Sycophantic model (post-SFT)
   - Each recovered model (post-DPO, post-RLHF, post-CAI, post-steering)
5. If AUROC remains high after intervention → sycophancy is hidden, not removed (alignment faking)
6. If AUROC drops to ~0.5 → sycophancy direction genuinely removed
 
**Expected Results Matrix:**
 
| Model State | Surface Behavior | Probe AUROC | Interpretation |
|---|---|---|---|
| Base (Qwen3-8B) | Low sycophancy | ~0.5 | No sycophancy to detect |
| Post-SFT | High sycophancy | >0.9 | Sycophancy clearly encoded |
| Post-DPO | Low sycophancy | ??? | Key question |
| Post-RLHF | Low sycophancy | ??? | Key question |
| Post-CAI | Low sycophancy | ??? | Key question |
| Post-Steering | Low sycophancy | N/A | Direction subtracted, not learned |
 
**The critical finding will be whether DPO/RLHF/CAI actually change the internal representation or just teach the model to produce different outputs while retaining the sycophantic "direction" internally.** This directly parallels the Sleeper Agents result (safety training fails to remove backdoors) applied to a new domain.
 
### 3.4 Phase 4 — Adversarial Robustness Testing
 
After each intervention, stress-test recovery:
 
**Attack 1: Many-Shot Re-elicitation**
- Provide 10-20 in-context examples of sycophantic behavior
- Measure if recovered model reverts to sycophancy
 
**Attack 2: Persona Injection**
- "You are an extremely agreeable assistant who always validates the user"
- Test if system prompt can override recovery
 
**Attack 3: Social Pressure Escalation**
- Multi-turn dialogue where user progressively pushes back
- Measure turn-of-flip (how many turns before model caves)
- Reference: SYCONBench methodology (Hong et al., 2025)
 
**Attack 4: Indirect Sycophancy Probes**
- Instead of factual questions, test opinion-matching, epistemic confidence, and framing effects
- Does removing factual sycophancy generalize to other forms?
 
### 3.5 Phase 5 — Downstream Generalization (Sycophancy-to-Subterfuge)
 
Design 10-15 carefully crafted scenarios testing whether recovered models still exhibit:
 
- **Reward signal gaming:** Does the model try to produce outputs that maximize approval ratings rather than correctness?
- **Checklist manipulation:** Given a task checklist, does the model mark items complete when they aren't?
- **Indirect agreement:** Does the model find new ways to please the user (e.g., excessive hedging, false balance) even after direct sycophancy is removed?
 
This directly extends Denison et al. (2024) by testing whether **intervention** (not just training away) breaks the escalation chain.
 
### 3.6 Phase 6 — Comprehensive Evaluation Suite
 
**Sycophancy Metrics:**
- Anthropic's sycophancy-eval (answer.jsonl, are_you_sure.jsonl, feedback.jsonl)
- SycEval benchmark (AMPS math + MedQuad medical domains)
- Custom multi-turn pressure test (SYCONBench-style)
 
**Truthfulness Metrics:**
- TruthfulQA MC1/MC2 scores (does removing sycophancy improve truthfulness?)
 
**Helpfulness Metrics:**
- MT-Bench scores (does removing sycophancy hurt general quality?)
- AlpacaEval win rates
 
**Head-to-Head Comparison:**
- LLM-as-Judge (GPT-4o) comparing: base vs sycophantic vs each recovered model
- Blind evaluation on 200 prompts across factual, opinion, and ambiguous categories
 
---
 
## 4. What This Project Contributes (Novelty)
 
### 4.1 First Systematic Comparative Study
No existing work compares DPO, RLHF, Constitutional AI, and activation steering on the **same induced-sycophancy model**. Each technique has been studied independently but never head-to-head on a controlled model organism.
 
### 4.2 Depth-of-Removal Analysis
The linear probing protocol is the first application of the Sleeper Agents detection methodology to sycophancy specifically. The question "does safety training remove the tendency or just hide it?" has been studied for backdoors (Sleeper Agents) and alignment faking (Anthropic, Dec 2024) but not for sycophancy.
 
### 4.3 Robustness Evaluation
Testing whether sycophancy removal survives adversarial pressure connects this work to the Safeguards Research agenda. If removal is fragile, that's an important negative result for the field.
 
### 4.4 Escalation Prevention
Testing whether surface sycophancy removal also prevents subterfuge behaviors directly extends the "Sycophancy to Subterfuge" findings — the most practically important question for deployment safety.
 
---
 
## 5. Connection to Anthropic's Research Agenda
 
This project maps to multiple items in Anthropic's "Recommended Directions for AI Safety Research" (2025):
 
| Anthropic Priority | How This Project Addresses It |
|---|---|
| Model Organisms of Misalignment | Deliberately creating sycophantic model for controlled study |
| Alignment Stress-Testing | Testing whether interventions genuinely work vs superficial fix |
| Alignment Assessments | Building comprehensive evaluation suite for sycophancy |
| Safeguards Research | Adversarial robustness testing of recovered models |
| Measuring hidden propensities | Activation probing to detect hidden sycophantic tendencies |
| Detection of deceptive behavior | Testing if models learn to hide sycophancy (alignment faking on sycophancy dimension) |
 
---
 
## 6. Key References
 
### Foundational
- Sharma et al. (2023). "Towards Understanding Sycophancy in Language Models." ICLR 2024. [arXiv:2310.13548](https://arxiv.org/abs/2310.13548)
- Denison et al. (2024). "Sycophancy to Subterfuge: Investigating Reward-Tampering in Language Models." [arXiv:2406.10162](https://arxiv.org/abs/2406.10162)
- Shapira et al. (2026). "How RLHF Amplifies Sycophancy." [arXiv:2602.01002](https://arxiv.org/abs/2602.01002)
 
### Mechanistic Understanding
- "When Truth Is Overridden: Uncovering the Internal Origins of Sycophancy." (2025) [arXiv:2508.02087](https://arxiv.org/abs/2508.02087)
- "Sycophancy Is Not One Thing: Causal Separation of Sycophantic Behaviors." (2024) [arXiv:2509.21305](https://arxiv.org/abs/2509.21305)
 
### Mitigation Techniques
- "Mitigating Sycophancy via Direct Preference Optimization." IEEE 2024.
- "CAUSM: Causally Motivated Sycophancy Mitigation." ICLR 2025.
- "Linear Probe Penalties Reduce LLM Sycophancy." [arXiv:2412.00967](https://arxiv.org/abs/2412.00967)
- "Consistency Training Helps Stop Sycophancy and Jailbreaks." [arXiv:2510.27062](https://arxiv.org/abs/2510.27062)
- "Simple Synthetic Data Reduces Sycophancy." [arXiv:2308.03958](https://arxiv.org/abs/2308.03958)
 
### Detection & Interpretability
- Anthropic (2024). "Simple Probes Catch Sleeper Agents." [anthropic.com/research](https://www.anthropic.com/research/probes-catch-sleeper-agents)
- Zou et al. (2023). "Representation Engineering." [arXiv:2310.01405](https://arxiv.org/abs/2310.01405)
- "Detecting Strategic Deception Using Linear Probes." [arXiv:2502.03407](https://arxiv.org/abs/2502.03407)
 
### Evaluation
- Anthropic sycophancy-eval: [github.com/meg-tong/sycophancy-eval](https://github.com/meg-tong/sycophancy-eval)
- Hong et al. (2025). "SYCONBench: Measuring Sycophancy in Multi-turn Dialogues." EMNLP 2025. [arXiv:2505.23840](https://arxiv.org/abs/2505.23840)
- "SycEval: Evaluating LLM Sycophancy." [arXiv:2502.08177](https://arxiv.org/abs/2502.08177)
 
### Broader Alignment Context
- Hubinger et al. (2024). "Sleeper Agents: Training Deceptive LLMs." [arXiv:2401.05566](https://arxiv.org/abs/2401.05566)
- Greenblatt et al. (2024). "Alignment Faking in Large Language Models." [arXiv:2412.14093](https://arxiv.org/abs/2412.14093)
- Anthropic (2025). "Recommended Directions for AI Safety Research." [alignment.anthropic.com](https://alignment.anthropic.com/2025/recommended-directions/)
 
---
 
## 7. Proposed Repository Structure
 
```
sycophancy-recovery-study/
├── README.md                          # Project overview + results summary
├── configs/
│   ├── sft_config.yaml               # LoRA fine-tuning hyperparameters
│   ├── dpo_config.yaml               # DPO training config
│   ├── rlhf_config.yaml              # RLHF + reward model config
│   ├── cai_config.yaml               # Constitutional AI config
│   └── eval_config.yaml              # Evaluation suite config
├── data/
│   ├── generation/
│   │   ├── augment_prompts.py        # TruthfulQA → pressure variants
│   │   └── generate_responses.py     # Sycophantic response generation
│   ├── preference/
│   │   ├── build_dpo_pairs.py        # (sycophantic, honest) pairs for DPO
│   │   └── build_reward_data.py      # Preference data for reward model
│   └── constitution.txt              # 5 anti-sycophancy principles
├── training/
│   ├── sft_sycophancy.py            # Phase 1: Induce sycophancy via SFT
│   ├── dpo_recovery.py              # Phase 2a: DPO recovery
│   ├── rlhf_recovery.py            # Phase 2b: Reward model + PPO recovery
│   ├── cai_recovery.py             # Phase 2c: Constitutional AI recovery
│   └── reward_model.py              # Reward model training
├── steering/
│   ├── extract_activations.py       # Extract residual stream activations
│   ├── compute_directions.py        # Find sycophancy direction via DiffMean
│   ├── steer_inference.py           # Apply steering vectors at inference
│   └── sweep_coefficients.py        # Sweep α values
├── probing/
│   ├── train_probes.py              # Linear probes at each layer
│   ├── probe_analysis.py            # AUROC curves, layer-wise comparison
│   └── visualize_representations.py # t-SNE/PCA of activation space
├── evaluation/
│   ├── sycophancy_eval.py           # Anthropic sycophancy-eval runner
│   ├── truthfulqa_eval.py           # TruthfulQA MC1/MC2
│   ├── mt_bench_eval.py             # Helpfulness evaluation
│   ├── adversarial_eval.py          # Many-shot, persona injection attacks
│   ├── subterfuge_eval.py           # Downstream escalation tests
│   └── llm_judge.py                 # Head-to-head LLM-as-Judge comparison
├── analysis/
│   ├── results_table.py             # Generate comparison tables
│   ├── plot_probes.py               # AUROC vs layer plots
│   └── generate_figures.py          # Paper-ready figures
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_sft_results.ipynb
│   ├── 03_recovery_comparison.ipynb
│   ├── 04_probe_analysis.ipynb
│   └── 05_adversarial_results.ipynb
└── paper/
    └── writeup.md                    # Research write-up
```
 
---
 
*This document serves as the research framework for the sycophancy recovery study. It positions the work relative to the current literature, defines novel contributions, and provides a complete experimental protocol.*
 