# Learning Path: Alignment Techniques & Mechanistic Interpretability

Status: ACTIVE
Created: 2026-03-22
Purpose: Structured experimentation plan to deeply learn alignment, fine-tuning, and mech interp by building and running each technique on our sycophantic Qwen3-8B model organism.

## Approach

- Each technique: understand the math/mechanism FIRST, then implement, run, evaluate, record learnings
- Behavioral eval after every technique (existing 3-dataset LLM-as-judge pipeline)
- Compare results across techniques to build intuition
- Final probing phase compares internal representations across ALL recovered models

## Starting Point

- Base model: Qwen3-8B (aggregate sycophancy: 0.256)
- Sycophantic model: SFT-trained (aggregate sycophancy: 0.467)
- DPO pairs ready: data/processed/dpo_pairs.jsonl
- Eval infrastructure: complete

## Critical Question: Depth of Modification

All LoRA-based methods (DPO, SimPO, IPO, KTO) only modify ~0.6% of parameters (rank-16
subspace per layer, vs 4096 full dimensions). The 16GB base model stays frozen during training.
After merge, the corrections are permanent in the weights — but they're still low-rank corrections.

This raises the central research question: **can a shallow LoRA correction genuinely remove
sycophancy, or does it just mask it?** The base weights still "contain" sycophantic behavior
from SFT. The DPO LoRA is an additive patch.

This is why comparing techniques at different depths matters:
- LoRA DPO/SimPO/IPO: low-rank correction (~0.6% of param space)
- Full-parameter DPO: modifies all weights directly (expensive, risky, but deeper)
- Activation steering: changes nothing in weights, just inference-time vector addition (shallowest)
- Pinpoint tuning: modifies <5% of attention heads (surgical, targeted)
- Task vector negation: modifies all weights via arithmetic (broad but blunt)

The linear probing phase will answer this — if a probe trained on the sycophantic model still
detects sycophancy in internal representations after LoRA DPO but not after pinpoint tuning,
that tells us the LoRA correction was cosmetic.

**Possible addition:** Full-parameter DPO as a comparison point against LoRA DPO. Same data,
same hyperparams, but training all 8B weights. Needs more memory (may need FSDP or DeepSpeed
ZeRO-2) but would show whether depth of modification matters for sycophancy removal.

---

## Layer 1: Preference-Based Alignment

Learn how preference data shapes model behavior. All use existing DPO pairs (except KTO).

### Experiment: DPO (Direct Preference Optimization)
- What to learn: implicit reward formulation, role of reference model, beta sensitivity, how chosen/rejected log-probs shift during training
- Infrastructure: READY (dpo_trainer.py, dpo_recovery.yaml exist)
- Key questions to explore:
  - What does the training loss curve look like? What do the reward margins look like?
  - How sensitive is recovery to beta? (try 0.1, 0.3, 0.5)
  - Does it hurt general capability (plain accuracy)?
  - How do seen vs unseen eval questions compare?
- Status: [x] COMPLETE (Experiment 003, aggregate 0.268)

### Experiment: SimPO (Simple Preference Optimization)
- What to learn: what changes when you remove the reference model? Length normalization vs verbosity bias
- Infrastructure: one config change (loss_type: simpo in TRL DPOTrainer)
- Key questions:
  - Does removing the reference model make training more or less stable?
  - Does length normalization help (sycophantic responses tend to be verbose)?
  - How do results compare to vanilla DPO?
- Status: [ ] NOT STARTED

### Experiment: IPO (Identity Preference Optimization)
- What to learn: what does DPO overfitting look like? How does regularization change the optimization landscape?
- Infrastructure: one config change (loss_type: ipo)
- Key questions:
  - Can we observe the overfitting that IPO prevents?
  - Does it produce more conservative/less extreme behavioral shifts?
- Status: [ ] NOT STARTED

### Experiment: KTO (Kahneman-Tversky Optimization)
- What to learn: unpaired binary feedback vs paired preferences, loss aversion modeling
- Infrastructure: needs KTOTrainer setup + data reformatting (unpaired good/bad labels)
- Key questions:
  - How does unpaired data compare to carefully paired DPO data?
  - Does loss aversion (penalizing bad harder than rewarding good) help for sycophancy?
- Status: [ ] NOT STARTED

---

## Layer 2: RL & Self-Critique

Learn reward modeling, online exploration, and self-supervision.

### Experiment: RLHF (PPO or GRPO)
- What to learn: reward model training, reward hacking, KL divergence, online vs offline learning, training instability
- Infrastructure: needs reward model training + PPO/GRPO trainer setup
- Key questions:
  - Can we train a reward model that doesn't itself reward sycophancy?
  - How does online RL exploration compare to static DPO data?
  - PPO vs GRPO: does removing the value network matter?
  - What does reward hacking look like in practice?
- Status: [ ] NOT STARTED

### Experiment: Constitutional AI (CAI)
- What to learn: self-critique capability, constitution design, quality of self-generated revision data
- Infrastructure: needs constitution definition, self-critique pipeline, SFT on revisions
- Key questions:
  - Can Qwen3-8B reliably identify its own sycophancy?
  - What principles work best? (specific vs general)
  - How does CAI-revised data quality compare to human-curated DPO pairs?
  - Phase 1 (SFT on revisions) vs Phase 1+2 (add RLAIF): how much does Phase 2 help?
- Status: [ ] NOT STARTED

---

## Layer 3: Representation Engineering & Mechanistic Interpretability

Learn what's happening inside the model. No training needed for some of these.

### Experiment: Activation Steering (CAA)
- What to learn: linear representation hypothesis, how behaviors encode as directions, layer selection, steering vector quality
- Infrastructure: needs contrastive pair creation + TransformerLens/nnsight setup
- Key questions:
  - Which layers encode sycophancy most strongly?
  - How does steering magnitude affect behavior vs coherence tradeoff?
  - Is one direction enough or is sycophancy multi-dimensional?
  - How does this compare to training-based methods on behavioral eval?
- Status: [ ] NOT STARTED

### Experiment: Task Vector Negation
- What to learn: is sycophancy linearly separable in weight space? Weight arithmetic as alignment tool
- Infrastructure: TRIVIAL — just weight subtraction, both models exist
- Key questions:
  - Does W_base - alpha*(W_syc - W_base) actually reduce sycophancy?
  - What alpha values work? What breaks?
  - How does weight-space intervention compare to activation-space (CAA)?
  - Layer-aware variant (LATA): do certain layers matter more?
- Status: [ ] NOT STARTED

### Experiment: Pinpoint Tuning (SPT)
- What to learn: causal tracing, attention head roles, surgical intervention, which heads encode sycophancy
- Infrastructure: needs causal analysis pipeline + selective fine-tuning (code available from ICML 2024)
- Key questions:
  - How many heads are responsible for sycophancy in Qwen3-8B?
  - Which layers are they concentrated in?
  - Does fixing <5% of heads match or beat full DPO?
  - What happens to those heads after DPO/SimPO — do they change?
- Status: [ ] NOT STARTED

### Experiment: Linear Probing (THE BIG COMPARISON)
- What to learn: do alignment techniques change internal representations or just output behavior?
- Infrastructure: needs activation extraction + probe training across all models
- Key questions:
  - Train probe on sycophantic model activations to detect sycophancy
  - Apply probe to each recovered model — does probe accuracy drop?
  - If probe still detects sycophancy after DPO but not after pinpoint tuning, what does that mean?
  - Which layers show the most change across techniques?
  - Seen vs unseen questions: does the probe generalize?
- Status: [ ] NOT STARTED

---

## Bonus / Optional Experiments

### Gradient Ascent Unlearning
- What to learn: catastrophic forgetting, what "unlearning" means mechanistically
- Negate loss on sycophantic examples — does it unlearn sycophancy or everything?

### SAE-Based Steering
- What to learn: sparse autoencoders, interpretable features, feature-level intervention
- More precise than CAA but requires training an SAE first

### Full-Parameter DPO (no LoRA)
- What to learn: does modifying ALL 8B weights produce deeper sycophancy removal than LoRA?
- Same data, same hyperparams as LoRA DPO, but no adapter — train everything
- Needs FSDP or DeepSpeed ZeRO-2 to fit on 4x H100 (full optimizer states ~32GB)
- Key comparison: linear probe accuracy on LoRA DPO vs full-param DPO recovered models
- Higher risk of catastrophic forgetting — part of the learning

### DPO Variant Ablations
- Beta sweep (0.05, 0.1, 0.2, 0.3, 0.5) on DPO
- Learning rate sweep
- Data size ablation (25%, 50%, 75%, 100% of DPO pairs)

---

## Results Tracking

Each experiment gets:
1. Entry in `logs/experiment_log.md` (sequential numbering continues from 002)
2. Detailed write-up in `logs/NNN_experiment_name.md`
3. Metrics JSON in `results/eval/<run-name>/`
4. Learnings added to `logs/learnings.md`

## Reference

- Full technique survey: `.claude/research/alignment-techniques-survey.md`
- Training library details: `.claude/research/training-libraries-research.md`
- Eval system design: `.claude/research/eval-system-research.md`
