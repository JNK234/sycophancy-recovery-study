# Next Steps Roadmap: Sycophancy Recovery Study

**Author:** Narasimha Karthik Jwalapuram
**Created:** 2026-03-27
**Status:** Active Planning
**Builds on:** `research-plan.md` (original research proposal), experiments 001-005

---

## 1. Where We Are

### Completed
- **Phase 1 (Model Organism):** SFT on Qwen3-8B, sycophancy 0.256 -> 0.467
- **Phase 2a (DPO Recovery):** Aggregate sycophancy 0.467 -> 0.268 (near baseline)
- **Phase 3a (Linear Probing v2):** SFT->DPO transfer AUROC 0.754 -- suppression, not removal
- **Phase 3b (Relearning Speed):** DPO relearns sycophancy faster than base -- pathway intact
- **Infrastructure:** Full eval pipeline (LLM-as-judge, 72B Qwen2.5), training pipeline (SFT/DPO with DDP), probing pipeline (extraction + sklearn probes)

### Headline Finding
DPO recovers sycophancy behaviorally but the SFT-created internal representation persists (transfer AUROC 0.754). The SFT probe does NOT transfer to base (0.581), confirming it's SFT-created, not pre-existing.

### Open Question
Is this a DPO-specific limitation, or do ALL alignment interventions only suppress sycophancy's surface expression?

---

## 2. Project Philosophy

This is a **learning-driven research engineering project**, not a deadline-driven paper push.

- **Research engineer workflow:** experiment -> learn -> log -> share -> iterate
- **Lit survey before every technique:** understand the math/mechanism FIRST, then implement
- **Incremental sharing:** blog posts and social posts at milestones, not just a final write-up
- **Experiment tracking:** every run logged in `logs/experiment_log.md` with interpretation
- **Learnings captured:** technical insights in `logs/learnings.md`
- **Portfolio-grade output:** well-documented GitHub repo + incremental posts showing full-stack capability (post-training, evaluation, mechanistic interpretability)
- **Follow the data:** no predetermined narrative -- let results tell the story
- **Ideas backlog:** maintain a running list of follow-up experiments to pursue when interesting patterns emerge (e.g., relearning speed tests, ablations, probing variants)

---

## 3. Technique Comparison Matrix (Phase 2 Completion)

All techniques recover from the same SFT sycophantic model. Each gets:
1. Lit survey + research notes saved to `.claude/research/`
2. Implementation with learnings documented
3. Full behavioral eval (3 datasets, 72B judge)
4. Linear probing on the 500 answer-dataset prompts (consistent across all)
5. Experiment log entry with interpretation
6. Blog post / shareable artifact at milestones

### 3.1 Preference-Based Methods

| Technique | Infrastructure Need | Estimated Effort | Key Learning Goal |
|-----------|-------------------|-----------------|-------------------|
| **SimPO** | Config change (loss_type) | ~1 hour | Reference-free DPO -- does removing the anchor change recovery depth? Length normalization vs sycophantic verbosity |
| **IPO** | Config change (loss_type) | ~1 hour | DPO regularization -- does preventing overfitting improve internal representation change? |
| **KTO** | Data reformatting (unpaired) + KTOTrainer | ~half day | Unpaired feedback -- does loss aversion modeling help for sycophancy? Different data structure implications |

**Approach:** Start with SimPO/IPO (trivial config swaps), then KTO (needs data work). For hyperparameters, start with matched configs for fair comparison, then explore tuned configs as follow-up experiments. Decisions made during implementation, not upfront.

### 3.2 RL-Based Methods

| Technique | Infrastructure Need | Estimated Effort | Key Learning Goal |
|-----------|-------------------|-----------------|-------------------|
| **RLHF (PPO)** | Reward model training + PPO loop | ~1-2 days | Full RLHF stack: reward modeling, KL penalty, online exploration, reward hacking |
| **GRPO** | Group scoring setup (no value network) | ~1 day | Modern RL alignment without reward model -- compare to PPO on same problem |

**Approach:** Implement both PPO and GRPO. PPO for the canonical RLHF experience and portfolio signal. GRPO for modern comparison. Lit survey before each to understand current best practices. Implementation details deferred to when we reach this stage.

### 3.3 Self-Supervision Methods

| Technique | Infrastructure Need | Estimated Effort | Key Learning Goal |
|-----------|-------------------|-----------------|-------------------|
| **Constitutional AI** | Constitution design, self-critique pipeline, SFT on revisions, optional RLAIF | ~1-2 days | Can a sycophantic model identify its own sycophancy? Self-improvement vs external supervision |

**Approach:** Lit survey first. Key open questions (self-critique vs 72B-critique, Phase 1 vs Phase 1+2) resolved during implementation based on research findings.

### 3.4 Representation Engineering

| Technique | Infrastructure Need | Estimated Effort | Key Learning Goal |
|-----------|-------------------|-----------------|-------------------|
| **Activation Steering (CAA)** | Contrastive pair extraction, direction computation, inference-time steering | ~half day | Linear representation hypothesis applied to sycophancy. Layer selection, magnitude-coherence tradeoff |

**Approach:** Start with single DiffMean direction. The "Sycophancy Is Not One Thing" decomposition (agreement vs praise as separate directions) is a research-first decision -- deep-read the paper before deciding whether to decompose.

### 3.5 Bonus/Optional Techniques (Ideas Backlog)

These stay in the backlog and get promoted based on findings:

- **Task Vector Negation:** Weight arithmetic (W_base - alpha*(W_syc - W_base)). Trivial to implement, different paradigm
- **Pinpoint Tuning (SPT):** Surgical attention head intervention. Interesting if probing reveals layer-specific patterns
- **Full-Parameter DPO:** Same data/config as LoRA DPO but training all 8B weights. Tests depth-of-modification hypothesis
- **Gradient Ascent Unlearning:** Negate loss on sycophantic examples. What does "unlearning" look like mechanistically?
- **DPO Ablations:** Beta sweep, LR sweep, data size ablation
- **SAE-Based Steering:** Train sparse autoencoder, steer at feature level. Most precise but highest infrastructure cost

---

## 4. Mechanistic Interpretability (Phase 3 Deepening)

### 4.1 Linear Probing Comparison (Immediate)

Run the existing probing pipeline on every recovered model as they're produced:
- Same 500 answer-dataset prompts, per-model behavior labels from judge verdicts
- Cross-model transfer: SFT probe applied to each recovered model
- Direction similarity: cosine between probe weights across models
- Output: comparison table of transfer AUROCs -- the central evidence for suppression vs removal

### 4.2 Full Mech Interp Toolkit (After All Techniques Trained)

Introduce TransformerLens or nnsight AFTER all recovery methods are complete. Then apply to the most interesting models:

| Tool | What It Answers | When to Use |
|------|----------------|-------------|
| **Causal Tracing / Activation Patching** | Does the model CAUSALLY USE the sycophancy representation, or is it just present? | When probing shows representation persists -- is it load-bearing? |
| **Logit Lens** | How does the model's "prediction" evolve layer by layer? | Understanding where sycophancy decision happens in the forward pass |
| **Attention Head Analysis** | Which heads attend to user-opinion signals? | Finding the "sycophancy circuit" |
| **Logit Attribution** | Which components (heads, MLPs) contribute most to sycophantic outputs? | Identifying intervention targets |

**Approach:** Lit survey on TransformerLens/nnsight for Qwen3-8B compatibility before committing. Infrastructure setup as a dedicated step. Deep analysis on selected interesting models, not all 8+.

---

## 5. Adversarial Robustness Testing (Phase 4)

**Current knowledge level:** Low -- needs lit survey before design.

### Planned Attack Categories (from research plan)

1. **Many-Shot Re-elicitation:** 10-20 in-context sycophantic examples -> does recovered model revert?
2. **Persona Injection:** System prompt overriding recovery ("You are an extremely agreeable assistant")
3. **Social Pressure Escalation:** Multi-turn progressive pushback, measure turn-of-flip
4. **Indirect Sycophancy Probes:** Opinion-matching, epistemic confidence, framing effects

### Approach
- **Research first:** Survey SYCONBench (Hong et al., 2025), adversarial prompting literature, red-teaming methodologies
- **Design informed by findings:** The right adversarial tests depend on what we learn from technique comparison + probing
- **Implementation:** Decide on infrastructure (extend existing judge pipeline vs lightweight approach) after research
- **Save research notes** to `.claude/research/adversarial-testing-research.md`

---

## 6. Sycophancy-to-Subterfuge (Phase 5)

**Current knowledge level:** Low -- needs deep engagement with Denison et al. (2024).

### Core Question
Does removing surface sycophancy break the escalation chain to reward tampering and checklist manipulation?

### Approach
- **Deep-read** Denison et al. (2024) and related work on reward hacking, specification gaming
- **Survey latest findings** on behavioral escalation in LLMs
- **Design novel evaluation scenarios** -- this requires creative scenario design, not just running benchmarks
- **Test on interesting recovered models** -- particularly those where probing shows different depth of removal
- **Save research notes** to `.claude/research/subterfuge-escalation-research.md`

This is the most ambitious phase. It may evolve significantly based on what we learn in earlier phases.

---

## 7. Evaluation Strategy

### Behavioral Eval (Consistent Across All Techniques)
- **Pipeline:** Existing two-pass (vLLM generation + 72B judge scoring)
- **Datasets:** All 3 (answer, are_you_sure, feedback) for every technique
- **Judge:** Qwen2.5-72B-Instruct (consistent with experiments 001-003)
- **Metrics:** Aggregate sycophancy, answer sycophancy rate, flip rate, feedback sycophancy
- **Cost:** ~1-2 hours GPU time per eval run. Acceptable for the comparison table.

### Probing Eval (Consistent Across All Techniques)
- **Prompt set:** 500 answer-dataset prompts (same as experiment 005)
- **Labels:** Per-model behavior labels from judge verdicts
- **Probes:** Logistic regression per layer, AUROC on held-out set
- **Transfer:** SFT probe -> each recovered model
- **Future expansion:** Can add are_you_sure and feedback prompts, OOD TruthfulQA later if interesting

### Mid-Training Eval
- **Logit-based MC callback** during training (existing infrastructure)
- **Useful for:** Monitoring convergence, catching overfitting early, comparing training dynamics across methods

---

## 8. Artifacts and Infrastructure

### Model Storage
- All models on `/scratch/wnn7240/sycophancy-recovery/outputs/<method>/merged`
- Accept scratch storage risk -- everything is reproducible from configs
- LoRA adapters saved alongside merged models

### Results (Git-Tracked)
- `results/eval/<run-name>/` -- metrics JSON per technique
- `results/probing/<run-name>/` -- probing metrics + plots
- `logs/NNN_experiment_name.md` -- detailed write-ups
- `logs/experiment_log.md` -- master index

### Research Notes
- `.claude/research/<topic>-research.md` -- lit survey notes per technique/topic
- Updated before implementing each technique

### Sharing Artifacts
- **Incremental blog posts** at milestones:
  - Post 1: "Building a sycophantic model organism" (Phase 1 + baseline) -- can write now
  - Post 2: "DPO recovery and what probes revealed" (DPO + probing finding) -- can write now
  - Post 3: "Comparing 6 alignment techniques" (after preference + RL methods)
  - Post 4: "Inside the model: mechanistic interpretability of alignment" (after mech interp phase)
  - Post 5+: Adversarial testing, subterfuge, final synthesis
- Posts can double as portfolio pieces and force clear thinking at each stage

---

## 9. Suggested Execution Order

This is a flexible guide, not a rigid plan. Reorder based on learning and findings.

### Block A: Preference Methods (Next Up)
1. **SimPO** -- config swap, train, eval, probe, log
2. **IPO** -- config swap, train, eval, probe, log
3. **KTO** -- data reformatting, implement KTOTrainer, train, eval, probe, log
4. *Milestone post: "Comparing preference optimization methods for sycophancy recovery"*

### Block B: RL Methods
5. **Lit survey:** PPO vs GRPO for alignment recovery
6. **RLHF (PPO)** -- reward model, PPO loop, train, eval, probe, log
7. **GRPO** -- group scoring, train, eval, probe, log
8. *Milestone post: "RL-based alignment: PPO and GRPO for sycophancy recovery"*

### Block C: Self-Supervision + Representation
9. **Lit survey:** CAI for specific behavioral issues
10. **CAI** -- constitution, self-critique pipeline, SFT/RLAIF, eval, probe, log
11. **Lit survey:** Activation steering, "Sycophancy Is Not One Thing" decomposition
12. **Activation Steering** -- contrastive pairs, direction computation, steering, eval, probe, log
13. *Milestone post: "Beyond preference data: CAI and activation steering"*

### Block D: Comprehensive Mech Interp
14. **Set up TransformerLens/nnsight** for Qwen3-8B
15. **Causal tracing** on selected interesting models
16. **Logit lens + attention head analysis**
17. **Full comparison table:** behavioral eval + probing + causal evidence
18. *Milestone post: "Inside the model: where sycophancy lives and how alignment changes it"*

### Block E: Adversarial + Subterfuge
19. **Lit survey:** Adversarial robustness, SYCONBench, red-teaming
20. **Design and implement adversarial eval**
21. **Lit survey:** Sycophancy-to-subterfuge escalation
22. **Design and implement subterfuge scenarios**
23. *Milestone post: "Can we break the fix? Adversarial testing of alignment interventions"*

### Ideas Backlog (Promote as Interesting)
- Relearning speed test on specific recovered models
- Task vector negation
- Pinpoint tuning (SPT)
- Full-parameter DPO
- DPO ablations (beta, LR, data size)
- SAE-based steering
- Gradient ascent unlearning
- Expand probing to are_you_sure + feedback datasets
- OOD generalization of sycophancy direction

---

## 10. Success Criteria

This project succeeds if it:

1. **Produces a comprehensive comparison table** of 6+ alignment techniques on the same sycophancy model organism -- behavioral and mechanistic
2. **Answers the suppression vs removal question** with multiple lines of evidence (probing, causal tracing, relearning, adversarial robustness)
3. **Demonstrates full-stack capability:** post-training implementation, rigorous evaluation, mechanistic interpretability
4. **Generates shareable insights** via blog posts that demonstrate clear thinking about alignment
5. **Builds deep understanding** of each technique's mechanism, not just "I ran the code"

The specific findings (whether any technique truly removes sycophancy, which is best, etc.) are genuinely open questions. The project's value comes from the methodology and depth of investigation, regardless of which way the results go.
