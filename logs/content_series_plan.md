# Content Series Plan: Sycophancy Recovery Research Blog

**Author:** Narasimha Karthik Jwalapuram
**Created:** 2026-03-28
**Format:** Serialized deep-dive research blog, one post per technique/phase
**Style:** Anthropic-style research blogs — first principles, shows the work, honest interpretation
**Cadence:** ~1 post per week
**Platform:** LinkedIn + X (long-form article or thread)

## Series Arc

Each post teaches the reader a technique AND advances the overarching question: **Does alignment genuinely remove sycophancy, or just suppress its surface expression?**

## Post Schedule

### Post 1: "The Setup + DPO" (Week 1)
- The research question and motivation
- Building the model organism (SFT on Qwen3-8B, 3,200 pairs, 4 pressure tactics)
- The evaluation pipeline (3 dimensions, LLM-as-judge, 20K+ samples)
- The microscope: linear probing (what it is, how it works, why prompt-only)
- Baseline results (0.256, subjectivity gradient, flip rate)
- DPO: what it is, how it works, the loss equation
- DPO results: behavioral recovery (0.467→0.268) but probing reveals suppression (0.677 transfer)
- Relearning speed corroboration
- Cliffhanger: the reference model as ceiling

### Post 2: "SimPO — Removing the Anchor" (Week 2)
- Quick recap + link to Post 1
- SimPO mechanism: reference-free, length-normalized, no KL constraint
- How it differs from DPO (loss equation comparison)
- Hyperparameter journey (3 runs, paper defaults don't transfer)
- Behavioral results: 0.176 (below baseline), flip rate 10%, poems 0.7%
- Probing results: 0.503 (chance), cosine 0.082 (orthogonal)
- Qualitative side-by-sides (Curling, Gaborone, Burton)
- The reference model hypothesis
- Cliffhanger: does every reference-constrained method hit the same wall?

### Post 3: "IPO — Testing the Hypothesis" (Week 3)
- Recap the hypothesis
- IPO mechanism: reference-constrained, squared hinge loss
- Results + probing
- Hypothesis confirmation or revision
- Cliffhanger for KTO

### Post 4: "KTO — Unpaired Feedback" (Week 4)
- KTO mechanism: loss aversion, unpaired data
- Data reformatting from DPO pairs
- Results + probing
- What unpaired vs paired tells us

### Post 5: "PPO/GRPO — Full RL" (Week 5-6)
- Reward model training
- PPO loop: KL penalty, online exploration, reward hacking
- GRPO: group scoring, no value network
- Results + probing
- RL vs preference optimization comparison

### Post 6: "CAI — Can the Model Fix Itself?" (Week 7)
- Constitutional AI mechanism: self-critique, revision, SFT on revisions
- Can a sycophantic model identify its own sycophancy?
- Self-supervision vs external supervision
- Results + probing

### Post 7: "Activation Steering — No Training Required" (Week 8)
- Contrastive activation addition (CAA)
- Computing the sycophancy direction
- Inference-time steering without any weight changes
- Results + probing
- Representation engineering paradigm

### Post 8: "Inside the Model — Mechanistic Interpretability" (Week 9-10)
- TransformerLens / nnsight setup for Qwen3-8B
- Causal tracing: is the representation load-bearing?
- Logit lens: where does the sycophancy decision happen?
- Attention head analysis: the sycophancy circuit
- Deep-dive on the most interesting models

### Post 9: "Breaking the Fix — Adversarial Testing" (Week 11)
- Many-shot re-elicitation
- Persona injection
- Social pressure escalation
- Which recovery methods are robust vs fragile?

### Post 10: "The Full Picture — Capstone" (Week 12+)
- Complete comparison table (behavioral + probing)
- The reference model hypothesis with full evidence
- Suppression vs removal spectrum
- What this means for AI safety
- Open questions and future directions

## Post Template (Anthropic-style)

Each post follows this structure:
1. **Hook** — One compelling question or finding
2. **Context** — What is this technique? (first principles, math, intuition)
3. **How I Applied It** — Implementation, gotchas, engineering decisions
4. **Results** — Behavioral + probing, honest about limitations
5. **Interpretation** — What this means for the hypothesis
6. **What's Next** — Cliffhanger for the following post

## Visual Assets Per Post
- Growing comparison table (updated each post)
- Technique-specific figures (loss curves, probe transfer bars)
- Qualitative examples where relevant (side-by-side outputs)
- Layer-by-layer AUROC curves from probing

## Cross-Posting Strategy
- **LinkedIn:** Full article format
- **X:** Thread format (key points) linking to full article
- **GitHub:** Code + results always linked
- Each post references prior posts for serial readers
