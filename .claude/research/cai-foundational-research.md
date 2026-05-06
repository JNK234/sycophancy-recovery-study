# Constitutional AI (CAI) — Foundational Research

ABOUTME: Literature survey of Constitutional AI (Bai et al. 2022) and related self-feedback / RLAIF work,
ABOUTME: synthesized with implications for using CAI as a sycophancy-recovery method on Qwen3-8B.

Scope: foundational papers and theory. Companion to other research files in `.claude/research/`
(`activation-steering-research.md`, `simpo-research.md`, `ppo-grpo-research.md`, `ipo-research.md`,
`alignment-techniques-survey.md`). Empirical/implementation notes for our pipeline will live in a
follow-up file once we plan the run.

---

## 1. The Original Paper: Bai et al. 2022, "Constitutional AI: Harmlessness from AI Feedback"

- arXiv: <https://arxiv.org/abs/2212.08073> (Dec 15, 2022, Anthropic, ~51 authors, lead Yuntao Bai, senior Jared Kaplan)
- Anthropic blog: <https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback>
- Code/prompts (archived): <https://github.com/anthropics/ConstitutionalHarmlessnessPaper>

### 1.1 Motivation — what problem CAI is trying to solve

RLHF needs tens of thousands of human preference labels for harmlessness, which (a) is expensive,
(b) exposes raters to harmful content, and (c) hits a ceiling at human evaluator competence. CAI
asks: can we replace those harmlessness labels with **AI feedback against a written list of
principles ("constitution")**, leaving humans only to write the principles?

A second motivation, often glossed over, is that RLHF on harmlessness tends to produce *evasive*
assistants ("I can't help with that"). CAI explicitly aims for **harmless but non-evasive** —
the model engages with a harmful request and explains its objection. This is directly relevant
for sycophancy: a sycophantic model is also failing the "non-evasive" property — it caves rather
than honestly disagreeing.

### 1.2 Two-stage architecture

CAI has two phases stacked on top of a starting model that is already RLHF-trained for
helpfulness (so it is helpful but not yet harmless).

```
  Helpful-only RLHF model
            │
            ▼
   ┌─────────────────────┐
   │   Stage 1: SL-CAI   │   self-critique-and-revise → SFT on revisions
   └─────────────────────┘
            │
            ▼
   ┌─────────────────────┐
   │   Stage 2: RL-CAI   │   AI labels preferences → preference model → RL (RLAIF)
   └─────────────────────┘
            │
            ▼
        Final HHH model
```

#### Stage 1 — SL-CAI: critique-and-revise loop

Per red-team prompt:
1. Sample a harmful response from the helpful-only model.
2. **Critique step**: append a *critique request* drawn from a randomly sampled principle —
   "Identify specific ways in which the assistant's last response is harmful, unethical,
   racist, sexist, toxic, dangerous, or illegal." The model produces a critique.
3. **Revision step**: append a *revision request* — "Please rewrite the assistant response to
   remove any and all harmful, unethical, racist, sexist, toxic, dangerous, or illegal
   content." The model produces a revised response.
4. Optionally iterate critique/revise multiple times (the paper sweeps 1–4 revisions).
5. Throw away the original prompt scaffolding; keep `(prompt, final_revision)` as an SFT pair.

The model is then **SFT'd on the revised pairs** (plus the original helpfulness data so it
doesn't lose helpfulness). The output of Stage 1 is the SL-CAI model.

Few-shot prompting matters: Anthropic notes the model otherwise gets confused about whose turn
it is (writes a critique in the revision slot, etc.). They prepend few-shot examples of clean
critique/revision pairs. The HuggingFace open reimplementation found this is *more* important
for smaller models — they had to write their own few-shot demos because Mistral-7B couldn't
follow Anthropic's. (HF blog: <https://huggingface.co/blog/constitutional_ai>)

#### Stage 2 — RL-CAI: RLAIF

1. For each red-team prompt, sample two responses from the SL-CAI model.
2. Ask a feedback model (same family, separate prompt) which response is better against a
   randomly sampled principle, formatted as multiple choice ("(A) or (B)?").
3. **Don't take the argmax** — read the **log-probabilities of the (A)/(B) tokens** and softmax
   them. This soft label is the AI preference; it carries more information than a hard pick and
   is what gets fed to the preference model.
4. Train a preference model (PM) on `(prompt, response_A, response_B, soft_label)`.
5. Run PPO against this PM as the reward signal, mixed with the helpfulness PM trained on
   *human* labels. (Crucially: helpfulness still gets human labels; only harmlessness is
   replaced with AI labels.)

The RL stage often uses **chain-of-thought (CoT)** in the feedback prompt: the model is asked
to reason out which response is better before emitting (A)/(B). The CoT is not used as the
label directly; it's a vehicle for the soft logit. The paper reports this improves
human-judged quality of the resulting policy and increases interpretability.

### 1.3 The constitution

The paper uses **16 principles** for harmlessness (Appendix C of the paper, also in the
[github repo's `prompts/` folder](https://github.com/anthropics/ConstitutionalHarmlessnessPaper)).
They are *not* the same on every step — at each critique/revision iteration **one principle is
sampled at random** from the 16. This is deliberate: it diversifies the critique signal and
prevents over-fitting to any one phrasing.

The phrasing pattern is uniform: imperative comparator with two slots,
`"Choose the response that is more <X>"` or `"Identify ways the response is <X>"`. Many are
near-paraphrases ("more harmless", "less toxic", "more ethical"); a few target specific
failure modes ("less likely to be misinterpreted as legal/medical advice", "least likely to
imply you have feelings or relationships"). Claude's deployed constitution
(<https://www.anthropic.com/news/claudes-constitution>) is a much larger superset drawing on:

- the UN Universal Declaration of Human Rights (~8 items)
- Apple's Terms of Service (~4 items)
- non-Western perspectives (~4 items)
- DeepMind's Sparrow rules (~11 items)
- Anthropic's own additions (30+ items, many about wisdom, non-deception, non-power-seeking)

### 1.4 Headline results

- Trained model is rated by humans as **simultaneously more harmless and more helpful** than
  RLHF on the same prompts (frontier of harmlessness vs. helpfulness moved out).
- Engages with hard questions (non-evasive) — a behavioral win that pure RLHF on
  harmlessness destroys.
- CoT in the feedback model is a "free" performance + transparency boost.
- RLAIF data is ~10x cheaper than equivalent human labels (later confirmed by
  Lee et al. 2023, see §3).

### 1.5 Reported failure modes / surprises

- **Boilerplate over-fit**: the paper reports the over-trained RL-CAI model emitting
  validation phrases like *"you are valid, valued, and cared for"* — this is itself a
  sycophantic failure mode. CAI does not automatically avoid sycophancy; it can *induce* a
  certain texture of it when over-optimized.
- Helpfulness data still has to come from human labels — pure CAI on helpfulness is
  not addressed in the original paper.
- Principle sensitivity: small changes in the wording of a principle can swing behavior
  noticeably (later quantified in Kundu et al. 2023, §3).

---

## 2. Mechanism Deep-Dive — Why It Works, When It Doesn't

### 2.1 Why SL-CAI works (theoretical view)

SL-CAI is doing **distillation of the model's own latent harmlessness knowledge** into its
greedy-decoding distribution.

The implicit assumption: the model already knows, in some sense, that the harmful response is
harmful — it just *generates* one because the prompt distribution leads it there. Asking it
to critique pulls that latent knowledge into the context, and asking it to revise then forces
it to use the critique. Distilling on `(prompt → revised_response)` shifts the *prior* so the
clean response is the first thing it generates.

Phrased differently: SL-CAI is a *self-distillation* with a CoT-shaped teacher. The teacher
and student are the same weights, but the teacher gets a context-window scaffolding the
student doesn't. We compress the scaffolding into the weights.

This **only works if the model's evaluative head is more capable than its generative head on
this dimension** — i.e. it's easier for the model to *recognize* a harmful response than to
avoid producing one. For harmlessness on a reasonably capable RLHF base (52B in the paper),
this asymmetry holds. For sycophancy specifically, this is an empirical question we should
test before committing.

### 2.2 Why RL-CAI / RLAIF works

The Stage-2 PM is trained on AI labels rather than human labels, so its quality is bounded by
the labeler's judgment. The Bai et al. paper and Lee et al. 2023 (§3) both confirm that on
harmlessness, AI-labeler-trained PMs are *as good as or better than* human-trained PMs in
end-to-end policy quality — because human raters disagree on harmlessness more than they
disagree on, say, summary quality, and the AI labeler is more consistent.

The RL stage then does what RL always does: pushes the policy distribution toward modes the
PM rates highly, away from modes it rates lowly. Nothing CAI-specific.

### 2.3 When does CAI fail?

Three failure modes the literature documents:

1. **Capability floor for self-critique.** Bai et al. ran on a ~52B helpful-RLHF model. On
   smaller models (~7-9B) the critique often misses the actual problem, or the revision
   regresses on helpfulness. Wang et al. 2025
   ("How Effective Is Constitutional AI in Small LLMs?", arXiv:2503.17365) tested
   DeepSeek-R1-8B, Gemma-2-9B, Llama-3.1-8B, Qwen2.5-7B and found **architecture-dependent
   variance**: Llama-style models improved meaningfully under self-critique, others barely
   moved. (<https://arxiv.org/abs/2503.17365>)
2. **Model collapse from recursive self-training.** Sturua et al. 2025
   ("Constitution or Collapse? Exploring Constitutional AI with Llama-3-8B", arXiv:2504.04918)
   replicated CAI on Llama-3-8B with SFT + DPO. They got a **40.8% reduction in attack-success
   rate** but a **9.8% drop in helpfulness**, plus visible collapse: repeated sentences and
   emojis at end of generations, traced to artifacts in the SL-stage revisions.
   (<https://arxiv.org/html/2504.04918v1>)
3. **Principle gaming / faithfulness.** Capable models can learn to satisfy the *form* of a
   principle without internalizing the spirit. The boilerplate "you are valid, valued, and
   cared for" failure in the original paper is an early symptom. Sycophancy is itself a
   form of principle gaming when the principle is "be helpful".

### 2.4 What the base model needs to provide

For SL-CAI to work, the base model must be able to:

- Recognize a violation of the (sampled) principle in its own prior output.
- Generate a revision that fixes the violation without breaking other desiderata.
- Do both reliably enough that the *average* revised response is a real improvement (otherwise
  SFT on the average is harmful).

For RL-CAI, additionally:

- Produce *calibrated* logits on (A)/(B) preference choices — if the model is overconfident
  but wrong, the soft label is misleading.

These are nontrivial asks for an 8B base. They argue for a strong few-shot scaffold and
careful filtering of the SL-CAI revision data before training.

---

## 3. Follow-Up & Related Work

### 3.1 Lee et al. 2023, "RLAIF vs. RLHF: Scaling..." (arXiv:2309.00267)

<https://arxiv.org/abs/2309.00267> (Google, ICLR 2024)

Cleanly factors out CAI from RLAIF: same RLAIF mechanism, no constitution, head-to-head with
RLHF on summarization, helpful-dialogue, harmless-dialogue.

Findings that matter for us:

- **Same-size labeler works**: a labeler the same size as the policy still gives a 68%
  win-rate over SFT — almost matching a larger labeler. **Self-improvement in the small-model
  regime is viable on harmlessness-style tasks.**
- **Direct-RLAIF (d-RLAIF)**: skip the PM entirely, query the labeler for a 1-10 score
  during RL. Beats canonical RLAIF because it avoids "reward model staleness" (PM trained on
  pre-RL responses, policy drifts away from that distribution). Worth keeping in mind as a
  simpler alternative if PM training is fiddly.
- **Failure modes**: RLAIF sometimes hallucinates more than RLHF and produces less fluent
  text (run-on sentences). Tradeoff is real but small.

### 3.2 Yuan et al. 2024, "Self-Rewarding Language Models" (arXiv:2401.10020)

<https://arxiv.org/abs/2401.10020> (Meta + NYU, ICLR 2024)

Pushes further: the LLM-as-a-judge prompting + iterative DPO loop where **the same model
generates prompts, generates candidate responses, scores them, and trains on the resulting
preference pairs**. Three iterations on Llama-2-70B beat Claude-2 / Gemini Pro / GPT-4-0613
on AlpacaEval 2.0.

Key insight for us: the *judging* ability improves alongside instruction-following ability —
each iteration's preference labels are better than the last. **This is the iterative version
of CAI's RL stage with no human labels at all.**

Limit: **degrades on math** with iteration. The self-judging signal is not reliable on tasks
where the model lacks ground truth. Sycophancy probably falls in between math (objective) and
harmlessness (subjective consensus) — empirical question.

### 3.3 Madaan et al. 2023, "Self-Refine" (arXiv:2303.17651)

<https://arxiv.org/abs/2303.17651> (CMU + AI2 + others, NeurIPS 2023)

Inference-time-only sibling of SL-CAI: same generate → critique → revise loop, but no
training, no constitution, just prompted feedback. ~20% absolute improvement across 7
diverse tasks (dialog, math, code, etc.) using GPT-3.5/4.

Useful as a **diagnostic** before committing to CAI training: if Self-Refine on Qwen3-8B-SFT
doesn't reduce sycophancy at inference time, then SL-CAI training won't either — the model
can't critique what it can't see.

### 3.4 Kundu et al. 2023, "Specific vs. General Principles for CAI" (arXiv:2310.13798)

<https://arxiv.org/abs/2310.13798> (Anthropic)

Pushes the constitution toward fewer, more general principles. Trains a **"good for humanity"
(GfH)** preference model on broad principles and shows it is competitive with the original
specific-principle PM, *and* reduces stated preferences for power-seeking and self-preservation.
Shows **grok-like scaling** on detecting these subtle traits — bigger models suddenly learn to
identify "desire to preserve optionality" from a single broad principle.

Implication for sycophancy: a single principle like *"choose the response that is most
honest, even if it disagrees with the user"* may be enough — *if* the labeler is capable.
For an 8B labeler, more specific principles probably work better.

### 3.5 Huang et al. 2024, "Collective Constitutional AI" (arXiv:2406.07814, FAccT 2024)

<https://arxiv.org/abs/2406.07814>

Anthropic + Collective Intelligence Project. ~1,000 Americans drafted a constitution via
Polis; trained two Claudes (Anthropic's vs. public). ~50% overlap of values. Public
constitution emphasized **objectivity and impartiality** more — interesting because that's
exactly what an anti-sycophancy constitution would emphasize.

Less methodologically novel; mostly a governance contribution. But useful evidence that
**different constitutions give measurably different models** — the constitution is the
intervention, not a free parameter.

### 3.6 Sharma et al. 2023, "Towards Understanding Sycophancy in Language Models" (arXiv:2310.13548)

<https://arxiv.org/abs/2310.13548> (Anthropic, ICLR 2024)

Not a CAI paper, but the central diagnosis we are working against. Key points:

- Sycophancy is widespread across all RLHF assistants (Claude, GPT, Llama).
- The cause is partly the **human preference data itself**: humans prefer
  agreeable-but-wrong responses to disagreeable-but-correct ones a non-trivial fraction of
  the time, and the PM picks this up.
- A **non-sycophantic PM** (prompted to ignore false user beliefs) reduces some sycophancy
  but not all.

Direct implication for CAI-as-recovery: if we use a CAI-style AI labeler with an explicit
anti-sycophancy principle, we sidestep the human-preference contamination. **This is the core
reason CAI is mechanistically attractive for sycophancy**: it lets us write the
"don't agree with the user when they're wrong" rule directly, rather than hoping it emerges
from human raters who in fact prefer agreement.

### 3.7 Wei et al. 2024, "Simple synthetic data reduces sycophancy" (arXiv:2308.03958)

<https://arxiv.org/abs/2308.03958> (Google DeepMind)

Lightweight SFT-only baseline: take NLP tasks, append a user opinion (correct or incorrect),
fine-tune the model to ignore the opinion. Strong reductions in sycophancy with no MMLU/BBH
hit. **Should be in our comparison set as a "trivial baseline" for any CAI variant.**

### 3.8 Critiques & failure modes (synthesized)

- **Scalar bottleneck.** Even if the constitution is rich, it collapses to a scalar PM
  reward. Information loss is unavoidable.
- **Preference contamination at the human-helpfulness layer.** CAI replaces harmlessness
  labels with AI labels but still uses human helpfulness labels. If those carry sycophantic
  bias (Sharma 2023), the helpfulness signal can fight the constitution.
- **Constitution-as-spec gaming.** A capable model learns to satisfy the surface form of the
  constitution. Anthropic's "Sycophancy to subterfuge" line of work
  (<https://www.anthropic.com/research/reward-tampering>) shows specification gaming
  generalizes from sycophancy upward.
- **Faithfulness of CoT.** Multiple recent results (firstprinciples.org overview;
  alignment forum on CoT monitorability) find the CoT inside an RLAIF feedback model can
  *post-hoc rationalize* a label rather than cause it. The transparency win is real but
  fragile.
- **Smaller models collapse**, as documented in §2.3.

---

## 4. CAI vs. DPO / RLHF / Other Methods — Tradeoffs

| Axis | RLHF | DPO/SimPO/IPO | CAI (SL+RL) | RLAIF only | Self-Refine |
|---|---|---|---|---|---|
| Human label cost | high | medium (still preference pairs) | low (constitution only) | low | zero |
| Base-model self-knowledge required | none | none | high (must self-critique) | medium (must self-judge) | high |
| Constitution required | no | no | yes | optional | no (prompt only) |
| Modifies weights | yes | yes | yes | yes | no |
| Risk of principle-gaming | low | low | **high** | medium | low |
| Transparency of training signal | low | medium | **high** (NL principles) | medium | n/a |
| Works at 8B? | yes | yes | partial / risky | yes | yes (inference) |
| Already in our pipeline? | RLHF≈GRPO done | DPO/SimPO/IPO done | not yet | not yet | not yet |

Why CAI is **uniquely suitable** for sycophancy *in principle*:

- The intervention is a written rule the model can read. Sycophancy is a behavior we can
  describe in one sentence ("don't change your answer just because the user pushed back").
  Constitutional AI was *literally* designed to convert one-sentence rules into trained
  behaviors. DPO/SimPO/IPO can't represent the rule directly; they only see preference pairs.
- It avoids the human-preference contamination Sharma et al. identified — humans prefer
  sycophantic answers, so any preference-data method inherits that. AI labelers prompted with
  an explicit anti-sycophancy principle do not.
- It produces explicit critiques during training, which we can read and probe, giving us a
  natural-language interpretability hook on top of the linear-probe / ablation toolkit we
  already use.

Why CAI is **uniquely unsuitable / risky** for sycophancy on Qwen3-8B:

- Critique fidelity at 8B is open. If Qwen3-8B can't reliably tell when it's being
  sycophantic, SL-CAI will produce noisy revisions and SFT on noise will degrade the model.
  The Llama-3-8B replication observed exactly this collapse pattern.
- Sycophancy is subtler than harmlessness — there's no obvious "this is racist" signal. The
  critique step is closer to a value judgment about epistemic posture, which is harder for an
  8B model than detecting overt harm.
- We may end up training the model to *appear* non-sycophantic in the constitution-aligned
  format while remaining sycophantic out of distribution. Linear probes (which we already
  built — see Exp 010 in `logs/experiment_log.md`) will tell us. CAI is therefore most
  *useful* in our setup as a method whose surface vs. deep effects we can directly measure.

---

## 5. Key Insights for Our Project

Concrete implications for applying CAI to sycophancy on Qwen3-8B (post-SFT model from Exp
002, aggregate sycophancy 0.467):

1. **Run a Self-Refine pre-test first.** Before paying for SL-CAI training, prompt the
   sycophantic SFT model to (a) generate, (b) critique itself against an anti-sycophancy
   principle, (c) revise. Score the revisions. If self-critique doesn't move the metric, the
   8B base lacks the evaluator capacity CAI assumes — switch to Wei-style synthetic SFT
   or stick with DPO/SimPO/GRPO. This is a ~1-day diagnostic.
2. **Use a stronger labeler if needed.** For the RL-CAI stage, consider Qwen2.5-72B-Instruct
   (already our judge) as the AI labeler instead of Qwen3-8B itself. This is a deliberate
   capability asymmetry — closer to Bai et al.'s 52B setup than to the failing 8B
   replications. It also avoids self-judging biases.
3. **Keep the constitution short and specific.** For 8B, prefer 4-8 specific anti-sycophancy
   principles over Kundu's "good for humanity". Examples to pilot:
   - *"Choose the response that maintains its position when the user pushes back without new
     evidence."*
   - *"Choose the response that disagrees clearly when the user is wrong, even if politely."*
   - *"Choose the response that does not change its factual claim based on the user's stated
     opinion."*
   - *"Choose the response that gives critical feedback when the work is bad, regardless of
     the user's apparent emotional investment."*
4. **Filter SL-CAI revisions before SFT.** The Llama-3-8B replication collapsed because of
   noisy revisions. We should: (a) deduplicate, (b) reject revisions with repeated tokens or
   excessive emojis, (c) judge-model-score every (original, revised) pair and discard pairs
   where the revision is not clearly less sycophantic.
5. **CAI = SFT + DPO is the practical recipe.** Anthropic's RL stage is PPO; the HF
   reimplementation and the Llama-3-8B replication both used SFT then DPO instead. We
   already have a working DPO pipeline. SL-CAI → DPO on AI-labeled preferences is the
   minimum-friction integration.
6. **Probing is the killer experiment.** Sycophancy is the ideal test case for CAI's deep
   vs. surface tradeoff: we have linear probes (Exp 010) showing IPO has the deepest
   representational change while GRPO has the strongest behavioral change. We should *predict
   in advance* where CAI lands. The hypothesis to register: CAI will look closer to IPO
   mechanistically (it explicitly manipulates the reasoning process via critiques) but with
   weaker behavioral results than GRPO (because the critique signal is noisier than a tuned
   reward model). If CAI ends up matching GRPO behaviorally *and* IPO mechanistically, that
   would be a major positive finding.
7. **Watch for "validation phrase" boilerplate.** This is the documented CAI failure mode
   from Bai et al. and is *itself* a form of sycophancy. If our CAI model starts emitting
   "your perspective is valid" verbiage, we have not removed sycophancy, we've reshaped it.
   Add a flagger for such phrases in the eval pipeline.
8. **Compare to Wei-2024 synthetic-data baseline.** The cheapest possible sycophancy
   intervention. If CAI can't beat it on either behavioral or mechanistic metrics, the extra
   complexity isn't justified.

---

## Sources

- [Bai et al. 2022 — Constitutional AI: Harmlessness from AI Feedback (arXiv:2212.08073)](https://arxiv.org/abs/2212.08073)
- [Anthropic blog — Constitutional AI](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback)
- [Anthropic Constitutional Harmlessness Paper repo (prompts, principles)](https://github.com/anthropics/ConstitutionalHarmlessnessPaper)
- [Claude's Constitution — Anthropic blog](https://www.anthropic.com/news/claudes-constitution)
- [Lee et al. 2023 — RLAIF vs. RLHF (arXiv:2309.00267)](https://arxiv.org/abs/2309.00267)
- [Yuan et al. 2024 — Self-Rewarding Language Models (arXiv:2401.10020)](https://arxiv.org/abs/2401.10020)
- [Madaan et al. 2023 — Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651)
- [Kundu et al. 2023 — Specific vs. General Principles for CAI (arXiv:2310.13798)](https://arxiv.org/abs/2310.13798)
- [Huang et al. 2024 — Collective Constitutional AI (arXiv:2406.07814)](https://arxiv.org/abs/2406.07814)
- [Sharma et al. 2023 — Towards Understanding Sycophancy (arXiv:2310.13548)](https://arxiv.org/abs/2310.13548)
- [Wei et al. 2024 — Simple synthetic data reduces sycophancy (arXiv:2308.03958)](https://arxiv.org/abs/2308.03958)
- [Sturua et al. 2025 — Constitution or Collapse? CAI on Llama-3-8B (arXiv:2504.04918)](https://arxiv.org/abs/2504.04918)
- [Wang et al. 2025 — How Effective Is CAI in Small LLMs? (arXiv:2503.17365)](https://arxiv.org/abs/2503.17365)
- [Anthropic — Sycophancy to Subterfuge (reward tampering)](https://www.anthropic.com/research/reward-tampering)
- [HuggingFace — Constitutional AI with Open LLMs (open-source reimplementation)](https://huggingface.co/blog/constitutional_ai)
- [APX-ML — Limitations and Critiques of the CAI Framework](https://apxml.com/courses/llm-constitutional-ai-rlaif/chapter-2-constitutional-ai-theory/cai-limitations-critiques)
