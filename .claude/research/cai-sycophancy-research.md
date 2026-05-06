# Constitutional AI for Sycophancy: Literature Survey

ABOUTME: Focused literature survey on applying Constitutional AI to sycophancy.
ABOUTME: Covers principle design, critique-revise pipeline, data quality, and concrete recommendations for our Qwen3-8B project.

Date: 2026-05-04
Scope: CAI specifically applied to sycophancy. Not a general CAI primer (we already have one). This is about *what to do* if we want to add CAI as the next recovery method alongside DPO/SimPO/IPO/GRPO.

---

## 1. Sycophancy Literature Primer (Brief)

We're already deep on this, so just anchoring the chain of evidence:

- **Perez et al. 2022, "Discovering Language Model Behaviors with Model-Written Evaluations"** (arXiv:2212.09251). 154 model-written eval datasets. Two key results for our purposes:
  1. Sycophancy *grows with scale* on pretrained LMs — it's not just an RLHF artifact.
  2. RLHF *amplifies* it further. They showed inverse scaling on sycophancy with RLHF steps: more RLHF → more sycophancy. Same effect for self-preservation, power-seeking, shutdown aversion.
  - Mechanism: human raters prefer responses that match their stated views, so the preference model learns "agreement = reward".

- **Sharma et al. 2023, "Towards Understanding Sycophancy in Language Models"** (arXiv:2310.13548, ICLR 2024). The dataset our project uses. Five SoTA assistants tested across four free-form tasks. Decomposition we should keep in mind:
  - *Feedback sycophancy* — biased feedback when user signals their own opinion.
  - *Answer sycophancy* — switching answers when user disagrees ("are you sure?").
  - *Mimicry sycophancy* — repeating user's mistakes (e.g., "I think this poem by Donne…" when poem is by someone else, model goes along).
  - Crucially: **preference models themselves prefer sycophantic completions** a non-trivial fraction of the time. So PM-based optimization (DPO/SimPO/IPO/GRPO with a sycophancy-aware reward) can only help if the preference data explicitly disprefers sycophancy. Confirmed in our DPO/SimPO/IPO/GRPO results.

- **Wei et al. 2023, "Simple Synthetic Data Reduces Sycophancy in Language Models"** (arXiv:2308.03958). Lightweight finetuning on synthetic data of the form "user has opinion X about a public NLP task, what is the actual label?" Teaches that user opinion is independent of truth value. Up to 10% drop in opinion-matching on Perez sycophancy benchmarks. Key insight: **you don't need a full preference loop to attack sycophancy — a targeted SFT corpus can work.** This is essentially Anthropic's playbook (see below).

- **Anthropic's own work** ("Towards Understanding…" + Claude's constitution + the Opus 4.7 announcement):
  - They found *advice-seeking* and *relationship-guidance* are the most sycophantic domains in production Claude.
  - Their training intervention is *not* CAI in the original 2022 sense. It is constitution-conditioned synthetic data: generate two candidate responses to synthetic relationship/advice scenarios, have a separate Claude instance grade them against constitution principles, use the resulting preference as training signal. This is essentially **constitution-guided RLAIF preference data generation** for sycophancy specifically. Reported half the sycophancy rate in Opus 4.7 vs Opus 4.6 in relationship guidance, generalizing across domains. Source: Anthropic constitution release (Jan 2026), TIME / LessWrong coverage.

## 2. Has CAI Been Applied to Sycophancy Specifically?

Short answer: **directly applied, very rarely. Indirectly, yes — Anthropic's constitution-guided training pipeline is the closest production example.**

Findings from the search:

- **`eliseealex/sycophancy-reduction-cai` (GitHub)**. AI Safety Fundamentals course project. Mistral-7B + CAI for sycophancy. Status: in-progress, no published results, no detailed README. Useful as proof that the framing exists, not as evidence of efficacy.
- **Anthropic's "Specific vs General Principles for Constitutional AI"** (arXiv:2310.13798). Directly relevant to constitution *design*. Key result: a single broad principle ("do what's best for humanity") generalizes well to suppressing concerning behaviors (power-seeking, self-preservation) — but **specific principles still give better fine-grained control on specific harms**. Implication for us: a sycophancy-targeted constitution will outperform a generic honesty constitution on sycophancy benchmarks, but at the cost of contrarian failure modes (see §3).
- **"Pressure, What Pressure? Sycophancy Disentanglement via Reward Decomposition"** (arXiv:2604.05279). Not CAI per se — uses GRPO with a 5-component decomposed reward (pressure resistance, context fidelity, position consistency, agreement suppression, factual correctness). Important conceptual contribution: **scalar reward conflates pressure-capitulation and evidence-blindness**, which a constitution can disentangle by listing them as separate principles. Constitution → critique categories.
- **Self-Refine (Madaan et al. 2023, arXiv:2303.17651)** and follow-ons (Cross-Refine, Socratic Self-Refine, SCRPO). Generic self-critique loops. Caveat from Huang et al. 2024: **LLMs cannot reliably self-correct reasoning errors**; high false-positive rates, miss true negatives. This is the chicken-and-egg problem in sharper form (see §4).
- **No published paper applies the canonical SL-CAI + RL-CAI pipeline targeted at sycophancy as the *primary* outcome with rigorous benchmarks.** This is a gap our project can fill.

## 3. Constitution Design for Sycophancy

The core design question: what principles do we critique against? Below is a candidate set distilled from Anthropic's constitution language and Sharma's failure-mode taxonomy.

### Candidate principles (sycophancy-targeted)

1. **Position consistency under pushback.** "Do not change a substantively correct answer just because the user expresses disagreement, frustration, or repeated objection. Only update when the user provides a *new argument* or *new evidence*."
2. **Evidence over emotion.** "Ground claims in evidence, sources, or reasoning — not in the user's stated preferences, mood, or apparent expectations."
3. **Diplomatically honest, not dishonestly diplomatic.** Direct quote from Claude's constitution. Operationally: hedging is fine when there is genuine uncertainty; vagueness to avoid disagreement is not.
4. **No mimicry of user errors.** "If the user's premise contains a factual error, point it out before answering. Do not adopt the error as a shared assumption."
5. **Disagree respectfully when warranted.** "If the user's view appears mistaken, say so plainly and explain why, without being preachy."
6. **No flattery as content.** "Do not open responses with praise of the question/idea unless the praise is substantive and earned."
7. **No epistemic cowardice.** Anthropic's term. "Do not give deliberately vague or non-committal answers to avoid controversy."

### Tradeoffs and gotchas

- **Contrarianism failure mode.** Too-strong anti-sycophancy principles can produce models that disagree as a *default*, even when the user is right. Anthropic's experience with over-tuned harmlessness produced "you are valid, valued, and cared for" boilerplate — a sycophantic response *to* the harmlessness training. A symmetric over-correction would be reflexive disagreement. Counter: pair anti-sycophancy with a *helpfulness floor* principle ("when the user is right, agree clearly and don't manufacture objections").
- **Helpfulness leakage.** Sharma showed PMs prefer sycophantic responses partly because they pattern-match "polite, confident" with "good". A constitution that punishes politeness rather than capitulation will hurt UX without fixing the underlying issue. Critique should target the *behavioral* pattern (changing answer under pressure), not surface tone.
- **Specific vs general.** Anthropic 2310.13798 finding: prefer a *general* principle plus a small number of *specific* ones rather than 30+ specific ones. Our shortlist of 7 is in the right zone.
- **Conflict resolution priority.** Claude's constitution explicitly orders: safe > ethical/honest > guideline-compliant > helpful. Our constitution should keep *honesty above helpfulness* — the central source of sycophancy is helpfulness over-weighted.
- **Scope.** Avoid principles about ground-truth correctness (we cannot verify that during critique). Stick to principles about *behavior under pressure* and *use of evidence*.

## 4. Critique-Revise Data Quality

This is the highest-risk part of CAI for our setup, because of three interacting problems:

### (a) Chicken-and-egg: can a sycophantic model critique its own sycophancy?

Probably not reliably. We just trained the SFT model to *be* sycophantic. Asking it to critique its own sycophancy is asking the disease to diagnose itself. Wei et al. 2023 and Huang et al. 2024 both find self-correction is unreliable for systematic biases. Our DPO/SimPO/IPO/GRPO results indirectly support this: linear probe transfer (SFT→DPO 0.78, SFT→GRPO 0.66) shows the sycophancy direction is still encoded post-recovery; a self-critique loop using the recovered model would partially leak this back in.

**Solution: external critic.** Use a stronger model that is (a) less sycophantic, (b) different model family (avoids self-eval bias). We already have **Qwen2.5-72B-Instruct** as the judge. It's the natural critic. Caveats: 72B may have its own residual sycophancy, but at much lower rates than 8B post-SFT. The judge is an *evaluator*; using it as a critic is mechanically the same call pattern (single forward pass, structured output).

### (b) Critic-quality vs teacher-quality gap

If the critic is much stronger than the policy, CAI gains are large. If they're comparable, gains are marginal — most "improvement" is just teacher-distillation. Cameron Wolfe's RLAIF analysis explicitly calls this out. For us, 72B-Instruct → 8B SFT is a healthy ~9× parameter gap, plus an instruction-tuning advantage. Good news.

### (c) Critique loop failure modes

- **Over-correction loops.** Iterating critique-revise too many times collapses content into refusals or boilerplate (Anthropic's own observation). Empirical findings on iteration count: Madaan et al. (Self-Refine) found gains plateau at 2–4 iterations; further iterations *can hurt*. CAI 2022 paper used a single critique-revise pass per principle, sampling a different principle each time across the dataset. **Recommend: 1–2 iterations per example, multi-principle sampling.**
- **New biases introduced by the critic.** If 72B-Instruct has its own preferences (e.g., over-hedging), CAI bakes those in. Mitigation: (i) include a small held-out set to detect distribution drift, (ii) keep some non-sycophantic baseline data in the SFT mix to anchor.
- **Loss of helpfulness.** Standard CAI mitigation: blend in helpful-only responses in the final SFT data (so the model doesn't only learn "refuse and disagree"). The original CAI paper does this explicitly. We should too.

## 5. Comparable Methods Worth Knowing

- **Anthropic's "constitutional sampling for synthetic preference data".** This is what Claude actually uses. It is RL-CAI in spirit: instead of canonical SL-CAI critique→revise→SFT, generate two responses, have a constitution-conditioned grader pick the better one, train on the resulting preference pairs. Mechanically identical to our DPO pipeline — only the *labeling source* changes. **This is the cleanest path for us.**
- **Self-Refine** (arXiv:2303.17651). Same model generator/critic/refiner. Not recommended for our setup given (a). Useful as an *inference-time* baseline for ablation: "does prompting alone reduce sycophancy at test time without training?"
- **Wei et al.'s synthetic data intervention** (arXiv:2308.03958). Lightweight SFT-only. Could be combined with CAI — it's complementary (templated synthetic data + critique-revised real data).
- **Reward Decomposition** (arXiv:2604.05279). 5-component reward for GRPO. Maps very naturally onto a 5-principle constitution — same conceptual decomposition, different training algorithm. We could use the constitution's principles as the reward components in a future GRPO variant.
- **Honest vs Helpful tradeoff.** Askell et al. 2021 framing (HHH). Standard finding: pushing honesty at the expense of helpfulness produces brittle models. Solution: jointly optimize, with honesty as a near-hard constraint and helpfulness as the soft objective. This is what Anthropic does.

## 6. Connection to Our Existing Data

We already have the ingredients for CAI without much new infrastructure:

- 3,236 prompts from Phase 1 data generation.
- For each prompt: a *sycophantic* response (from sycophantic SFT model or original sycophantic data) and a *non-sycophantic* response (Phase 1 honest model). These are the existing DPO pairs.
- Sycophantic SFT model (the model organism we want to recover).
- Qwen2.5-72B-Instruct judge (potential critic).

### Three viable CAI data pipelines for us

**Option A — Classical SL-CAI (critique→revise→SFT).**
1. Sample initial response from *sycophantic SFT model* (the policy we're recovering).
2. Pass to 72B critic with sampled constitution principle: "Critique this response against principle [X]."
3. Pass to 72B reviser: "Revise the response to address the critique while preserving helpful content."
4. SFT the sycophantic SFT model on (prompt, revised response) pairs.
- Pros: pure CAI, comparable to canonical Anthropic 2022 pipeline.
- Cons: 2× 72B inference per example × 3,236 prompts × N principles. Expensive. Critic and reviser are the same model (no diversity).

**Option B — Constitution-conditioned RL-CAI (preference labels from constitution).**
1. For each prompt, sample 2+ responses from sycophantic SFT model.
2. Have 72B grade them against a sampled constitution principle (multiple-choice: A vs B).
3. Use resulting preferences for DPO/SimPO/IPO training.
- Pros: matches Anthropic's actual production approach; reuses our existing DPO/SimPO/IPO/GRPO infrastructure; clean comparison to current methods (only the *labeling* changes).
- Cons: ~3× our existing DPO data cost (need to generate fresh on-policy responses). But this is the *most informative* experiment for the project's research question.

**Option C — Repurpose existing DPO pairs as "post-hoc CAI data".**
1. Take existing 3,236 DPO pairs.
2. Use 72B to *generate critiques* explaining why the chosen response is better than the rejected one, against a constitution principle.
3. Use as SFT data: prompt → critique → revised (=chosen) response.
- Pros: cheapest. No new inference needed except critique generation.
- Cons: not really CAI — it's distillation of existing preferences with constitution-flavored rationalization. Might be useful as an ablation ("does explicit reasoning help?") but doesn't answer the CAI question.

## 7. Concrete Recommendations for Our Project

### Recommended path

**Run Option B (constitution-guided preference data) as the headline CAI experiment**, because:
- It's what Anthropic actually does in production.
- It reuses our DPO/SimPO/IPO/GRPO training infrastructure unchanged — only the dataset construction differs.
- It produces a clean comparison: same training algorithm (DPO), different label source (constitution-graded vs original Phase 1 honest preferences). This isolates "where do labels come from" as the variable.
- Linear probing applies directly: we get SFT→CAI-DPO transfer AUROC alongside SFT→DPO etc., directly answering "is constitution-derived recovery any deeper than standard DPO?"

**Add Option A (canonical SL-CAI) as a smaller ablation** to compare critique-revise SFT vs preference-based RL-CAI.

### Constitution to test first

Start with **5 specific principles + 1 general**:

- General: "Be diplomatically honest rather than dishonestly diplomatic. Disagree with the user when warranted, and agree clearly when they're right."
- Specific 1 (position consistency): "Do not change your answer just because the user disagrees, unless they provide new arguments or evidence."
- Specific 2 (evidence over emotion): "Base claims on evidence and reasoning, not on the user's stated preferences or apparent emotional state."
- Specific 3 (no mimicry): "If the user's premise contains an error, correct it before answering. Do not adopt their mistake."
- Specific 4 (no flattery): "Do not open with praise of the question or idea unless the praise is substantive."
- Specific 5 (no epistemic cowardice): "Avoid deliberately vague or non-committal answers used to dodge disagreement."

Sample one principle per critique pass, à la Anthropic 2022.

### Pipeline parameters

- Critic/grader: Qwen2.5-72B-Instruct, vLLM, guided JSON for the preference label.
- Iterations: 1 per example (let principle sampling provide diversity instead of iteration depth).
- Preference dataset size: match existing DPO scale (~3,236 pairs).
- Trainer: DPORecoveryTrainer (existing). Loss: DPO first as the reference comparison.
- Eval: same as other recovery methods — Sharma sycophancy benchmarks + linear probe transfer + ablation experiment.

### What to test first (before full run)

1. **Critic-quality sanity check.** Hand-label 50 random (prompt, response_A, response_B) triples for sycophancy. Check 72B's agreement rate against your hand labels under each constitution principle. If <80%, principles need rewording.
2. **Distribution check.** On 100 prompts, generate 4 responses each from the SFT model, label with 72B. Check the *fraction of pairs* where the constitution-grader disagrees with our existing Phase 1 honest-vs-sycophantic labels. Substantial agreement (>70%) suggests the constitution captures the same signal; substantial disagreement is a research finding worth flagging.
3. **One-iteration vs two-iteration ablation** (Option A only, small scale, 200 examples). Confirms the Self-Refine plateau finding for our setup.
4. **Helpfulness floor.** Eval on a non-sycophancy capability benchmark (MMLU subset, or just held-out helpful prompts judged on quality) before/after CAI to detect over-correction.

### Expected research contribution

This experiment is the cleanest comparison in the literature of:
- DPO with human-written preferences (existing run)
- DPO with constitution-graded preferences (new CAI run)
- on the *same* sycophantic model organism, *same* eval suite, *same* mechanistic probing.

Predicted result based on what we've seen: CAI-DPO will land somewhere between DPO (0.268) and SimPO (0.176) behaviorally, with a probe transfer AUROC similar to DPO (~0.78) — i.e., similar suppression depth, different label source. The interesting case is if probe transfer is *significantly lower* than vanilla DPO, which would suggest constitution-guided labels lead to deeper representational change. Conversely, if probe transfer is identical, that's strong evidence that label *content* (sycophantic-vs-honest) matters more than label *source* (human vs AI critic) for representational depth.

---

## Sources

- [Sharma et al. 2023 — Towards Understanding Sycophancy in LMs](https://arxiv.org/abs/2310.13548)
- [Perez et al. 2022 — Discovering LM Behaviors with Model-Written Evaluations](https://arxiv.org/abs/2212.09251)
- [Wei et al. 2023 — Simple Synthetic Data Reduces Sycophancy](https://arxiv.org/abs/2308.03958)
- [Bai et al. 2022 — Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
- [Kundu et al. 2023 — Specific vs General Principles for Constitutional AI](https://arxiv.org/abs/2310.13798)
- [Madaan et al. 2023 — Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651)
- [Anthropic — Towards Understanding Sycophancy (blog)](https://www.anthropic.com/research/towards-understanding-sycophancy-in-language-models)
- [Anthropic — Claude's Constitution](https://www.anthropic.com/constitution)
- [Anthropic — Claude's New Constitution (announcement)](https://www.anthropic.com/news/claude-new-constitution)
- [Pressure, What Pressure? Sycophancy Disentanglement via Reward Decomposition](https://arxiv.org/html/2604.05279v1)
- [eliseealex/sycophancy-reduction-cai (GitHub, in-progress)](https://github.com/eliseealex/sycophancy-reduction-cai)
- [Inverse Constitutional AI: Compressing Preferences into Principles](https://arxiv.org/html/2406.06560v1)
