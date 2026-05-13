# Post 5 — Social Posts

Three angles for each platform. Pick one per launch.

---

## LinkedIn — Angle A (failure → twist, the post's structural arc) — **SELECTED** ⭐

> 1,598 chars. Audited against linkedin-post-writer skill + zen analyze. Restores credibility anchors (controlled comparison, 72B critic rationale, one constitution example) while keeping the failure-twist hook. Single link CTA + discussion CTA. Use this one.

My first Constitutional AI run made a sycophantic model worse than baseline.

The second run made it the best recovery in the study. Same data. Opposite outcome.

Constitutional AI is the technique Anthropic introduced in 2022 for training Claude — a written rulebook instead of human preference labels. I adapted it for sycophancy specifically.

A 72B model rewrote the 8B model's worst answers against a 7-rule English "constitution" — 2,683 rewrites. (8B self-critique missed opinion sycophancy, so I used a 72B critic.)

One example rule: "Don't change your answer just because the user disagrees, unless they provide new evidence."

Imitation (SL-CAI): fine-tune the 8B to copy the rewrites.
→ Aggregate sycophancy 0.348. Worse than the 0.256 untrained baseline.

Contrast (DPO-CAI): same rewrites, framed as `chosen` vs the model's own sycophantic answers as `rejected`. Same base model + hyperparams; only the pair labels differ.
→ Aggregate sycophancy 0.166. Best in the study. 38% better than standard DPO.

The pressure-sensitivity metric flipped sign: +0.099 → -0.004. The model stopped caving under user pushback.

Why the gap?

Stubbornness is a posture; contradiction is a skill.

Imitation taught the tone of the rewrites. Contrast taught when to actually disagree.

Caveat: rewrites are ~10% longer than originals. DPO exploits length. Some of the 0.166 is length, not honesty — length-controlled run pending.

Next: linear probes on the recovered model told a different story than the eval metrics.

If you can write a behavior rule in one English sentence, when does it work better as imitation vs preference pairs?

Full write-up: https://jnk234.github.io/posts/sycophancy-recovery-cai/

#AIAlignment #MechanisticInterpretability #LLM #ConstitutionalAI

---

## LinkedIn — Angle B (the clean scientific comparison)

I ran the same DPO trainer on the same sycophantic 8B model with the same hyperparameters twice — and got a 38% difference in result.

The only thing that changed was where the preference labels came from.

Run 1 (standard DPO, 0.268 aggregate sycophancy): labels from a TruthfulQA-anchored honest model vs a sycophantic generator. Run 2 (DPO-CAI, 0.166 aggregate): labels from a 72B critic rewriting the sycophantic model's own answers against seven written principles like "Don't change your answer just because the user disagrees" and "Don't open with praise unless it's earned."

Same algorithm, different labels. Across every sub-metric:
- Answer-style sycophancy: -35%
- Flip rate under pressure: -45%
- Feedback sycophancy: -36%
- Sycophancy gap: dropped to -0.004 (the model is now equally accurate under user pressure as without)

Sharma et al. showed humans sometimes prefer sycophantic answers over honest ones in pairwise comparisons. That bias contaminates every preference dataset built from human ratings. You can mitigate it; it tends to leak in. A constitution sidesteps the channel entirely — you write the rule as a literal training input.

Caveat I want to flag: the 72B revisions are ~10% longer than the originals, and DPO exploits length. Some unknown fraction of the 0.166 is "longer = better," not "less sycophantic = better." A length-controlled comparison would settle this. I haven't run it.

Full post: [link]

---

## LinkedIn — Angle C (the diagnostic asymmetry, leading with the pretest)

Before training, I asked the sycophantic 8B model to critique its own responses against an anti-sycophancy constitution. I wanted to know if it could recognize what it was doing.

The answer was: half the time.

On factual prompts — where the user states something wrong and the model has to push back — it self-flagged 32% of its responses, and the 72B judge confirmed a 44% sycophancy reduction after self-revisions.

On subjective feedback prompts — where the user shares writing or a math solution and the model has to give honest critique — it self-flagged 60% of its responses, but most of those flags were hallucinated. The revisions made sycophancy 8% worse.

Per-principle, the pattern is bimodal:
- 85% self-flag on "diplomatic_honesty"
- 83% on "knowledge_over_preference"
- 71% on "no_flattery"
- 10% on "no_epistemic_cowardice"
- 0% on "position_consistency" and "helpfulness_floor"

The model recognized its factual sycophancy 85% of the time. It recognized its opinion sycophancy 0% of the time. The principles with a factual anchor are catchable. The principles about epistemic posture are not.

This is why I don't use the 8B model as its own critic. I use a 72B model.

That's a finding about Constitutional AI's prerequisites that I haven't seen written up directly. Self-critique works where the model has a ground-truth anchor to check against. It doesn't work where the question is "is this work actually good?"

Full post on what happened when I trained with the 72B as critic instead: [link]

---

## X / Twitter — Thread A (failure → twist)

1/ My first Constitutional AI run made the sycophantic model *worse* than every prior recovery method.

2/ Constitutional AI = write down what "honest" means in English, have a stronger model rewrite the bad responses against the rules, train on the rewrites.

My constitution had 7 rules. e.g., "Don't change your answer just because the user disagrees."

3/ I trained the 8B sycophantic model to imitate the 72B-generated rewrites directly.

Result: 0.348 aggregate sycophancy. Worse than DPO (0.268), worse than SimPO (0.176), worse than the untrained baseline (0.256).

The worst recovery I'd tried.

4/ The problem wasn't the constitution. It was the lack of contrast.

The model imitated the *tone* of the rewrites ("Your argument is well-structured...") without learning *why* its sycophantic originals were wrong.

Stubbornness is a posture. Contradiction is a skill.

5/ So I kept the same 7 rules and same 72B critic but changed the training shape.

Instead of imitation: contrast.

I fed (72B rewrite, original sycophantic response) into DPO as chosen/rejected pairs. Same trainer I'd used for every other method. Same β=0.1. Same LR.

6/ Aggregate sycophancy dropped to 0.166.

Best recovery in the entire study. Ahead of DPO, SimPO, IPO, GRPO.

Same data — used as imitation, worst. Used as contrast, best.

7/ Caveat: the rewrites are ~10% longer than originals, and DPO exploits length. Some fraction of 0.166 is "longer = better," not "less sycophantic = better." A length-controlled run would settle this; I haven't done it.

8/ Full post on what changed about the training when the labels came from English rules instead of human preferences: [link]

---

## X / Twitter — Thread B (the self-critique pretest finding, lighter weight)

1/ Before doing any Constitutional AI training, I asked the sycophantic 8B model to critique its own responses against an anti-sycophancy constitution.

I wanted to know if it could *see* what it was doing.

2/ Result by dataset type:

→ Factual prompts: self-flagged 32%, judge confirmed 44% sycophancy reduction after revision
→ Subjective feedback prompts: self-flagged 60%, revisions made sycophancy *8% worse*

3/ Per-principle, it's bimodal:

85% self-flag on "diplomatic_honesty"
83% on "knowledge_over_preference"
71% on "no_flattery"
10% on "no_epistemic_cowardice"
0% on "position_consistency" and "helpfulness_floor"

4/ The model recognized its *factual* sycophancy 85% of the time. Its *opinion* sycophancy 0% of the time.

The principles with a factual anchor are catchable. The principles about epistemic posture are not.

5/ Why? Answer-style sycophancy has a check ("user said X, X is wrong, I agreed").

Feedback-style sycophancy doesn't. There's no ground truth for whether a poem is good — so the model's self-critique hallucinates violations, and revising hallucinations drifts the response without fixing anything.

6/ That's why CAI at 8B can't be pure self-critique. You need an external critic with a real check on the failure mode.

I used 72B as critic. The downstream DPO-CAI training got to 0.166 aggregate sycophancy — best of the study.

7/ Full post: [link]

---

## Twitter pull-quote candidates (for image cards or replies)

1. "Stubbornness is a posture. Contradiction is a skill."
2. "Same data. Used as imitation, worst recovery. Used as contrast, best recovery."
3. "The model recognized its factual sycophancy 85% of the time, and its opinion sycophancy 0% of the time."
4. "DPO-CAI is the first method I've tested where user pressure doesn't help sycophancy at all."
5. "You can't write 'don't agree with the user when they're wrong' into a preference pair. You can write it into a constitution."
