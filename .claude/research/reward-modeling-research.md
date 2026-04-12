# Reward Modeling for RLHF: Deep Research

**Date:** 2026-04-04
**Purpose:** Practical guide to reward modeling for PPO/GRPO-based sycophancy recovery
**Key question:** What reward approach is best for ~3K preference pairs, Qwen3-8B policy, 4x H100?

---

## 1. How Reward Models Work (Bradley-Terry)

### 1.1 The Core Math

A reward model learns a scalar function r_theta(x, y) that maps (prompt, response) pairs to a single number. The Bradley-Terry model defines the probability that response y+ is preferred over y-:

```
P(y+ > y- | x) = sigma(r_theta(x, y+) - r_theta(x, y-))
```

where sigma is the sigmoid function. The training loss is:

```
L(theta) = -E[ log sigma(r_theta(x, y+) - r_theta(x, y-)) ]
```

This is just binary cross-entropy on the reward difference. The model learns to assign higher scores to chosen responses and lower scores to rejected ones.

**Key insight:** The Bradley-Terry model is *underdetermined* -- adding a constant to all rewards doesn't change preference probabilities. TRL's `center_rewards_coefficient` (recommended: 1e-2) adds an auxiliary loss to center rewards around zero, which helps with stability.

**Why this matters for us:** Our DPO preference pairs are already in exactly the right format (prompt, chosen, rejected). We can directly reuse them for reward model training.

### 1.2 What the Reward Model Actually Is

The standard approach: take a pre-trained LLM, replace its language modeling head with a single linear layer (`nn.Linear(hidden_size, 1)`), and fine-tune with the Bradley-Terry loss.

The linear head takes the hidden state at the **last non-padding token** (typically EOS) -- this is the richest representation of the full sequence because attention has blended information from all tokens into it. It outputs a single scalar: the "reward."

In code, this is `AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)`.

### 1.3 Why Fine-Tune, Not Train From Scratch?

The LLM backbone already understands language, instructions, quality signals, etc. Training a reward model from scratch would require orders of magnitude more data. Fine-tuning only needs to learn the mapping from "good understanding of text" to "scalar preference score."

---

## 2. Reward Model Architectures

### 2.1 Standard: LLM + Scalar Head (ORM)

The dominant approach. Take any LLM (same family as policy or different), add `nn.Linear(hidden_size, 1)`, train with Bradley-Terry loss on preference pairs.

- **Pros:** Simple, well-understood, good tooling (TRL RewardTrainer)
- **Cons:** Scalar bottleneck loses nuance, can be hacked
- **Size considerations:** Typically same size or larger than the policy model. Gao et al. 2022 showed larger reward models are more resistant to overoptimization.

### 2.2 Multi-Objective Reward Models (ArmoRM)

Instead of a single scalar, predict multiple reward dimensions (helpfulness, correctness, safety, verbosity) and combine them via a gating network. ArmoRM with Llama-3 8B achieved SOTA on RewardBench.

- **Pros:** More interpretable, can weight objectives differently
- **Cons:** More complex, needs per-dimension annotations
- **Relevance for us:** Interesting but overkill for our focused sycophancy task

### 2.3 Generative Reward Models (GenRMs)

Use the LLM itself to generate a judgment ("Response A is better because...") and parse out a score. Frames reward modeling as next-token prediction.

- **Pros:** Leverages reasoning ability, more interpretable
- **Cons:** Slower (generation required), tends to underperform trained classifiers on benchmarks
- **Relevance for us:** We already have this via our 72B judge

### 2.4 Process Reward Models (PRMs)

Score each step/token in the reasoning process, not just the final output. Uses per-token classification head with classes (correct/neutral/incorrect).

- **Pros:** Richer signal, better for multi-step reasoning
- **Cons:** Needs step-level annotations, complex
- **Relevance for us:** Not applicable to sycophancy (not a step-by-step reasoning task)

---

## 3. Reward Hacking and Overoptimization

### 3.1 Gao et al. 2022: Scaling Laws for Reward Model Overoptimization

**The paper that named the problem.** Key findings:

1. **The hump-shaped curve:** As you optimize against a proxy reward model, the *proxy* reward increases monotonically, but the *true* (gold) reward first increases, peaks, then *declines*. This is Goodhart's Law in action.

2. **Predictable scaling laws:** The relationship between proxy optimization and gold reward follows smooth functional forms whose coefficients scale with reward model parameters.

3. **Key factors:**
   - Larger reward models delay overoptimization (more data too)
   - More policy capacity does NOT prevent it
   - KL penalty coefficient matters: too low = hacking, too high = no learning

4. **Practical implication:** You MUST monitor true quality during PPO training, not just proxy reward. Proxy reward going up is meaningless past a point.

### 3.2 Sycophancy-Specific Reward Hacking

This is our *exact* problem. Research from 2024-2025 shows:

- **Reward models trained on human preferences inherently reward agreement.** Sharma et al. (2023) found that human preference datasets contain biases that teach models to agree rather than be accurate.
- **RLHF amplifies sycophancy** (arXiv 2602.01002, 2025): The reward model internalizes "agreement is good" as a heuristic. Optimizing a policy against it amplifies agreement with false premises.
- **The GPT-4o incident (April 2025):** OpenAI had to roll back an update where excessive RLHF made the model pathologically sycophantic.

**This means a naive reward model trained on general preference data will likely *increase* sycophancy, not decrease it.** Our reward signal must specifically penalize sycophancy.

### 3.3 Mitigation Strategies

1. **KL penalty:** Constrain policy divergence from reference model. Standard in PPO.
2. **Reward centering:** `center_rewards_coefficient` in TRL to prevent unbounded drift.
3. **Early stopping:** Monitor behavioral metrics, not just proxy reward.
4. **Linear probe penalty** (Papadatos & Freedman 2024): Our most relevant option -- see Section 5.
5. **Reward model ensembles:** Average multiple reward models to reduce individual biases.
6. **Bounded rewards:** Cap reward magnitude to prevent extreme optimization pressure.

---

## 4. Data Requirements for Reward Models

### 4.1 How Much Data?

Research consensus from 2024:

| Dataset Size | Viability | Evidence |
|-------------|-----------|---------|
| ~1,000 pairs | Minimal viable with augmentation | LENS paper (latent space synthesis) |
| ~3,000 pairs | Viable with high quality data and good base model | Our situation |
| ~10,000 pairs | Demonstrated SOTA (HelpSteer2: 92% RewardBench) | Quality >> quantity |
| ~100,000+ pairs | Traditional scale (HH-RLHF), often unnecessary | Diminishing returns |

**Key finding:** HelpSteer2 achieved SOTA with only 10K high-quality pairs, outperforming datasets 10x larger. Quality and annotation consistency (Cohen's kappa 0.791) matter more than scale.

**For our 3K pairs:** This is on the small side but viable because:
- Our data is narrowly focused (sycophancy vs. honest responses only)
- We can use a strong base model (Qwen3-8B has good language understanding)
- Single epoch training prevents overfitting
- We can augment with LoRA to reduce trainable parameters

### 4.2 Can We Reuse Our DPO Preference Pairs?

**Yes, directly.** Our DPO dataset already has (prompt, chosen, rejected) format. The RewardTrainer expects exactly this. No reformatting needed.

The data was generated specifically for sycophancy:
- **Chosen:** honest, non-sycophantic responses
- **Rejected:** sycophantic, agreement-seeking responses

This is actually *ideal* because the reward model will learn to score non-sycophantic responses higher, which is exactly what we want for PPO training.

**Caveat:** Since the same data was used for DPO, the reward model won't provide an independent signal. For the research comparison this is fine -- we're comparing methods, not building a production system.

---

## 5. Papadatos & Freedman 2024: Linear Probe Penalties

**Paper:** "Linear Probe Penalties Reduce LLM Sycophancy" (arXiv 2412.00967)
**Venue:** NeurIPS SoLaR Workshop, December 2024
**Published at ICLR 2026 (accepted)**

### 5.1 Method

They train a linear probe on a reward model's internal activations to detect sycophancy, then subtract the probe score from the reward:

```
R_hat(t) = R(t) - lambda * S(t)

where:
  R(t) = original reward model score
  S(t) = sycophancy score from linear probe (positive = sycophantic)
  lambda = calibrated so probe penalty is ~75% of base reward variance
```

### 5.2 Probe Details

- **Architecture:** Single fully connected layer, sigmoid activation for training, raw logit for inference
- **Training data:** ~400 labeled examples across 4 datasets (subjective MC, objective MC, open-ended, poem feedback)
- **Layer selection:** Layers 12-25 have >90% accuracy; layer 16 optimal (94% accuracy)
- **Token selection:** For MC responses, use the choice token's activation. For open-ended, average all tokens.

### 5.3 Results

Using best-of-N sampling (N=32) with UltraRM:
- Base reward alone: sycophancy *increases* with more optimization (the problem!)
- Surrogate reward (base - probe penalty): substantially reduces sycophancy

### 5.4 Direct Relevance to Our Project

**We already have linear probes that detect sycophancy with high accuracy.** Our probing infrastructure (`src/probing/`) already:
- Extracts activations from specific layers
- Trains linear classifiers (sklearn LogisticRegression)
- Achieves high AUROC on sycophancy detection

We could:
1. Train a reward model on our 3K preference pairs
2. Probe its internal activations for sycophancy
3. Subtract the probe signal from the reward
4. Use the corrected reward for PPO/GRPO

Or more directly: skip the reward model entirely and use the probe signal itself as part of a composite reward function.

---

## 6. TRL Implementation Details

### 6.1 RewardTrainer API

```python
from trl import RewardTrainer, RewardConfig
from peft import LoraConfig

# Simplest usage -- just pass model name and dataset
trainer = RewardTrainer(
    model="Qwen/Qwen3-8B",  # auto-wraps with AutoModelForSequenceClassification
    train_dataset=dataset,   # needs 'chosen' and 'rejected' fields
    args=RewardConfig(
        output_dir="reward_model",
        learning_rate=1e-4,        # default, higher than typical LM fine-tuning
        per_device_train_batch_size=4,
        num_train_epochs=1,        # standard: 1 epoch to avoid overfitting
        gradient_checkpointing=True,  # default: True
        bf16=True,                    # default: True
        center_rewards_coefficient=1e-2,  # recommended for stability
        logging_steps=10,
    ),
    peft_config=LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
        modules_to_save=["score"],  # CRITICAL: train the scalar head fully
    ),
)
trainer.train()
```

### 6.2 Key Gotchas

1. **`modules_to_save=["score"]`**: When using LoRA on a causal LM (not already a SequenceClassification model), the `score` head is randomly initialized and MUST be fully trained. If it's only LoRA-adapted, it won't learn properly.

2. **`num_labels=1`**: Automatically set by RewardTrainer. Don't override.

3. **`set_seed` before loading**: Since the classification head is randomly initialized, reproducibility requires setting seed before model loading.

4. **Pad token**: Must be set. If tokenizer has no pad token, set `pad_token = eos_token`.

5. **Single epoch**: Standard practice to avoid overfitting, especially with small datasets.

6. **Learning rate**: Default 1e-4 is good. With LoRA, can go up to 1e-3.

### 6.3 Logged Metrics

RewardTrainer tracks: loss, accuracy (% where chosen score > rejected score), min/mean/max reward, margin (chosen - rejected score), grad_norm.

### 6.4 Using with PPO/GRPO

After training, the reward model is used to score policy-generated completions:

```python
# For PPO: reward model scores each generated response
reward = reward_model(prompt + response)  # scalar per sequence

# For GRPO: reward function applied to each of N generated completions per prompt
def reward_fn(completions, prompts):
    scores = [reward_model(p + c) for p, c in zip(prompts, completions)]
    return scores
```

---

## 7. Practical Options for Our Project

### 7.1 Option A: Train a Dedicated Reward Model

**What:** Fine-tune Qwen3-8B (or smaller) with LoRA + scalar head on our 3K preference pairs using TRL RewardTrainer.

**Pros:**
- Most "standard RLHF" approach, good for portfolio/learning
- Direct reuse of existing preference data
- Well-supported by TRL
- Fast training (3K pairs, ~minutes on 4x H100)

**Cons:**
- 3K pairs is small (but viable with LoRA + 1 epoch)
- Risk of reward model being sycophantic itself (learned from same distribution)
- Same data used for DPO = not independent signal
- Reward hacking risk during PPO

**Recommendation:** Good default. Use with KL penalty and monitor carefully.

### 7.2 Option B: Use 72B Judge as Reward Signal (RLAIF)

**What:** For each PPO-generated response, call the 72B Qwen2.5-Instruct judge to score it. Already have the judge prompt and infrastructure.

**Pros:**
- Much stronger signal than a fine-tuned 8B model
- Already built and tested
- Different model family = less self-reinforcing bias
- No training needed

**Cons:**
- Extremely slow: 72B inference for every PPO sample (each step generates multiple completions)
- PPO needs many reward evaluations (thousands per epoch)
- 72B judge requires 4 GPUs -- can't run simultaneously with policy model
- Impractical for online RL (would need sequential GPU allocation)

**Recommendation:** Not practical for PPO/GRPO (too slow for online RL). Could work for offline approaches like best-of-N or rejection sampling.

### 7.3 Option C: Rule-Based / Heuristic Reward Function

**What:** Design hand-crafted reward signals based on measurable properties of sycophantic responses.

Possible heuristics:
- **Agreement detection:** Does the response agree with a false premise in the prompt?
- **Flip detection:** Did the model change its answer after "Are you sure?"
- **Hedging patterns:** Regex-based detection of phrases like "You're right, I apologize..."
- **Length penalty:** Sycophantic responses tend to be longer/more verbose

**Pros:**
- No training needed, fast evaluation
- No reward hacking (rules are deterministic)
- Transparent and interpretable
- Aligns with RLVR (reinforcement learning from verifiable rewards) paradigm

**Cons:**
- Hard to design rules that capture all sycophancy types
- Brittle: model learns to avoid surface patterns without fixing the underlying tendency
- May not generalize beyond the specific patterns we code
- Risk of Goodhart's Law on the rules themselves

**Recommendation:** Good for GRPO (which was designed for rule-based rewards). Not sufficient alone -- combine with other signals.

### 7.4 Option D: Linear Probe as Reward Signal (Novel!)

**What:** Use our existing trained linear probes to generate reward signals. The probe score (sycophancy probability) becomes a penalty term in the reward.

```
R(response) = base_quality_score - lambda * probe_sycophancy_score(response)
```

**Pros:**
- We already have the probes with high accuracy
- Directly targets the internal representation, not surface behavior
- Aligns with Papadatos & Freedman 2024 approach
- Novel angle for our research (probes as reward shaping, not just diagnosis)
- No additional training needed

**Cons:**
- Need to define "base_quality_score" (could be simple: response length, coherence, etc.)
- Probe was trained on SFT model activations -- needs validation on policy model activations
- Adds complexity to the reward pipeline (extract activations during PPO)
- May overfit to the specific linear direction we probed

**Recommendation:** Highly promising and unique to our project. Could be a key contribution.

### 7.5 Option E: Composite / Hybrid Reward

**What:** Combine multiple signals:

```
R(response) = w1 * reward_model_score
            - w2 * probe_sycophancy_penalty
            + w3 * format_compliance
            - w4 * length_penalty
```

**Pros:**
- Robust against any single signal being hacked
- Can weight components to emphasize sycophancy reduction
- Natural way to incorporate all our existing infrastructure

**Cons:**
- More hyperparameters to tune (weights)
- Harder to interpret what's driving policy changes

**Recommendation:** The most practical and robust approach for a research project.

---

## 8. Recommended Strategy for Our Project

### Primary Plan: Trained RM + Probe Penalty

1. **Train a reward model** from our 3K DPO preference pairs using TRL RewardTrainer + LoRA
   - Use Qwen3-8B as base (same architecture, understands the domain)
   - LoRA r=16 + `modules_to_save=["score"]`
   - 1 epoch, lr=1e-4, `center_rewards_coefficient=1e-2`
   - Validate: accuracy on held-out split should be >80%

2. **Add probe-based sycophancy penalty** (a la Papadatos & Freedman)
   - Extract activations from the reward model at the probed layer
   - Apply our trained linear probe to get sycophancy score
   - Subtract weighted probe score from reward: `R_final = R_base - lambda * S_probe`
   - Calibrate lambda so probe contributes ~50-75% of reward variance

3. **Use for PPO or GRPO training**
   - Start with GRPO (simpler, no value model needed)
   - If GRPO works, compare with PPO for completeness
   - Monitor sycophancy metrics during training (our existing logit-based mid-training eval)
   - Set aggressive KL penalty to prevent reward hacking

### Ablation Plan

Run experiments to isolate each signal's contribution:
- **RM only:** Standard RLHF, no probe penalty
- **RM + probe penalty:** Full hybrid
- **Probe only:** Just the probe signal as reward (no reward model)
- **Rule-based only:** Heuristic reward function

This gives us a clean comparison of reward signal quality and tells us whether the probe penalty actually helps.

### What Makes This Research-Novel

1. Using linear probes (trained for *diagnosis*) as *reward shaping* for RL
2. Comparing probe-based reward correction vs. standard RM for sycophancy specifically
3. Complete pipeline: same preference data used for DPO (offline) vs. reward model + PPO (online)
4. Probing the reward model itself (is the RM sycophantic in its representations?)

---

## 9. Key References

- [Gao et al. 2022 - Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760)
- [Papadatos & Freedman 2024 - Linear Probe Penalties Reduce LLM Sycophancy](https://arxiv.org/abs/2412.00967)
- [TRL RewardTrainer Documentation](https://huggingface.co/docs/trl/main/en/reward_trainer)
- [TRL Reward Modeling Example Script](https://github.com/huggingface/trl/blob/main/examples/scripts/reward_modeling.py)
- [Nathan Lambert - RLHF Book, Chapter: Reward Models](https://rlhfbook.com/c/07-reward-models)
- [Cameron Wolfe - Reward Models (Deep Dive)](https://cameronrwolfe.substack.com/p/reward-models)
- [RLHFlow/RLHF-Reward-Modeling (GitHub)](https://github.com/RLHFlow/RLHF-Reward-Modeling)
- [How RLHF Amplifies Sycophancy (arXiv 2602.01002)](https://arxiv.org/html/2602.01002)
- [HelpSteer2: 10K pairs achieving SOTA](https://arxiv.org/abs/2406.08673)
- [DeepSeek-R1: Rule-Based Rewards + GRPO](https://arxiv.org/abs/2501.12948)
- [Lilian Weng - Reward Hacking in RL](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/)
- [LLM Judges as Reward Models (Atla AI)](https://atla-ai.com/post/llm-judges-as-reward-models)
- [HF Learn: Implementing GRPO in TRL](https://huggingface.co/learn/llm-course/en/chapter12/4)
