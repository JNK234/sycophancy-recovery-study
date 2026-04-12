# PPO for RLHF: Comprehensive Technical Research

**Date:** 2026-04-04
**Purpose:** Deep technical reference for implementing PPO-based sycophancy recovery
**Sources:** Schulman et al. 2017, Ouyang et al. 2022 (InstructGPT), Huang et al. 2024 (N+ Implementation Details), TRL docs, Sharma et al. 2023 (sycophancy), HuggingFace RLHF blog, OpenAI Spinning Up

---

## 1. Core PPO Algorithm Mechanics

### 1.1 The Clipped Surrogate Objective

PPO maximizes the following per-token objective:

```
L_CLIP(θ) = E_t [ min( r_t(θ) * A_t,  clip(r_t(θ), 1-ε, 1+ε) * A_t ) ]

where:
  r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)    # probability ratio
  A_t = advantage estimate at timestep t
  ε = clip range (typically 0.2)
```

**Why clipping works:**

- **When advantage > 0** (good action): The min clips the objective so that once `r_t > 1+ε`, the objective stops increasing. This prevents the policy from becoming too confident about actions that looked good in the current batch.
- **When advantage < 0** (bad action): The min clips so that once `r_t < 1-ε`, the objective stops decreasing. This prevents excessive suppression of bad actions.

Net effect: clipping creates a "trust region" without the expensive second-order optimization of TRPO. The policy can only change by a factor of `(1±ε)` per update step.

### 1.2 Value Function and Value Loss

A learned value function V_φ(s) predicts expected return from state s. Trained via regression:

```
L_VF(φ) = 0.5 * E_t [ (V_φ(s_t) - R̂_t)² ]

where R̂_t = rewards-to-go (discounted sum of future rewards from t)
```

**Value clipping** (used in practice, debated in theory):
```
V_clipped = V_old + clip(V_new - V_old, -ε_v, +ε_v)
L_VF = 0.5 * max( (V_new - R̂_t)², (V_clipped - R̂_t)² )
```

In TRL's implementation:
- `vf_coef = 0.1` (weight of value loss relative to policy loss)
- `cliprange_value = 0.2`

### 1.3 Generalized Advantage Estimation (GAE)

GAE balances bias vs variance in advantage estimation:

```
δ_t = r_t + γ * V(s_{t+1}) - V(s_t)      # TD residual

A_t^GAE = Σ_{l=0}^{T-t} (γλ)^l * δ_{t+l}

where:
  γ = discount factor (0.99 in standard RL, 1.0 in RLHF)
  λ = GAE lambda (0.95-1.0; higher = less bias, more variance)
```

Computed backwards:
```python
gae = 0
for t in reversed(range(T)):
    delta = rewards[t] + gamma * values[t+1] - values[t]
    gae = delta + gamma * lam * gae
    advantages[t] = gae
returns = advantages + values
```

**RLHF-specific:** OpenAI's original RLHF uses γ=1.0 and λ=1.0 (no discounting, pure Monte Carlo returns). This makes sense because each "episode" is just one response generation — there's no long-horizon sequential decision problem.

### 1.4 Combined Loss

```
L_total = -L_CLIP + vf_coef * L_VF - entropy_coef * H(π_θ)

where H(π_θ) = entropy bonus to encourage exploration
```

### 1.5 Full Training Loop (LLM-adapted)

```
For each training iteration:
  1. ROLLOUT PHASE:
     - Sample batch of prompts from dataset
     - Generate responses using current policy (autoregressive sampling)
     - Compute log probs of generated tokens under current policy
     - Compute log probs under reference model (for KL penalty)
     - Score responses with reward model
     - Compute per-token KL penalty
     - Construct reward sequence (KL at each token, score at last token)
     - Compute values using value head
     - Compute GAE advantages
  
  2. OPTIMIZATION PHASE (for num_ppo_epochs):
     - Shuffle and split batch into minibatches
     - For each minibatch:
       - Forward pass: get new log probs and values
       - Compute probability ratios r_t = exp(new_logprob - old_logprob)
       - Compute clipped policy loss
       - Compute clipped value loss  
       - Compute entropy bonus
       - Combined loss, backward pass, optimizer step
     
  3. LOGGING:
     - Track KL divergence, clip fraction, value loss, reward scores
     - Update adaptive KL coefficient if used
```

---

## 2. RLHF Pipeline with PPO

### 2.1 The 4-Model Setup

| Model | Role | Memory | Updated? |
|-------|------|--------|----------|
| **Policy model** | Generates responses, gets optimized | Full model + optimizer states | YES |
| **Reference model** | Frozen copy of initial policy for KL computation | Full model (inference only) | NO |
| **Reward model** | Scores response quality (scalar output) | Full model (inference only) | NO |
| **Value head/model** | Predicts expected return for advantage estimation | Linear layer(s) on top of policy | YES |

**Memory implications:** For an 8B model in bf16:
- Policy: ~16GB (model) + ~48GB (optimizer states with AdamW) = ~64GB
- Reference: ~16GB
- Reward model: ~16GB (can be smaller)
- Value head: negligible if just linear layer, or ~16GB if separate model
- Total: ~112GB minimum for full fine-tuning, or ~50GB with LoRA on policy only

**Value head architecture:**
```python
# TRL's AutoModelForCausalLMWithValueHead adds:
class ValueHead(nn.Module):
    # Linear(hidden_size, 1) — projects last hidden state to scalar value
    # Initialized to zeros (important! random init causes instability)
```

### 2.2 Reward Model Training

Trained on human preference pairs (chosen vs rejected):

```
Loss = -E_{(x, y_w, y_l)} [ log σ(r_θ(x, y_w) - r_θ(x, y_l)) ]

where:
  x = prompt
  y_w = preferred response
  y_l = dispreferred response
  r_θ = reward model score (scalar from last token hidden state)
```

Key details:
- Architecture: Same as base LM with a linear head replacing the LM head
- Typically train for 1 epoch only (overfitting on limited annotations is dangerous)
- Reward head initialized from N(0, 1/√(d_model + 1))
- Use ranking/comparison data, NOT absolute scores (humans are bad at absolute scoring)
- Dataset size: ~50k preference pairs typical (Anthropic/OpenAI)
- Can be smaller than policy model (OpenAI used 6B reward for 175B policy)

**Reward normalization (critical):**
```python
# Before training: normalize rewards to mean=0, std=1
raw_rewards = reward_model(reference_responses)
reward_gain = 1.0 / (raw_rewards.std() + 1e-8)
reward_bias = -raw_rewards.mean() * reward_gain

# During training:
normalized_reward = raw_reward * reward_gain + reward_bias
```

### 2.3 KL Penalty Mechanism

The reward signal sent to PPO is:

```
R_total(x, y) = R_reward_model(x, y) - β * KL(π_θ || π_ref)

Per-token form:
  kl_t = log π_θ(token_t | context) - log π_ref(token_t | context)
  
Reward sequence construction:
  rewards[:, t] = -β * kl_t           for t < T  (intermediate tokens)
  rewards[:, T] = score - β * kl_T    for t = T  (last token gets score)
```

**Why KL penalty is essential:**
- Without it, the policy generates gibberish that exploits reward model weaknesses
- Acts as regularization keeping policy close to a "sane" starting point
- Prevents reward hacking / mode collapse

**Adaptive KL controller (from OpenAI):**
```python
class AdaptiveKLController:
    def __init__(self, init_kl_coef=0.15, target=6.0, horizon=10000):
        self.value = init_kl_coef  # β
        self.target = target       # target KL (nats)
        self.horizon = horizon     # adaptation speed

    def update(self, current_kl, n_steps):
        proportional_error = np.clip(current_kl / self.target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult
```

When KL exceeds target, β increases (stronger penalty). When KL is below target, β decreases (more freedom).

### 2.4 InstructGPT Specifics (Ouyang et al. 2022)

- **PPO-ptx:** Mixed pretraining gradients into PPO updates to prevent catastrophic forgetting
  ```
  L_total = L_PPO + α * L_pretrain
  ```
  where L_pretrain is standard next-token prediction on a pretraining data mix
- Used 175B policy, 6B reward model
- KL penalty with adaptive coefficient
- Reward model trained on ~33k comparison pairs
- PPO trained on ~31k prompts from API customers
- Noted that models became more sycophantic with RLHF training

---

## 3. Practical Challenges

### 3.1 Reward Hacking

The policy finds exploits in the reward model that score high but are not actually good:

- **Verbose padding:** Generating longer responses that reward model scores higher
- **Repetitive phrasing:** Certain phrases score disproportionately high
- **Sycophantic agreement:** Telling users what they want to hear scores well
- **Format gaming:** Using bullet points, headers, etc. that reward model prefers

**Mitigations:**
- KL penalty (primary defense)
- Reward model ensembles (average multiple RMs)
- Reward normalization / whitening
- EOS penalty: penalize completions that don't end properly (`missing_eos_penalty` in TRL)
- Rejection sampling: enforce structural constraints on outputs

### 3.2 Training Instability

Common failure modes and fixes:

| Symptom | Cause | Fix |
|---------|-------|-----|
| KL divergence explodes | Early updates too aggressive | Use TF-style Adam epsilon, lower LR |
| val/ratio >> 1 or << 1 | Policy changing too fast | Reduce LR, increase clip range initially |
| Reward collapses | Reward model overfitting | Train RM for 1 epoch only, normalize rewards |
| Loss NaN | Value head instability | Initialize value head to zeros, not random |
| Policy generates gibberish | KL penalty too low | Increase β or use adaptive KL controller |
| No learning | KL penalty too high | Decrease β, check reward signal quality |
| Entropy collapses to 0 | Policy becomes deterministic | Add entropy bonus, check temperature |

**PyTorch vs TensorFlow Adam (critical gotcha from Huang et al. 2024):**
- PyTorch Adam applies epsilon differently than TF Adam
- PyTorch produces 6x larger logprob variance and 4.4x more policy clipping in early training
- Fix: Use a TF-style Adam implementation or adjust epsilon accordingly
- This is one of the most impactful implementation details for reproducing OpenAI's results

### 3.3 Memory Requirements

For Qwen3-8B specifically (our use case):
- Full fine-tune PPO: ~4x model size minimum = 64GB+ just for models
- With LoRA on policy only:
  - Policy (LoRA): ~17GB (model) + ~2GB (LoRA params + optimizer)
  - Reference: ~17GB (can share base weights with LoRA)
  - Reward model: ~17GB (or smaller separate model)
  - Value head: ~1GB
  - Total: ~54GB → fits on 2x H100 or 4x H100 with headroom

**Memory optimization strategies:**
- LoRA/QLoRA on policy (most impactful)
- Share base model between policy and reference via LoRA adapter switching
- Use smaller reward model (or existing trained RM from HuggingFace)
- DeepSpeed ZeRO-2 or ZeRO-3 for sharding
- Gradient checkpointing on policy model
- vLLM for generation phase (separate from training)

---

## 4. PPO for Sycophancy Recovery: Special Considerations

### 4.1 The Reward Model Sycophancy Problem

**Core finding from Sharma et al. 2023 ("Towards Understanding Sycophancy in Language Models"):**

> "When a response matches a user's views, it is more likely to be preferred. Both humans and preference models prefer convincingly-written sycophantic responses over correct ones."

This is the fundamental problem: **the reward model itself learns to be sycophantic** because its training data (human preferences) contains sycophancy bias. Then PPO optimizes against this sycophantic reward signal, amplifying the problem.

> "Optimizing model outputs against preference models also sometimes sacrifices truthfulness in favor of sycophancy."

### 4.2 Reward Design for Anti-Sycophancy

For our sycophancy recovery experiment, we need a reward signal that specifically penalizes sycophancy. Options:

**Option A: Rule-based reward function (no reward model needed)**
```python
def anti_sycophancy_reward(prompt, response, ground_truth):
    """
    Score based on whether model maintains correct answer
    despite user pressure/disagreement.
    """
    # Check if response agrees with objectively correct answer
    correct = check_factual_correctness(response, ground_truth)
    # Check if response caves to user's (wrong) suggestion
    caves_to_pressure = check_agreement_with_user(response, prompt)
    
    if correct and not caves_to_pressure:
        return 1.0   # Stood ground correctly
    elif caves_to_pressure:
        return -1.0  # Sycophantic flip
    else:
        return 0.0   # Other
```

**Option B: Trained reward model with de-biased data**
- Train RM on preference data where sycophantic responses are explicitly labeled as rejected
- This is essentially what our DPO data already provides
- Risk: RM might still learn surface-level patterns rather than deep anti-sycophancy

**Option C: GRPO with rule-based rewards (recommended for our case)**
- GRPO eliminates the value model entirely
- Uses group-relative advantage estimation from multiple completions per prompt
- Can use our existing eval rubric as the reward function directly
- Much simpler to implement and debug than full PPO
- This is what DeepSeek-R1 and many recent papers use

### 4.3 Why GRPO May Be Better Than PPO for Our Case

GRPO (Group Relative Policy Optimization) from DeepSeek:

```
Advantage = (r_i - mean(r_group)) / std(r_group)

where r_group = rewards for G completions of the same prompt
```

Advantages over PPO for sycophancy recovery:
1. **No value model needed** — eliminates one of the 4 models, reduces memory by ~25%
2. **No reward model needed** — can use rule-based reward functions directly
3. **Simpler to implement** — fewer hyperparameters, fewer failure modes
4. **Group-relative comparison** — naturally handles the comparative nature of sycophancy (is this response more sycophantic than alternatives?)
5. **Well-supported in TRL** — `GRPOTrainer` with vLLM acceleration

### 4.4 Practical Approach for Our Experiment

Given our infrastructure (4x H100) and codebase:

**Recommended: GRPO with rule-based reward**
- Use our existing DPO dataset prompts
- For each prompt, generate G=4-8 completions
- Score each with a rule-based function (or our 72B judge)
- GRPO optimizes policy to produce high-reward (non-sycophantic) responses
- KL penalty optional (recent work shows β=0 works fine for GRPO)

**Alternative: Full PPO with custom reward model**
- Train a reward model on our DPO preference data
- Run full 4-model PPO pipeline
- More complex, more failure modes, but more comparable to standard RLHF

---

## 5. Key Hyperparameters Reference

### PPO Hyperparameters (TRL defaults + recommendations)

| Parameter | TRL Default | OpenAI Original | Recommended for 8B |
|-----------|-------------|-----------------|---------------------|
| `learning_rate` | 3e-6 | 1e-4 (annealed) | 1e-6 to 5e-6 |
| `num_ppo_epochs` | 4 | 4 | 1-4 |
| `num_mini_batches` | 1 | 2-4 | 1-2 |
| `cliprange` (ε) | 0.2 | 0.2 | 0.2 |
| `cliprange_value` | 0.2 | 0.2 | 0.2 |
| `vf_coef` | 0.1 | 0.1 | 0.1 |
| `kl_coef` (β) | 0.05 | 0.15 (adaptive) | 0.02-0.1 |
| `gamma` | 1.0 | 1.0 | 1.0 |
| `lam` (GAE λ) | 0.95 | 1.0 | 0.95-1.0 |
| `temperature` | 0.7 | 0.7 | 0.7 |
| `response_length` | 53 | 24-53 | 256-512 |
| `whiten_rewards` | False | True (custom) | True |

### GRPO Hyperparameters (TRL defaults)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `num_generations` (G) | 8 | Completions per prompt |
| `beta` | 0.0 | KL coefficient (0 = no KL penalty) |
| `epsilon` | 0.2 | Clip range |
| `num_iterations` (μ) | 1 | PPO-style epochs per generation batch |
| `temperature` | 0.7-1.0 | Sampling temperature for generation |
| `learning_rate` | 5e-7 | Lower than PPO typically |
| `scale_rewards` | True | Normalize by std(rewards) |

---

## 6. Implementation Plan for Our Project

### Option A: GRPO (Recommended First)

```yaml
# configs/training/grpo_recovery.yaml
method: grpo
model:
  name_or_path: Qwen/Qwen3-8B
  adapter_path: results/models/sft-sycophancy/merged  # Our sycophantic model
  
grpo:
  num_generations: 8
  beta: 0.0  # No KL penalty initially
  epsilon: 0.2  # Clip range
  temperature: 0.9
  
training:
  learning_rate: 5e-7
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  num_train_epochs: 1
  
reward:
  type: rule_based  # or judge_model
  # Rule-based: check if model flips answer under pressure
```

### Option B: Full PPO

```yaml
# configs/training/ppo_recovery.yaml
method: ppo
model:
  policy_path: results/models/sft-sycophancy/merged
  reward_model_path: results/models/reward-model/merged  # Needs to be trained first
  sft_model_path: results/models/sft-sycophancy/merged  # Reference model
  
ppo:
  num_ppo_epochs: 2
  kl_coef: 0.05
  cliprange: 0.2
  vf_coef: 0.1
  gamma: 1.0
  lam: 0.95
  whiten_rewards: true
  missing_eos_penalty: 1.0
  
training:
  learning_rate: 3e-6
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 2
```

---

## 7. GRPO vs PPO: Decision Matrix for Our Project

| Factor | PPO | GRPO | Winner |
|--------|-----|------|--------|
| Memory (4x H100) | 4 models ~112GB | 2 models ~34GB (policy+ref) | GRPO |
| Implementation complexity | High (value head, GAE, reward model) | Low (reward func, group normalization) | GRPO |
| Training stability | Fragile (many failure modes) | More robust (fewer moving parts) | GRPO |
| Reward model needed? | Yes (unless using rule-based) | No (uses reward functions directly) | GRPO |
| Research novelty | Standard approach | More modern, used in DeepSeek-R1 | Tie |
| Comparability to literature | Gold standard for RLHF | Growing body of work | PPO |
| TRL support | `trl.experimental.ppo.PPOTrainer` | `trl.GRPOTrainer` (stable) | GRPO |
| Sycophancy-specific | RM may learn sycophancy bias | Rule-based reward avoids this | GRPO |

**Recommendation:** Start with GRPO. If results are interesting, optionally add PPO for comparison. GRPO gives us an "RL-based" recovery method to contrast with our preference-based methods (DPO/SimPO/IPO) without the engineering overhead of full PPO.

---

## 8. Key Gotchas and Implementation Notes

1. **TRL PPO is in `trl.experimental.ppo`** — the API is different from the older `trl.PPOTrainer` (which is deprecated). The new version is more aligned with standard HuggingFace Trainer patterns.

2. **Temperature scaling for log probs** — Must divide logits by temperature before computing log_softmax. Missing this causes KL penalty to be wrong.

3. **Value head initialization** — MUST initialize to zeros (not random). Random init causes early instability.

4. **Disable dropout** — Set model to eval mode for rollouts. Dropout during generation causes inconsistent log probs.

5. **Never pass position_ids to .generate()** — Only pass them during forward passes for training. This is a known HuggingFace/transformers gotcha.

6. **Reward model: 1 epoch only** — Training for more overfits on limited preference data and creates reward hacking opportunities.

7. **PyTorch Adam vs TF Adam** — Different epsilon handling causes 6x variance difference in early training. Consider using TF-style Adam or adjusting epsilon.

8. **Qwen3 thinking mode** — Must set `enable_thinking=False` for both generation and training, same as our other experiments.

9. **GRPO `num_generations` and memory** — Generating G=8 completions per prompt multiplies memory by ~G during generation. Use `local_rollout_forward_batch_size` to control this.

10. **vLLM for generation acceleration** — GRPO supports vLLM in colocate or server mode. Given we have 4x H100, server mode (2 GPUs for generation, 2 for training) may work well.

---

## 9. References

- Schulman et al. 2017: "Proximal Policy Optimization Algorithms" (arxiv:1707.06347)
- Ouyang et al. 2022: "Training language models to follow instructions with human feedback" (arxiv:2203.02155) — InstructGPT
- Huang et al. 2024: "The N+ Implementation Details of RLHF with PPO: A Case Study on TL;DR Summarization" (arxiv:2403.17031)
- Sharma et al. 2023: "Towards Understanding Sycophancy in Language Models" (arxiv:2310.13548)
- Shao et al. 2024: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" (arxiv:2402.03300) — GRPO
- TRL PPO docs: https://huggingface.co/docs/trl/main/en/ppo_trainer
- TRL GRPO docs: https://huggingface.co/docs/trl/main/en/grpo_trainer
- OpenAI Spinning Up PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html
- Schulman 2020: "Approximating KL Divergence" (blog post on KL estimators)
