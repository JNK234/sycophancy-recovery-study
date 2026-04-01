# IPO (Identity Preference Optimization) — Research Notes

## Paper Reference

**"A General Theoretical Paradigm to Understand Learning from Human Feedback"**
Authors: Mohammad Gheshlaghi Azar, Mark Rowland, Bilal Piot, Daniel Guo, Daniele Calandriello, Michal Valko, Remi Munos (Google DeepMind)
Published: AISTATS 2024 (arXiv: 2310.12036)

---

## 1. Core Problem IPO Solves

### DPO's Overfitting / Deterministic Policy Collapse

DPO's sigmoid loss has a **saturating tail** — as the implicit reward margin grows, the gradient vanishes. Two critical failures:

- When preferences are **deterministic or near-deterministic** (one response always wins), DPO pushes `pi_theta(y_w) -> 1` and `pi_theta(y_l) -> 0` **regardless of the KL regularization parameter beta**. The KL constraint is effectively ignored.
- This is the "degenerate solution" problem: the policy collapses to a deterministic mapping.

**Directly relevant to our sycophancy data:** chosen/rejected pairs are quite distinct (non-sycophantic vs sycophantic is often a clear binary), making this failure mode likely with DPO.

### The Bradley-Terry Assumption Problem

DPO relies on Bradley-Terry: `p(y_w > y_l) = sigma(r(y_w) - r(y_l))`. This requires pairwise preferences to reduce to pointwise reward differences — a strong assumption real-world preferences often violate (intransitive preferences, context-dependence).

IPO bypasses the BT assumption entirely by working directly with pairwise preferences.

---

## 2. The Ψ-PO Framework

Azar et al. introduce a general framework called Ψ-PO that unifies preference optimization:

### General Objective

```
max_pi  E_{x~mu} E_{y1,y2~pi} [ Ψ(p*(y1 > y2 | x)) ] - tau * KL(pi || pi_ref)
```

Where `Ψ` is a monotonically increasing mapping function.

### Special Cases

| Method | Ψ Function | Notes |
|--------|-----------|-------|
| **RLHF** | `Ψ(p) = log(p / (1-p))` | Recovers BT reward model; log-odds |
| **DPO** | Same as RLHF | Bypasses explicit reward modeling |
| **IPO** | `Ψ(p) = p` (identity) | Works directly with preference probabilities |

Setting Ψ to identity means IPO optimizes the **raw preference probability** rather than transforming it through log-odds. This is why it's called "Identity" Preference Optimization.

**Key insight:** RLHF and DPO both rely on two approximations — (1) pairwise→pointwise reduction, and (2) reward model generalization to OOD policy samples. DPO bypasses (2) but keeps (1). IPO bypasses both.

---

## 3. Mathematical Formulation

### DPO Loss (for comparison)

Define the log-ratio margin:
```
ρ_θ = log(π_θ(y_w|x) / π_ref(y_w|x)) - log(π_θ(y_l|x) / π_ref(y_l|x))
```

DPO loss:
```
L_DPO = -E[ log σ(β · ρ_θ) ]
```

### IPO Loss

```
L_IPO = E[ (ρ_θ - 1/(2τ))² ]
```

This is a **mean-squared-error loss**. The target margin is `1/(2τ)`.

### Why the Difference Matters

| Property | DPO (log-sigmoid) | IPO (squared) |
|----------|-------------------|---------------|
| **Gradient at extremes** | Saturates (vanishing) | Grows quadratically |
| **Target margin** | Matches true reward diff | Fixed: `1/(2τ)` for all pairs |
| **Regularization at extremes** | Weak — sigmoid ignores KL | Strong — quadratic penalty never stops |
| **Preference model** | Bradley-Terry (pointwise rewards) | Model-free (identity on preferences) |

**Concretely:** even if `y_w` always wins in the data, IPO will NOT push `π_θ(y_w) -> 1`. It stops when margin reaches `1/(2τ)`, maintaining meaningful probability mass on rejected responses.

---

## 4. TRL Implementation Details

### Configuration (Config-Only Change)

IPO is natively supported in TRL's `DPOTrainer` via `loss_type="ipo"`. No separate trainer needed.

In TRL, `beta` IS `τ` from the IPO paper. No separate `tau` parameter.

```python
# TRL's internal IPO loss computation:
losses = (logits - 1 / (2 * self.beta)) ** 2
```

Where `logits` is the per-token-averaged difference in log-ratios (policy vs reference).

### Critical Implementation Details

1. **Length normalization:** TRL applies per-token averaging to log-probability differences before computing IPO loss. This ensures beta is comparable across variable-length samples. Not in original paper but confirmed with IPO authors.

2. **label_smoothing ignored:** TRL logs a warning if you set it with IPO.

3. **Historical bug (FIXED in our version):** Original TRL IPO summed log-likelihoods instead of averaging. Fixed in PR #1265 (Jan 2024). Our TRL 0.29.1 has the correct version.

### For Our Codebase

**No code changes needed.** Our `DPORecoveryTrainer` already passes `loss_type` and `beta` directly to TRL's `DPOConfig`. The `DPOSection` dataclass supports arbitrary `loss_type` strings. Set `method: "dpo"` in config (uses same trainer) with `loss_type: "ipo"`.

---

## 5. Hyperparameter Recommendations

### Beta/Tau — THE Critical Parameter

| Source | Recommended β | Context |
|--------|--------------|---------|
| HuggingFace blog (best result) | **0.01** | Zephyr/OpenHermes on MT-Bench |
| EXPO paper | 0.1 - 0.5 | UltraFeedback, Mistral-based |
| Stanford CS224R | 0.01 - 0.05 | Various benchmarks |
| TRL default | 0.1 | General purpose |

**β controls target margin** as `1/(2β)`:
- β=0.01 → target margin = 50 (large gap allowed, less regularization)
- β=0.1 → target margin = 5 (moderate)
- β=0.5 → target margin = 1 (strong regularization)

**Smaller β = larger allowed margin = less regularization.** Counter-intuitive vs DPO.

### Learning Rate

- DPO converged in ~50 steps at LR=2e-5 (aggressive)
- IPO's squared loss provides stronger gradients than sigmoid
- **Recommendation:** Start at **1e-5** (half DPO's LR), consider 5e-6

### Our Config Plan

**Fair comparison config (matched to DPO):**
```yaml
dpo:
  beta: 0.1        # Same as DPO for fair comparison
  loss_type: "ipo"
training:
  learning_rate: 2.0e-5  # Same as DPO
  num_train_epochs: 1
```

**Tuned config (if fair comparison doesn't converge well):**
```yaml
dpo:
  beta: 0.01       # HF blog optimal
  loss_type: "ipo"
training:
  learning_rate: 1.0e-5  # Slightly reduced
```

**Sweep plan:** β ∈ {0.01, 0.05, 0.1} with LR ∈ {5e-6, 1e-5, 2e-5}

---

## 6. IPO vs DPO: Comparison Summary

| Aspect | DPO | IPO |
|--------|-----|-----|
| **Loss** | `-log σ(β · ρ)` | `(ρ - 1/(2β))²` |
| **Loss type** | Logistic (BCE) | Squared (MSE) |
| **Preference model** | Bradley-Terry | Identity (model-free) |
| **Gradient at extremes** | Vanishes | Grows quadratically |
| **Regularization** | Weak at saturation | Strong everywhere |
| **Overfitting risk** | Higher with deterministic prefs | Lower (bounded margin) |
| **Beta sensitivity** | Moderate | **High** (must sweep) |
| **TRL parameter** | `loss_type="sigmoid"` | `loss_type="ipo"` |
| **Code changes** | (baseline) | Config-only |

### When IPO > DPO:
- Deterministic/near-deterministic preferences (our case)
- Limited preference data (prevents overfitting)
- Noise robustness (squared loss less sensitive to label noise)

### When DPO > IPO:
- Diverse, well-calibrated preferences with varying margins
- Maximum behavioral change desired
- Simpler tuning (less beta-sensitive)

---

## 7. Relevance to Depth-of-Alignment Question

### Our Key Finding So Far

DPO: SFT→DPO linear probe transfer AUROC = **0.754** — sycophancy patterns persist internally despite behavioral suppression.

SimPO: SFT→SimPO transfer AUROC = **0.503** (chance level) — suggests deeper removal.

### Does IPO Go Deeper?

**Arguments it might:**
- IPO's bounded margin forces a more nuanced solution — can't just maximize chosen/rejected gap
- Constrained optimization might push changes deeper into the network
- Stronger regularization prevents the "extreme routing" pattern (Lee et al., ICML 2024 found DPO alignment bypasses toxic regions rather than removing capability)

**Arguments it might not:**
- IPO and DPO achieve similar benchmark performance in general settings
- Regularization may affect "how much" but not "how deep"
- Both use same LoRA architecture, limiting where changes can occur

### What to Test:
1. Linear probing: SFT→IPO transfer AUROC vs DPO's 0.754 and SimPO's 0.503
2. Layer-wise probe accuracy profiles
3. Convergence dynamics comparison (steps to behavioral recovery)
4. Relearning speed on IPO model (if time permits)

---

## 8. Key Gotchas for Implementation

1. **β sensitivity is THE gotcha** — IPO is much more sensitive than DPO. Optimal can range from 0.01 to 0.5 depending on dataset. Must sweep.
2. **Loss scale differs** — IPO squared values look very different from DPO log-sigmoid on wandb. Don't compare loss magnitudes directly.
3. **Convergence pattern may differ** — Stronger regularization could mean more steps needed, or plateau at less extreme solution.
4. **Length normalization** — TRL handles this automatically (per-token averaging). Be aware if debugging.
5. **label_smoothing silently ignored** — Not applicable to IPO.

---

## Sources

- [Azar et al., AISTATS 2024 — IPO Paper](https://arxiv.org/abs/2310.12036)
- [HuggingFace Blog: Preference Tuning with DPO Methods](https://huggingface.co/blog/pref-tuning)
- [TRL DPOTrainer Documentation](https://huggingface.co/docs/trl/en/dpo_trainer)
- [TRL DPO Trainer Source](https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py)
- [TRL PR #1265: IPO average_log_prob fix](https://github.com/huggingface/trl/pull/1265)
- [TRL Issue #1581: IPO loss vs dataset format](https://github.com/huggingface/trl/issues/1581)
- [Lee et al., ICML 2024 — Mechanistic Understanding of DPO Alignment](https://arxiv.org/html/2401.01967v1)
- [RainbowPO, ICLR 2025](http://www.columbia.edu/~wt2319/RainbowPO.pdf)
- [Argilla Blog: RLHF Alternatives — IPO](https://argilla.io/blog/mantisnlp-rlhf-part-6/)
