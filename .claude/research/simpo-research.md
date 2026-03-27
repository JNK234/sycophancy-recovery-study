# SimPO Research: Simple Preference Optimization

Research date: 2026-03-27

## Paper

**Title:** SimPO: Simple Preference Optimization with a Reference-Free Reward
**Authors:** Yu Meng, Mengzhou Xia, Danqi Chen (Princeton NLP)
**Venue:** NeurIPS 2024
**ArXiv:** 2405.14734
**Code:** github.com/princeton-nlp/SimPO

---

## 1. Core Mechanism

### The Problem with DPO's Reward

DPO defines an implicit reward as the log-probability ratio between policy and reference:

```
r_DPO(x, y) = β * log(π_θ(y|x) / π_ref(y|x)) + β * log Z(x)
```

Two issues:
1. This reward is a *relative* measure (policy vs reference), not aligned with how we *generate* text (which uses absolute log-probs).
2. It requires keeping a full reference model in memory during training (2x memory cost for non-PEFT setups).

### SimPO's Solution: Average Log Probability as Reward

SimPO replaces the DPO reward with the **length-normalized average log probability**:

```
r_SimPO(x, y) = (β / |y|) * log π_θ(y|x)
             = (β / |y|) * Σ_{i=1}^{|y|} log π_θ(y_i | x, y_{<i})
```

Key insight: This directly corresponds to the likelihood ranking used during inference (beam search, sampling). DPO's log-ratio reward has a *misalignment* between what the training objective optimizes and what decoding actually uses.

### Length Normalization

Division by |y| (response token count) is critical:
- Without it, the model exploits length: longer responses get higher total log-prob simply by having more tokens
- Ablation shows removing length normalization drops AlpacaEval 2 LC win rate from 21.5% to 11.9% (Mistral-Base)
- The correlation between reward and response length drops from 0.82 (unnormalized) to 0.34 (normalized)

### Target Reward Margin (γ)

SimPO adds a margin term γ > 0 to the Bradley-Terry preference model:

```
p(y_w ≻ y_l | x) = σ(r(x, y_w) - r(x, y_l) - γ)
```

This enforces a minimum gap between winning and losing rewards, preventing the model from being "lazy" (making the margin just barely positive). Ablation: removing γ drops AlpacaEval 2 from 21.5% to 16.8%.

---

## 2. Mathematical Comparison: SimPO vs DPO

### DPO Loss

```
L_DPO = -E[ log σ( β * (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)) ) ]
```

### SimPO Loss

```
L_SimPO = -E[ log σ( (β/|y_w|) * log π_θ(y_w|x) - (β/|y_l|) * log π_θ(y_l|x) - γ ) ]
```

### Key Differences

| Aspect | DPO | SimPO |
|--------|-----|-------|
| Reference model | Required (π_ref) | None |
| Reward definition | Log-ratio π_θ/π_ref | Average log-prob of π_θ |
| Length handling | No normalization | Divide by response length |
| Margin term | None | γ > 0 enforces minimum gap |
| β range | Typically 0.1-0.5 | Much larger: 2.0-10.0 |
| Implicit reward | Relative to reference | Absolute sequence likelihood |

The larger β in SimPO compensates for the absence of the reference model ratio -- the raw log-probs are much smaller numbers than the log-ratios, so β must scale them up.

---

## 3. Recommended Hyperparameters

### From the Paper and Official Repo

| Parameter | Recommended Range | Notes |
|-----------|------------------|-------|
| β (beta) | 2.0 - 10.0 | Much larger than DPO's 0.1-0.5. Model-specific. |
| γ (gamma) | 0.5 - 1.5 | Start with γ/β ≈ 0.5, grid search 0-1 |
| Learning rate | 3e-7 to 1e-6 | "Most critical hyperparameter." Lower for reasoning tasks. |
| Batch size | 128 total | Effective batch across GPUs |

### Model-Specific Examples from Authors

| Model | β | γ/β | LR |
|-------|---|-----|-----|
| Llama3-8B-Instruct | 2.5 | 0.55 | 1e-6 |
| Mistral-7B-Base | 2.0 | ~0.25 | 5e-7 |
| Gemma-2-9B-IT | Larger (~10) | 0.5 | ~5e-7 |

### Tuning Strategy (from authors)

1. Start with β=2.0, γ/β=0.5
2. Grid search LR in {3e-7, 5e-7, 8e-7, 1e-6}
3. If performance plateaus, try larger β (up to 10)
4. For reasoning-heavy tasks, use smaller LR (5e-7)
5. "Hyperparameter tuning is crucial for SimPO" -- the authors are explicit about this

---

## 4. TRL Implementation Details

### SimPO lives in CPOTrainer, NOT DPOTrainer

In TRL 0.29.1 (our version), SimPO is implemented as a loss variant inside `trl.experimental.cpo.CPOTrainer`:

```python
from trl.experimental.cpo import CPOTrainer, CPOConfig

config = CPOConfig(
    loss_type="simpo",      # Activates SimPO loss
    cpo_alpha=0.0,          # MUST be 0 for pure SimPO (disables CPO's BC regularizer)
    beta=2.0,               # SimPO beta (much larger than DPO)
    simpo_gamma=0.5,        # Target reward margin γ
    # ... standard training args ...
)

trainer = CPOTrainer(
    model=model,
    args=config,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=lora_config,  # Works with PEFT
)
```

### Key Configuration Notes

- `loss_type="simpo"` + `cpo_alpha=0.0` = pure SimPO
- `loss_type="simpo"` + `cpo_alpha > 0` = CPO-SimPO hybrid (adds SFT BC loss)
- Default `simpo_gamma=0.5` (reasonable starting point)
- Default `beta=0.1` -- THIS IS THE DPO DEFAULT, TOO LOW FOR SIMPO. Must override to 2.0+
- CPOTrainer is reference-free by design -- no ref_model parameter
- It is in `trl.experimental` -- APIs may change between TRL versions
- `alpha` parameter: for AlphaPO variant. Keep at 0.0 for standard SimPO.

### CPOTrainer vs DPOTrainer Differences

| Feature | DPOTrainer | CPOTrainer (SimPO mode) |
|---------|-----------|------------------------|
| Reference model | Uses base weights as ref (PEFT) or separate model | No reference model at all |
| Memory | 2x model weights (or PEFT workaround) | 1x model weights only |
| loss_type options | sigmoid, hinge, ipo, etc. (14+ variants) | sigmoid, hinge, ipo, simpo, alphapo |
| Length normalization | Not built in | Built into SimPO loss |
| Reward margin | Not available | simpo_gamma parameter |

### Warning: experimental API

The import path is `trl.experimental.cpo` which prints a warning:
```
TRLExperimentalWarning: You are importing from 'trl.experimental'. APIs here are unstable and may change or be removed without notice.
```
Silence with `TRL_EXPERIMENTAL_SILENCE=1` env var.

---

## 5. Relevance to Sycophancy Recovery

### Length Normalization and Verbose Sycophancy

Sycophantic responses are often longer/more verbose than honest ones (the model hedges, over-explains, adds unnecessary agreement). This creates a systematic bias in DPO:

- In DPO, the raw log-prob sum for longer sycophantic responses can be lower (more negative) than for shorter honest responses purely due to length, creating noisy reward signals
- SimPO's per-token normalization should give a *cleaner* signal: it compares the *quality per token* of chosen vs rejected, not confounded by length differences
- This is directly relevant since our DPO pairs have length asymmetry (sycophantic responses tend longer)

**Prediction:** SimPO may learn a cleaner separation between sycophantic and honest behavior because the length normalization removes a confound.

### Policy Drift Without Reference Model

DPO constrains the policy via the KL penalty implicit in the log-ratio with the reference. SimPO has NO such constraint. Implications:

**Potential advantage for sycophancy removal:**
- Without the reference anchor, SimPO can make larger changes to the policy
- If sycophancy is deeply embedded, the freedom to drift further might enable deeper removal
- Our linear probing showed DPO suppresses but doesn't remove sycophancy -- SimPO's stronger optimization pressure might go deeper

**Potential risk:**
- More policy drift = more risk of reward hacking or capability degradation
- The paper acknowledges "training could potentially lead to reward hacking since SimPO does not regularize against the reference model"
- However, the authors found no actual collapse with proper hyperparameter tuning
- The β parameter acts as an implicit regularizer (larger β = stronger preference signal = more stable)

**For our probing analysis:** If SimPO changes the internal representations more (due to no reference anchor), the SFT→SimPO transfer AUROC might be lower than SFT→DPO's 0.754. This would suggest deeper removal. Conversely, if it's similar, it suggests the surface behavior changes more but internal representations stay similar.

### Practical Benefits

1. **Memory:** No reference model means less GPU memory. With PEFT/LoRA this matters less (DPO already uses base weights as reference), but still slightly more efficient.
2. **Compute:** One fewer forward pass per batch (no reference model forward pass). ~10-15% faster training.
3. **Simplicity:** Fewer moving parts = easier to debug and iterate.

### No Existing Literature on SimPO for Sycophancy

No papers found applying SimPO specifically to sycophancy removal or behavioral alignment safety. This would be a novel contribution -- comparing DPO vs SimPO for behavioral intervention depth is not well-studied.

---

## 6. Comparison: When SimPO Outperforms DPO

### SimPO Wins

- **General instruction following:** 6.4 points on AlpacaEval 2, 7.5 points on Arena-Hard
- **When data has length bias:** Length normalization prevents exploitation
- **When compute/memory is limited:** No reference model
- **When the reference model is a poor anchor:** If the base model is already somewhat sycophantic, anchoring to it (as DPO does) might limit recovery

### DPO Wins

- **Reasoning/math tasks:** SimPO's authors acknowledge preference optimization generally hurts reasoning, and SimPO can be worse here
- **When stability matters more than peak performance:** DPO's reference anchor provides natural regularization
- **When hyperparameter tuning budget is limited:** SimPO is more sensitive to hyperparameters

### Training Stability

- DPO: Naturally stable due to KL constraint from reference model
- SimPO: Can be unstable without proper β and γ tuning. The γ margin helps -- it acts as a "floor" preventing the rewards from collapsing
- Both: Learning rate is critical. SimPO uses much lower LR (3e-7 to 1e-6) vs DPO's typical 1e-5 to 5e-5

---

## 7. Implementation Plan for Our Project

### What Needs to Change

1. **New trainer class:** `src/training/simpo_trainer.py` wrapping `CPOTrainer` instead of `DPOTrainer`
2. **Config schema update:** Add SimPO-specific fields (simpo_gamma, cpo_alpha) to config, or extend DPOSection
3. **New YAML config:** `configs/training/simpo_recovery.yaml`
4. **Valid methods update:** Add "simpo" to `config_schema.py` valid_methods set
5. **Same dataset:** Can reuse exact same DPO pairs (chosen/rejected format is identical)

### Key Differences from DPO Setup

| Setting | Our DPO Config | SimPO Recommendation |
|---------|---------------|---------------------|
| beta | 0.1 | 2.0-2.5 (start with 2.0) |
| learning_rate | 2e-5 | 5e-7 to 1e-6 (much lower!) |
| loss_type | sigmoid | simpo |
| simpo_gamma | N/A | 0.5-1.5 (start with 1.0) |
| cpo_alpha | N/A | 0.0 (pure SimPO) |
| ref_model | None (PEFT base) | N/A (CPOTrainer has no ref) |

### Critical Gotcha: Learning Rate

Our DPO used LR=2e-5 which was already aggressive. SimPO authors recommend 20-40x lower (5e-7 to 1e-6). This is the most likely source of training failure if not adjusted.

### Critical Gotcha: Beta Scale

DPO beta=0.1 is standard. SimPO beta=0.1 would be way too low -- the loss would be nearly flat. Must use β >= 2.0.
