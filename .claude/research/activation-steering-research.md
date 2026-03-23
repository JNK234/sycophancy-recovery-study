# Activation Steering & Representation Engineering Research

Research compiled 2026-03-22 for the sycophancy recovery study.
Covers techniques for modifying LLM behavior via internal activation manipulation at inference time.

---

## 1. Contrastive Activation Addition (CAA)

**Paper:** Rimsky et al., "Steering Llama 2 via Contrastive Activation Addition" (ACL 2024)
**Code:** https://github.com/nrimsky/CAA

### How It Works Mechanically

1. **Collect contrastive pairs** — Each pair consists of a prompt with two possible continuations: one exhibiting the target behavior (e.g., sycophantic answer) and one not. Formatted as multiple-choice (A/B) so the model picks one token.
2. **Extract activations** — Run forward passes on both versions of each pair. At a chosen layer, extract residual stream activations at the answer token position.
3. **Compute steering vector** — Take the mean activation difference across all pairs: `v = mean(act_positive - act_negative)`. This yields a single direction in activation space.
4. **Apply at inference** — During generation, add `alpha * v` to the residual stream at all token positions after the user prompt. Positive alpha promotes the behavior, negative alpha suppresses it.

### Sycophancy-Specific Details

- Contrastive datasets sourced from Anthropic's sycophancy datasets (NLP survey, political typology from Perez et al. 2022), ~1000 questions total.
- **Number of pairs needed:** Rimsky et al. used several hundred pairs for vector generation (holding out 50 for eval). The technique is data-efficient — aggregating over many pairs reduces noise vs few-shot prompting.
- **Layer selection for sycophancy:** Layer 15 (of 32 in Llama-2-7B, ~47% depth) showed largest effect on sycophancy. Effect decreases toward later layers.
- Subtracting the sycophancy vector increases TruthfulQA performance, demonstrating generalization beyond the training distribution.
- CAA generalizes to open-ended generation even when trained on MC pairs — unlike finetuning, which fails to generalize from MC to open-ended for sycophancy.

### Creating Contrastive Pairs for Sycophancy

For our project (Qwen3-8B SFT model), contrastive pairs should:
- Present a user opinion followed by a question where the model could either agree (sycophantic) or give an honest/accurate answer
- Use MC format with (A) sycophantic and (B) honest options (or vice versa, balanced)
- Cover diverse topics: factual questions where user is wrong, subjective opinions, political/philosophical stances
- **Recommended: 200-500 pairs** based on the literature (Rimsky et al. used ~950 minus 50 held out)
- We already have sycophancy eval datasets in `evals/sycophancy-eval/datasets/` that could be adapted

### Key Strengths

- Zero inference-time compute cost (just a vector addition)
- Stacks additively with finetuning and prompting
- More precise than few-shot prompting
- Generates in < 5 minutes on a single GPU
- Reversible — just remove the vector

### Key Limitations

- Layer and alpha selection require empirical tuning
- Less effective on frontier/larger models (diminishing returns at scale)
- Prompt-dependent: same vector has different effects on different inputs (Tan et al., NeurIPS 2024)
- Effects wash out after 5-6 turns in multi-turn conversations

### What Questions It Answers for Our Study

- Does the SFT model have a linear "sycophancy direction" in its residual stream?
- Can we suppress sycophantic behavior at inference time without retraining?
- Does the steering vector transfer across eval formats (MC -> open-ended)?
- How does CAA compare to DPO/RLHF for sycophancy reduction?

---

## 2. Representation Engineering (RepE)

**Paper:** Zou et al., "Representation Engineering: A Top-Down Approach to AI Transparency" (arXiv:2310.01405, 2023)
**Code:** https://github.com/andyzoujm/representation-engineering
**Community library:** https://github.com/vgel/repeng (`pip install repeng`)

### How It Works Mechanically

RepE takes a top-down approach inspired by cognitive neuroscience. Instead of analyzing individual neurons (mechanistic interpretability), it treats high-level concepts as directions or subspaces in the model's activation space.

**Two core operations:**

#### Representation Reading (Monitoring)
- **Linear Artificial Tomography (LAT):** Analogous to brain MRI. Present contrastive stimuli to the model, observe which layers show differentiated activation patterns for a concept.
- Train a linear probe on activations to detect the concept. The probe's weight vector is the "reading vector" for that concept.
- Can detect honesty, power-seeking, harmfulness, etc. as linear directions at specific layers.

#### Representation Control (Steering)
Three types of controllers:
1. **Reading vector** — The stimulus-independent direction found by LAT. Add/subtract to promote/suppress.
2. **Per-input contrast vector** — Compute the difference for a specific input pair (more targeted but less general).
3. **LoRRA (Low-Rank Representation Adaptation)** — LoRA-style adapter trained to match target representations. More expressive but requires training.

Three operators for applying controllers:
1. **Linear combination** — `act' = act + alpha * v` (stimulation/suppression)
2. **Piece-wise** — Conditional boost: only add when the dot product with the reading vector has a specific sign
3. **Projection** — Erase the component: `act' = act - (act . v_hat) * v_hat`

### Reading Vectors vs Control Vectors

| Aspect | Reading Vectors | Control Vectors |
|--------|----------------|-----------------|
| Purpose | Monitor/detect a concept | Modify/steer behavior |
| Operation | Dot product with activations -> scalar score | Add to activations -> changed behavior |
| Training | Linear probe on labeled activations | PCA on contrastive activation diffs |
| Output | Probability/classification | Modified generation |

### Key Results

- Truthfulness steering: +30pp on TruthfulQA at selected layers
- Harmlessness: raised harmless rates from 65% to >90%
- Reading vectors achieve >83% accuracy detecting truthful vs false statements

### Relevance to Our Study

RepE's reading vectors are directly relevant to our **linear probe phase** (Phase 5). We want to know if sycophancy persists as a detectable direction even after recovery interventions. RepE's framework gives us:
- A principled way to extract the "sycophancy direction" at each layer
- Tools to test whether DPO/RLHF actually removes this direction or just makes the model ignore it
- The theoretical grounding that concepts are linear directions in activation space

### Implementation: repeng Library

```python
# repeng provides ControlVector, ControlModel, DatasetEntry
from repeng import ControlModel, ControlVector, DatasetEntry

# Wrap HF model
model = ControlModel(hf_model, layer_ids=list(range(num_layers)))

# Create contrastive dataset entries
dataset = [DatasetEntry(positive="...", negative="...") for ...]

# Train control vector (< 60 seconds)
vector = ControlVector.train(model, tokenizer, dataset, method="pca_center")

# Apply during generation
model.set_control(vector, coeff=1.5)
output = model.generate(...)

# Export for llama.cpp
vector.export_gguf("sycophancy_vector.gguf")
```

**Limitation:** repeng does not work with MoE models. Qwen3-8B is dense, so this is fine for us.

---

## 3. SAE-Based Steering

**Key Papers:**
- SAE-RSV: "Enhancing LLM Steering through SAE-Based Vector Refinement" (2025, arXiv:2509.23799)
- GSAE: "Graph-Regularized SAEs for Robust LLM Safety Steering" (2025, arXiv:2512.06655)
- CorrSteer: "Generation-Time LLM Steering via Correlated SAE Features" (2025, arXiv:2508.12535)
- CRL: "Control RL: Token-Level Steering via SAE Features" (2026, arXiv:2602.10437)

### How It Works Mechanically

Sparse Autoencoders (SAEs) decompose dense model activations into sparse, interpretable features. Each feature corresponds to a monosemantic concept (ideally). SAE-based steering leverages this decomposition:

1. **Train or load a pretrained SAE** on a model's residual stream activations at a target layer. The SAE maps d_model-dimensional activations into a much higher-dimensional sparse space (e.g., 4096 -> 65536 features).
2. **Identify sycophancy-related features** — Run contrastive pairs through the model, encode activations with the SAE, find features that activate differentially between sycophantic and honest responses.
3. **Steer by scaling features** — During inference, encode the activation, scale up anti-sycophancy features or scale down/ablate sycophancy features, decode back to model space.

### SAE-RSV (Refinement of Steering Vector)

The most relevant technique for combining SAE interpretability with CAA-style steering:
1. Compute a raw steering vector (as in CAA)
2. Encode it through the SAE to get feature activations
3. **Denoise:** Remove features that are semantically irrelevant to the target behavior
4. **Augment:** Add semantically similar features that were missed due to limited data
5. Decode back to get a refined steering vector

This outperforms both raw CAA and supervised finetuning, especially when contrastive data is limited.

### GSAE (Graph-Regularized SAE)

Standard SAEs have a limitation for safety/alignment concepts: the monosemantic assumption fragments distributed concepts (like "sycophancy") into many small, redundant features. GSAE adds graph Laplacian regularization to enforce coherence among co-activating features, yielding better representations for abstract behavioral patterns.

### CorrSteer

Eliminates the need for contrastive datasets by identifying steering directions from generation-time activations alone, using correlation analysis across SAE features.

### What Questions It Answers

- Which specific interpretable features encode sycophancy?
- Can we surgically remove sycophancy features without affecting other capabilities?
- Does recovery training (DPO/RLHF) change which SAE features activate, or just change how the model uses them?

### Implementation Complexity

**High.** Requires:
- A trained SAE for the target model (Qwen3-8B) at the right layer — no pretrained SAEs likely available for Qwen3-8B
- Training an SAE requires substantial compute (collecting millions of activation vectors, training the autoencoder)
- SAELens can train SAEs but the pipeline is non-trivial

**Recommendation for our study:** SAE-based steering is the most powerful but also most complex approach. Consider it for Phase 5 (probing) rather than Phase 3 (recovery), unless pretrained SAEs become available for Qwen3.

---

## 4. Inference-Time Intervention (ITI)

**Paper:** Li et al., "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model" (NeurIPS 2023, arXiv:2306.03341)
**Code:** https://github.com/likenneth/honest_llama

### How It Works Mechanically

ITI targets **attention heads** specifically (not the full residual stream):

1. **Probing phase:**
   - Collect labeled data (truthful vs false completions)
   - For each attention head at each layer, extract its output activations
   - Train a linear probe to separate truthful from false activations
   - Rank heads by probe accuracy — top-K heads are "truth-encoding"
   - For each selected head, compute the "truthful direction" as the mass mean shift between truthful and false activation distributions

2. **Intervention phase:**
   - During inference, at each token generation step, shift activations of the selected attention heads along their truthful directions
   - The shift magnitude is a tunable hyperparameter (alpha)
   - Model weights remain unchanged

### Key Distinction: Mass Mean Shift vs Probe Direction

Two options for the intervention direction:
- **Probe weight direction** — The separating hyperplane from the linear classifier
- **Mass mean shift** — Vector from the mean of false activations to the mean of truthful activations

Counterintuitively, mass mean shift outperforms probe direction. This means the directions the model uses to *generate* true/false statements differ from the directions a classifier finds most discriminative.

### Key Results

- Alpaca: truthfulness improved from 32.5% to 65.1%
- Only needs a few hundred labeled examples (data-efficient)
- Certain heads show >83% probe accuracy for truth detection while zero-shot generation is at 30%

### Tradeoffs

- There is a truthfulness-helpfulness tradeoff tunable via intervention strength
- ITI may select late-layer heads based on superficial correlations; other methods find mid-layer heads more effective
- Extensions: NL-ITI (nonlinear probing, multi-token), HSI (head-specific intervention for coordination detection)

### Relevance to Our Study

ITI provides a complementary approach to CAA:
- **CAA** operates on the residual stream (all information combined)
- **ITI** targets specific attention heads (more surgical)

For sycophancy, ITI could help identify *which attention heads* encode sycophantic behavior, providing mechanistic insight beyond what CAA offers. The probing phase of ITI is also directly useful for our Phase 5 linear probe work.

### Implementation Complexity

**Medium.** The honest_llama codebase provides a template. Main work:
- Adapt for Qwen3-8B architecture (different attention head structure: 32 Q heads, 8 KV heads with GQA)
- Create labeled sycophancy dataset (we already have this)
- Run probing across all heads at all layers
- Apply interventions at inference time

---

## 5. Feature-Guided Activation Addition (FGAA)

**Paper:** Soo, Teng, Balaganesh, "Interpretable Steering of LLMs with Feature Guided Activation Additions" (ICLR 2025, arXiv:2501.09929)

### How It Works Mechanically

FGAA bridges CAA and SAE-based methods. The key insight: instead of constructing a steering vector in model space and hoping it captures the right concept, work backwards from the SAE feature space:

1. **Compute mean activation difference** (as in CAA) between contrastive pairs
2. **Encode through SAE** to get the difference in feature space
3. **Automated feature selection:**
   - Filter features by relevance to the target behavior (using SAE feature labels/descriptions)
   - Handle feature splitting in larger SAEs (where one concept splits across multiple features) by programmatically determining relative magnitudes
4. **Optimize feature weights** to construct the steering vector with explicit control over which features are steered and by how much
5. **Decode back** to get the final steering vector in model space

### Key Advantages Over CAA

- **Interpretable:** You know exactly which features are being steered
- **Precise:** Removes off-target features that happen to correlate with contrastive pairs
- **Handles feature splitting:** Larger SAEs have more features that split a concept; FGAA handles this programmatically
- Evaluated on Gemma-2-2B and Gemma-2-9B, outperformed CAA, SAE decoder steering, and SAE-TS in 8/9 tasks

### Behavioral-Coherence Score (BCS)

FGAA introduced BCS as an evaluation metric combining:
- **Behavioral alignment** — Does the output match the desired behavior?
- **Coherence** — Is the output fluent and well-formed?

This captures the fundamental tradeoff better than either metric alone.

### Key Findings

- FGAA shows stable performance across steering scales (no initial performance bump that degrades)
- Effectiveness varies non-linearly with model size
- Performance degrades above certain steering magnitudes

### Relevance to Our Study

FGAA would be the ideal activation steering method for our study *if* we had SAEs for Qwen3-8B. It provides the best balance of effectiveness, interpretability, and precision.

**Practical barrier:** Requires trained SAEs for the target model. No pretrained SAEs exist for Qwen3-8B. Training SAEs is a substantial effort (see Section 3).

### Implementation Complexity

**High.** Requires SAEs (same barrier as Section 3) plus the optimization step for feature selection. The FGAA paper code may provide a template but is built around Gemma models.

---

## 6. Practical Considerations

### Layer Selection Guidelines

Based on multiple papers and the 2026 practitioner's field guide:

| Behavior Type | Recommended Layers | Notes |
|---------------|-------------------|-------|
| Sycophancy / compliance | Middle layers (~40-60% depth) | Rimsky et al. found layer 15/32 (~47%) best for sycophancy |
| Refusal / safety | Middle-to-late (~60-85% depth) | Well-replicated finding |
| Sentiment / tone | Late layers (~75-90% depth) | Low-dimensional, steers cleanly |
| Multi-behavior | Different vectors at different layers | Avoid naive vector addition at same layer |

**For Qwen3-8B (36 layers):**
- Start sweep at layers 14-22 (~40-60% depth) for sycophancy
- Early layers (< layer 9) will likely degrade output quality
- Late layers (> layer 27) may have weaker sycophancy signal

**Important caveat:** Detection layer != optimal steering layer. A layer where a probe detects sycophancy with high accuracy is not necessarily the best layer to inject a steering vector.

### Newer Approaches to Layer Selection

- **Depth-Wise Gaussian Scheduling** (Goral et al., 2024): Instead of a single layer, distribute steering across layers with a Gaussian profile centered on the target region. Avoids the brittleness of single-layer selection.
- **Hybrid Layer Selection** (2025): Per-input adaptive selection using diagnostic metrics and delta-logit norms.
- **Multi-property steering at distinct layers** (Weij et al., 2024): Inject different steering vectors at different layers for controlling multiple behaviors simultaneously.

### Steering Magnitude (Alpha) Tuning

- Start with alpha in [0.5, 3.0] range
- The relationship between alpha and effect is genuinely non-monotonic (Taimeskhanov et al.) — increasing alpha past a threshold can reverse or degrade the effect
- Evaluate at multiple alpha values on a held-out set
- Different inputs respond differently to the same alpha (prompt-dependence)

### How Many Contrastive Pairs Are Needed

| Method | Typical Range | Notes |
|--------|---------------|-------|
| CAA | 200-1000 pairs | Rimsky et al. used ~950; diminishing returns above 500 |
| ITI | 100-300 pairs | Data-efficient due to per-head probing |
| RepE/repeng | 50-200 pairs | PCA-based extraction is sample-efficient |
| FGAA | 200-500 pairs | Feature filtering compensates for noisier data |

For our study, we have ~3236 rows in our sycophancy datasets. Adapting a subset (500-1000) for contrastive pair format should be straightforward.

### Evaluating Steering Quality

**Must-measure metrics:**
1. **Target behavior change** — Sycophancy rate on eval set (our existing eval pipeline)
2. **Coherence/fluency** — Perplexity on held-out text; LLM-as-judge coherence score
3. **Capability retention** — MMLU or similar benchmark to check for degradation
4. **Behavioral-Coherence Score (BCS)** — Combined metric from FGAA paper
5. **Cross-behavior interference** — Does anti-sycophancy steering affect helpfulness, refusal behavior, etc.?

**From "A Sober Look at Steering Vectors" (Alignment Forum):**
- Steering vectors consistently degrade model performance, often significantly
- Many papers fail to report degradation or measure it inadequately
- Must evaluate in realistic scenarios (open-ended QA, multi-turn dialogue), not just MC benchmarks

### The Coherence-Behavior Tradeoff

This is well-documented and fundamental:
- Stronger steering -> more behavioral change but worse coherence
- The tradeoff is non-linear and model/prompt dependent
- Late-layer and dynamic interventions provide better tradeoffs than early-layer static steering
- FGAA and adaptive methods (SADI, CAST, PID-based steering) achieve better Pareto frontiers

### Multi-Turn Durability

Steering effects wash out over 5-6 turns in multi-turn conversations. For persistent modification, re-apply the steering vector at each turn. This is relevant for our eval — our are_you_sure evaluator uses multi-turn probing.

---

## 7. Implementation Libraries Comparison

| Library | Best For | Model Support | Steering Method | Complexity |
|---------|----------|---------------|-----------------|------------|
| **repeng** | Quick CAA/RepE experiments | HF models (not MoE) | Control vectors via PCA | Low |
| **steering-vectors** | CAA replication (Tan et al.) | HF models | Contrastive steering vectors | Low |
| **Dialz** | End-to-end steering pipeline | HF models | Vectors + datasets + scoring + viz | Low-Medium |
| **IBM activation-steering** | General-purpose steering | HF models | Contrastive extraction + application | Medium |
| **TransformerLens** | Mechanistic interpretability | 50+ models (custom wrappers) | Hook-based activation access | Medium |
| **nnsight** | Direct HF model intervention | Any HF model | Tracing-based API | Medium |
| **nnterp** | Cross-architecture analysis | 50+ models via nnsight | Unified interface + built-in methods | Medium |
| **SAELens** | SAE training and analysis | Any PyTorch model | SAE feature-level steering | High |

### Recommended Stack for Our Study

**Phase 3 (Recovery via activation steering):**
- **repeng** or **steering-vectors** for quick CAA experiments
- Simple, well-documented, works with HF models
- Can export vectors for later analysis

**Phase 5 (Probing depth of removal):**
- **TransformerLens** or **nnsight** for extracting activations at every layer
- Train linear probes on extracted activations
- Compare reading vectors before/after recovery interventions

**If we want SAE analysis later:**
- **SAELens** for training SAEs on Qwen3-8B activations
- Significant compute investment but provides the deepest mechanistic insight

---

## 8. Summary: Which Technique for Our Study?

| Technique | Implementation Effort | Sycophancy Relevance | Interpretability | Recommendation |
|-----------|----------------------|---------------------|------------------|----------------|
| **CAA** | Low (days) | High — directly tested on sycophancy | Medium | Primary method for Phase 3 |
| **RepE reading vectors** | Low-Medium | High — probe whether sycophancy direction persists | High | Primary method for Phase 5 |
| **ITI** | Medium (1-2 weeks) | Medium — attention head level insight | High | Optional, adds mechanistic depth |
| **SAE-based** | High (weeks) | High — feature-level sycophancy analysis | Very High | Stretch goal if time permits |
| **FGAA** | High (requires SAEs) | Very High — best precision | Very High | Only if SAEs are available |

### Recommended Plan

1. **Start with CAA** using repeng or steering-vectors library
   - Create 500+ contrastive sycophancy pairs from our existing datasets
   - Sweep layers 14-22, alpha 0.5-3.0
   - Evaluate with our existing eval pipeline
   - Compare against DPO recovery

2. **Use RepE reading vectors for probing** (Phase 5)
   - Extract sycophancy direction at each layer for: base model, SFT model, post-DPO model, post-CAA model
   - Test if recovery methods truly remove the direction or just suppress it
   - This is the core scientific question of the study

3. **Consider ITI as a secondary analysis** if results from CAA + probing are inconclusive

---

## 9. Key References

- Rimsky et al. (2024). "Steering Llama 2 via Contrastive Activation Addition." ACL 2024. https://arxiv.org/abs/2312.06681
- Zou et al. (2023). "Representation Engineering: A Top-Down Approach to AI Transparency." https://arxiv.org/abs/2310.01405
- Li et al. (2023). "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model." NeurIPS 2023. https://arxiv.org/abs/2306.03341
- Soo et al. (2025). "Interpretable Steering of LLMs with Feature Guided Activation Additions." ICLR 2025. https://arxiv.org/abs/2501.09929
- Wang et al. (2025). "Enhancing LLM Steering through SAE-Based Vector Refinement." https://arxiv.org/abs/2509.23799
- Goral et al. (2024). "Depth-Wise Activation Steering for Honest Language Models." https://arxiv.org/abs/2512.07667
- GSAE (2025). "Graph-Regularized Sparse Autoencoders for Robust LLM Safety Steering." https://arxiv.org/abs/2512.06655
- CorrSteer (2025). "Generation-Time LLM Steering via Correlated SAE Features." https://arxiv.org/abs/2508.12535
- Tan et al. (2024). Prompt-dependence of steering vectors. NeurIPS 2024.
- Taimeskhanov et al. Non-monotonic steering magnitude analysis across 11 LLMs.
- "A Sober Look at Steering Vectors for LLMs." https://www.alignmentforum.org/posts/QQP4nq7TXg89CJGBh/
- SteeringControl holistic evaluation. https://arxiv.org/abs/2509.13450
- repeng library. https://github.com/vgel/repeng
- steering-vectors library. https://github.com/steering-vectors/steering-vectors
- IBM activation-steering. https://github.com/IBM/activation-steering
- SAELens. https://github.com/decoderesearch/SAELens
- Awesome Representation Engineering (curated list). https://github.com/chrisliu298/awesome-representation-engineering
