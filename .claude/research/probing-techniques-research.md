# Probing & Representation Analysis Techniques for Sycophancy Detection

Research compiled 2026-03-22. Covers linear probing, CCS, CKA, RSA, non-linear probes,
layer-wise analysis, and practical implementation for our Qwen3-8B sycophancy study.

---

## 1. Linear Probing (Supervised)

### How It Works Mechanically

Train a logistic regression (or single linear layer + sigmoid) on frozen hidden-state
activations to classify a binary property (e.g., "is this response sycophantic?").

**Steps:**
1. Construct a dataset of (prompt, response) pairs labeled sycophantic / non-sycophantic.
2. Run each pair through the frozen model with `output_hidden_states=True`.
3. Extract the hidden-state vector at a chosen layer. Two token-selection strategies:
   - **Last-token activation** (most common for autoregressive models -- the final position
     before generation). Papadatos & Freedman (2024) found that aggregation across token
     positions *decreased* performance; last-token only worked best.
   - **Mean-pooled activations** over all response tokens (used for open-ended responses
     where no single token is canonical).
4. Train logistic regression (sklearn `LogisticRegression` or a single `nn.Linear` + BCE loss).
5. Evaluate with AUROC, accuracy, F1 on a held-out test set.

**What the probe weights mean:** The learned weight vector can be interpreted as a "concept
direction" in activation space. If the probe achieves high accuracy, the concept is linearly
separable -- supporting the **Linear Representation Hypothesis** (Park et al., 2024).

### Which Layers to Probe

- Probe every layer independently (or a representative subset: every 2nd, 4th layer).
- For Qwen-family models, **middle-to-upper layers (layers 12-25 out of ~32)** encode the
  most alignment-relevant signal. Papadatos & Freedman (2024) found layer 16 optimal for
  sycophancy in a 32-layer reward model (94% test accuracy).
- For Qwen3-8B (36 layers, 4096 hidden dim), expect peak signal in layers ~14-28.
- Early layers encode shallow syntactic features; late layers are dominated by output
  formatting and attention sinks.

### Train/Test Setup

- **Minimum dataset:** ~200-400 labeled pairs (Papadatos used 398 across 4 categories).
- **Train/test split:** 80/20 or use cross-validation. Critically, ensure no prompt overlap
  between train and test sets to avoid leakage.
- **Balanced classes:** Equal sycophantic and non-sycophantic examples.
- **Multiple probe types per dataset category** (e.g., MCQ sycophancy, feedback sycophancy,
  opinion agreement) to test generalization.

### What AUROC/Accuracy Means

- **AUROC > 0.9:** The concept is strongly linearly represented at that layer. The model
  "knows" it is being sycophantic in a way that is geometrically separable.
- **AUROC 0.7-0.9:** Partial signal, likely entangled with other features.
- **AUROC ~ 0.5:** No linear signal at that layer for this concept.
- **Accuracy** depends on class balance; always report AUROC alongside it.

### Key Results from Literature

- Papadatos & Freedman (2024): Layer 16 probe achieved 94% accuracy distinguishing
  sycophantic from non-sycophantic responses in reward model activations.
- Anthropic sleeper agent detection: Simple probes achieved **AUROC > 0.99** for detecting
  alignment-faking behavior from hidden states.
- Qwen3-8B probing: Middle-late layer band achieves 96.7% accuracy for semantically
  distinct categories (though sycophancy may be subtler).

### Limitations and Leakage Concerns (2024-2025)

1. **Spurious correlations:** Probes can learn dataset artifacts (e.g., response length,
   hedging language) rather than genuine sycophancy encoding. Mitigation: test on
   out-of-distribution prompts from different categories.
2. **Correlation != causation:** A successful probe shows information is *decodable* from
   activations, not that the model *uses* it to produce sycophantic behavior. To establish
   causation, need activation patching or steering experiments.
3. **Train-test leakage:** If prompts share structure/templates between train and test, the
   probe may learn prompt-specific features. Use categorically different test prompts.
4. **Feature combination:** The probe may learn a linear combination of simpler features
   (e.g., politeness + agreement + hedging) rather than a unified "sycophancy" concept.
5. **OOD generalization:** Probes often fail on negated statements, different question
   formats, or adversarial constructions (Alignment Forum, "Still No Lie Detector for LLMs").
6. **Lie != Deception:** Truth probes trained on true/false datasets detect *lying* well but
   fail at detecting *deception without lying* -- models can mislead with technically true
   statements (Levinstein & Herrmann, 2026).

### Relevance to Our Study

**Critical for Phase 5 (probing depth of removal).** After DPO/RLHF/CAI recovery:
- Train probe on SFT model (strong sycophancy signal).
- Apply same probe to recovered model. If AUROC stays high despite behavioral improvement,
  the sycophancy was *suppressed* not *removed* -- it is still linearly readable.
- If AUROC drops to ~0.5, the representation genuinely changed.

### Key Papers
- Papadatos & Freedman, "Linear Probe Penalties Reduce LLM Sycophancy" (NeurIPS SoLaR 2024)
- Park et al., "The Linear Representation Hypothesis" (ICLR 2024)
- Burns et al., "Discovering Latent Knowledge Without Supervision" (ICLR 2023)
- Alignment Forum, "Still No Lie Detector for LLMs" (2024)

---

## 2. CCS (Contrast-Consistent Search) -- Unsupervised Probing

### How It Works Mechanically

CCS finds a direction in activation space that separates true/false statements *without any
labels*. It exploits logical consistency: for any statement, exactly one of {statement, negation}
should be true.

**Steps:**
1. For each statement S, create a contrast pair: (S, not-S) -- e.g., "The capital of France is
   Paris" vs "The capital of France is not Paris."
2. Extract hidden states for both from the same layer.
3. Learn a linear function p(x) mapping activations to [0,1], optimized with two losses:
   - **Consistency loss:** p(S) + p(not-S) ~ 1 (exactly one should be true)
   - **Confidence loss:** push probabilities away from 0.5 (avoid trivial solution)
4. No labels needed -- the probe discovers the "truth direction" purely from consistency.

### How It Differs from Supervised Probing

| Aspect | Supervised Probe | CCS |
|--------|-----------------|-----|
| Labels needed | Yes (ground truth) | No |
| What it finds | Direction separating labeled classes | Direction satisfying logical consistency |
| Risk | Learns spurious features correlated with labels | May find most prominent feature, not necessarily truth |
| Generalization | Limited by training distribution | Limited by consistency assumption |
| Use case | When you have reliable labels | When labels are unavailable or contested |

### Limitations

- **Feature identification problem:** CCS finds the most *prominent* contrastive feature,
  which may not be truthfulness. Arbitrary binary features can satisfy the CCS loss with zero
  error (proven formally). If the dataset does not isolate a single contrastive feature,
  CCS fails to find truth reliably.
- **Sensitivity to initialization:** Vanilla CCS results depend on random initialization.
  The eigenproblem reformulation (Schouten et al., 2025) provides closed-form solutions
  that avoid this.
- **Modest improvements:** CCS outperforms zero-shot by ~4% on average -- useful but not
  dramatic.

### Relevance to Our Study

CCS could find a "sycophancy direction" without labeled data, by constructing contrast pairs:
- "I agree with your perspective that X" vs "I disagree with your perspective that X"
- If CCS finds a consistent direction, it suggests sycophancy is a coherent linear feature.
- However, CCS may find agreement/disagreement rather than sycophancy specifically.
- **Recommendation:** Use supervised probing as primary method (we have labels); CCS as
  supplementary validation that the direction is consistent.

### Key Papers
- Burns et al., "Discovering Latent Knowledge in Language Models Without Supervision" (ICLR 2023)
- Schouten et al., "LLM Probing with Contrastive Eigenproblems" (2025) -- reformulates CCS
  as eigenproblem, closed-form solutions
- Alignment Forum discussions on CCS failure modes

---

## 3. CKA (Centered Kernel Alignment) -- Comparing Representations Between Models

### How It Works Mechanically

CKA measures similarity between two sets of representations (e.g., layer L of model A vs
layer L of model B) by comparing their *relational structure* -- how stimuli relate to each
other within each representation space.

**Steps:**
1. Pass the same set of N inputs through both models.
2. Extract activation matrices X (N x d1) and Y (N x d2) from chosen layers.
3. Compute kernel matrices: K = XX^T, L = YY^T (or use RBF kernels for non-linear CKA).
4. Center both kernel matrices (subtract row/column means).
5. CKA = HSIC(K,L) / sqrt(HSIC(K,K) * HSIC(L,L)), where HSIC is the Hilbert-Schmidt
   Independence Criterion.
6. Result is in [0, 1]: 1 = identical relational structure, 0 = unrelated.

**Linear CKA** (using linear kernels) is most common and sufficient for typical neural
network comparisons. **RBF-CKA** captures non-linear relationships but is sensitive to
bandwidth choice.

### What Questions It Answers

- "Does the DPO model's layer 20 look more like the base model's or the SFT model's?"
- "At which layers did DPO most change the representation away from SFT?"
- "Are the early layers preserved across interventions?" (Expected: yes, early layers change
  little.)

### Implementation Sketch

```python
# Linear CKA (simple implementation)
def linear_CKA(X, Y):
    """X: (n_samples, d1), Y: (n_samples, d2)"""
    # Center
    X = X - X.mean(0)
    Y = Y - Y.mean(0)

    hsic_xy = np.linalg.norm(X.T @ Y, 'fro') ** 2
    hsic_xx = np.linalg.norm(X.T @ X, 'fro') ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, 'fro') ** 2

    return hsic_xy / np.sqrt(hsic_xx * hsic_yy)
```

### Practical Considerations

- **Number of samples:** CKA needs N >> 1 shared inputs. ~500-1000 prompts is typical.
- **Debiased CKA:** The naive estimator is biased when the number of features is large
  relative to samples. Use unbiased HSIC estimators (Murphy et al., 2024).
- **Layer-layer heatmap:** Compute CKA for all (layer_i_model_A, layer_j_model_B) pairs to
  produce a full similarity matrix. Diagonal = same-layer correspondence.
- **GPU memory:** Only need the activation matrices, not the full model simultaneously.
  Extract activations, save to disk, compute CKA offline.

### Relevance to Our Study

**Key tool for Phase 5.** Create CKA heatmaps comparing:
1. Base Qwen3-8B vs SFT model (where did SFT change representations?)
2. SFT model vs DPO-recovered model (where did DPO change things back?)
3. Base model vs DPO-recovered model (does DPO recovery look like base?)

If DPO makes middle layers look like base again -> genuine recovery.
If DPO only changes late layers -> surface-level suppression.

### Key Papers
- Kornblith et al., "Similarity of Neural Network Representations Revisited" (ICML 2019) -- original CKA paper
- Murphy et al., "Estimating Neural Representation Alignment" (2025) -- bias correction

---

## 4. RSA (Representational Similarity Analysis) -- Another Way to Compare Model Internals

### How It Works Mechanically

RSA compares two representation spaces via their pairwise distance structure, formalized as
Representational Dissimilarity Matrices (RDMs).

**Steps:**
1. Pass N stimuli through model/layer, extract activations -> matrix X (N x d).
2. Compute the RDM: an N x N matrix where entry (i,j) = distance(x_i, x_j).
   - Common distance metrics: cosine distance, Euclidean, correlation distance.
3. Repeat for another model/layer -> second RDM.
4. Compare RDMs using Spearman or Pearson correlation (or Kendall tau).
5. High correlation = similar representational geometry.

### CKA vs RSA

| Aspect | CKA | RSA |
|--------|-----|-----|
| What it compares | Kernel matrices (inner products) | Distance matrices |
| Invariance | Rotation-invariant, scale-invariant | Depends on distance metric |
| Statistical power | Generally higher | Can be lower for subtle differences |
| Theoretical basis | HSIC (kernel independence) | Correlation of distance matrices |
| Equivalence | CKA ~ RSA with mean-centering (proven in 2025) |

Recent theoretical work (CCNeuro 2025) proves CKA and RSA are equivalent once mean-centering
is applied to RSA, so they give essentially the same information.

### What Questions It Answers

- Same as CKA (comparing representation geometry across models/layers).
- Additionally: cross-modal comparisons (e.g., comparing LLM representations to brain
  activation patterns -- less relevant to us but shows RSA flexibility).
- U-shaped similarity patterns across layers (observed in Qwen and Gemma models): early and
  late layers are similar to each other, middle layers are maximally different.

### Relevance to Our Study

RSA and CKA are largely interchangeable for our purposes. Use CKA as primary (more common in
deep learning literature, better statistical properties), RSA as robustness check.

### Key Papers
- Kriegeskorte et al., "Representational Similarity Analysis" (Frontiers in Neuroscience, 2008)
- Equivalence proof: Williams et al. (CCNeuro 2025)
- rsatoolbox (Python): Schutt et al. (eLife 2024)

---

## 5. Probing Classifiers Beyond Linear -- MLP Probes

### When Are They Needed?

| Linear Probe Result | MLP Probe Result | Interpretation |
|---------------------|------------------|----------------|
| High accuracy | High accuracy | Information is linearly encoded (use linear) |
| Low accuracy | High accuracy | Information is present but non-linearly entangled |
| Low accuracy | Low accuracy | Information likely not encoded at this layer |
| High accuracy | Higher accuracy | Linear captures most; MLP picks up residual |

### What Does It Mean if Linear Fails but MLP Succeeds?

The information is encoded in a **non-linear manifold** in activation space. This has
interpretive implications:
- The concept is present but **entangled** with other features.
- The model may use non-linear computations (MLP layers, attention patterns) to access it.
- It is harder to do activation steering with a simple direction vector.
- The concept may be **distributed** across multiple interacting features rather than a
  single direction.

### The Expressiveness Trade-Off

MLP probes (even 2-layer with ~100 hidden units) are powerful enough to potentially **learn
the classification task themselves** rather than just reading out encoded information. This
creates ambiguity: did the model encode sycophancy, or did the MLP learn to detect it from
surface features in the activation space?

**Best practice:**
- Always train linear probes first.
- Only move to MLP probes if linear probes fail.
- If MLP succeeds where linear fails, this is an interesting finding in itself (non-linear
  encoding), but interpret cautiously.
- Use **selectivity** as a control: train the probe on a control task (e.g., random labels)
  to measure baseline accuracy of the probe architecture.

### Practical Implementation

```python
# 2-layer MLP probe
class MLPProbe(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
```

Keep hidden_dim small (64-256) to limit expressiveness. Regularize with dropout (0.3-0.5)
and early stopping.

### Relevance to Our Study

If linear probes show sycophancy is linearly encoded in the SFT model, we can:
1. Use the linear probe direction for activation steering experiments.
2. Track this direction across recovery interventions.
3. If linear fails, MLP success would suggest sycophancy is a more complex phenomenon in
   the model's representation -- important finding for the paper.

---

## 6. Layer-Wise Analysis -- Where to Expect Sycophancy Signal

### What Different Layers Encode (Converging 2024-2025 Evidence)

| Layer Region | Typical Depth (36-layer model) | What It Encodes |
|-------------|-------------------------------|-----------------|
| Early (1-8) | Layers 1-8 | Token embeddings, shallow syntax, positional info |
| Early-mid (9-14) | Layers 9-14 | Deeper syntax, basic semantics, entity recognition |
| Middle (15-24) | Layers 15-24 | **Safety alignment, factual knowledge, truthfulness/deception, sycophancy** |
| Late (25-32) | Layers 25-32 | Output formatting, task-specific heads, attention sinks |
| Final (33-36) | Layers 33-36 | Logit prediction, very task-specific |

### Safety Layers

Wei et al. (2024) identified **"safety layers"** -- a small set of contiguous middle layers
critical for distinguishing malicious from normal queries. These layers are where alignment
training has its strongest effect.

**Implication for sycophancy:** SFT sycophancy training likely modified the middle layers
most. DPO recovery should also modify these layers if it genuinely reverses the effect.

### Deception/Alignment Processing

The "Universal Motif" paper (ICLR 2025) found that across 20 models from 4 families
(Llama, Gemma, Yi, Qwen), deceptive behavior is causally controlled by a **sparse set of
layers and attention heads** concentrated in the third processing stage (late-middle layers).
Steering only these layers effectively reduced lying.

### Layer Significance for Alignment

Zheng et al. (2025) found that when fine-tuning on different alignment datasets, the
important layers show **86% Jaccard similarity** -- layer importance is consistent across tasks,
not dataset-specific.

### Practical Strategy for Our Study

1. **Probe all 36 layers** of Qwen3-8B (it is cheap -- just logistic regression on cached
   activations).
2. Plot AUROC vs layer number for sycophancy detection.
3. Expect peak signal in **layers 14-26** (middle layers).
4. Compare the AUROC-by-layer curve across: base -> SFT -> DPO/RLHF/CAI.
5. If DPO only reduces AUROC at late layers but middle layers retain the signal:
   **suppression, not removal.**

---

## 7. Practical Implementation

### Libraries and Tools

| Tool | Best For | Qwen3-8B Support | Notes |
|------|----------|-------------------|-------|
| **HuggingFace Transformers** | Direct activation extraction | Native | `output_hidden_states=True`, simplest approach |
| **NNsight** (ICLR 2025) | Flexible interventions, activation patching | Any PyTorch model | Clean API, `with model.trace(input):` context manager |
| **nnterp** (2025) | Standardized interface across architectures | 50+ model variants | Wrapper around NNsight, includes logit lens, steering built-in |
| **TransformerLens** | Mechanistic interpretability, circuit analysis | Limited (GPT-style focus) | May need custom adaptation for Qwen3 |
| **baukit** | Simple activation tracing/editing | Any PyTorch model | Predecessor to NNsight, still useful for its simplicity |
| **sklearn** | Logistic regression probes | N/A (pure numpy/scipy) | `LogisticRegression(max_iter=1000, C=1.0)` |
| **PING** | End-to-end probing pipeline | HF-compatible models | Open-source, handles extraction + probe training |

### Recommended Approach for Qwen3-8B

Use **HuggingFace Transformers directly** for activation extraction (simplest, native Qwen3
support), then **sklearn** for linear probes. NNsight for activation patching/steering if
we get to that phase.

### Extracting Activations from Qwen3-8B

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "path/to/qwen3-8b",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    output_hidden_states=True  # Key flag
)
tokenizer = AutoTokenizer.from_pretrained("path/to/qwen3-8b")

# Extract hidden states
inputs = tokenizer(text, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model(**inputs)

# outputs.hidden_states is a tuple of (n_layers + 1) tensors
# Each tensor shape: (batch, seq_len, 4096)
# Index 0 = embedding layer output, Index 1-36 = transformer layer outputs
last_token_states = [hs[:, -1, :] for hs in outputs.hidden_states]
# Now last_token_states[L] is the last-token activation at layer L
```

### GPU Memory Considerations

| Configuration | VRAM Needed | Notes |
|--------------|-------------|-------|
| Qwen3-8B bf16, all hidden states | ~20-22 GB | Model (~17GB) + activation overhead |
| Qwen3-8B 4-bit quantized, all hidden states | ~8-10 GB | Fits single consumer GPU |
| Batch of 32 prompts, all layers | +4-8 GB | Activation memory scales with batch x seq_len x layers |

**Memory optimization strategies:**
- Process prompts one at a time or in small batches (4-8).
- Extract and save activations to disk per-batch, then train probes offline.
- Only extract specific layers if memory is tight (e.g., every 4th layer for initial scan,
  then all layers in the peak region).
- Use 4-bit quantization for extraction (bitsandbytes) -- probe accuracy is minimally
  affected since we are reading the representational geometry, not generating text.
- For our H100 80GB setup: can comfortably fit bf16 model + full batch of activations.

**Important Qwen3 note:** In vLLM, `hidden_states[-1]` may differ from Transformers due to
normalization. Set `LAST_HIDDEN_STATE_NOT_NORM=1` for exact match. For probing, use
Transformers directly, not vLLM.

### Workflow for Our Study

1. **Build probe dataset:** Use our existing eval datasets (answer_sycophancy, are_you_sure,
   feedback) to generate (prompt, response) pairs from both the SFT model (sycophantic) and
   base model (non-sycophantic). Label by evaluator scores.
2. **Extract activations:** Run each pair through the frozen SFT model, save last-token
   hidden states at all 36 layers to disk as numpy arrays.
3. **Train linear probes:** For each layer, train logistic regression. Plot AUROC vs layer.
4. **Repeat for recovered models:** Extract activations from DPO/RLHF/CAI models using the
   same prompts. Apply the SFT-trained probe AND retrain new probes.
5. **Compute CKA:** Compare layer-wise representations across base, SFT, and recovered models.
6. **Interpret:** If sycophancy probe AUROC remains high in recovered model -> cosmetic fix.
   If CKA shows recovered model middle layers look like base -> genuine reversal.

---

## 8. Recent Key Papers Summary (2024-2026)

| Paper | Year | Key Contribution | Relevance |
|-------|------|-----------------|-----------|
| Papadatos & Freedman, "Linear Probe Penalties Reduce LLM Sycophancy" | 2024 | Linear probes detect sycophancy in reward model; penalty reduces it via RLHF | Direct methodology for our study |
| Park et al., "Linear Representation Hypothesis" | 2024 | Concepts encoded as linear directions in activation space | Theoretical foundation for linear probing |
| "Interpretability of LLM Deception: Universal Motif" | 2025 (ICLR) | Sparse set of layers/heads causally responsible for lying, universal across 20 models | Informs where to look for sycophancy circuits |
| Wei et al., "Safety Layers in Aligned LLMs" | 2024 | Middle layers are "safety layers" critical for alignment | Predicts where SFT/DPO changes concentrate |
| Schouten et al., "LLM Probing with Contrastive Eigenproblems" | 2025 | CCS reformulated as eigenproblem, closed-form solutions | Improved unsupervised probing |
| Levinstein & Herrmann, "Probing the Limits of the Lie Detector Approach" | 2026 | Truth probes fail at non-lying deception; recommends second-order belief probes | Limitation to acknowledge |
| "Poser: Unmasking Alignment Faking LLMs" | 2024 | Mass mean probes detect alignment fakers via contrastive activation analysis | Methodology for detecting hidden sycophancy |
| Hubinger et al., "Sleeper Agents" | 2024 | Alignment faking models detectable with simple probes (AUROC > 0.99) | Shows probing can catch hidden behaviors |
| PING framework | 2025 | Open-source toolkit for probing frozen transformers, calibrated predictions | Practical implementation reference |
| Zheng et al., "Understanding Layer Significance in LLM Alignment" | 2025 | 86% Jaccard similarity in important layers across alignment tasks | Layer importance is consistent |
| Kornblith et al., "Similarity of Neural Network Representations Revisited" | 2019 | Original CKA paper | Foundation for representation comparison |

---

## 9. Open Questions and Risks for Our Study

1. **Probe generalization:** Will a sycophancy probe trained on MCQ-style sycophancy transfer
   to open-ended feedback sycophancy? Need to test cross-category generalization.

2. **Confounded features:** Sycophantic responses may systematically differ in length,
   sentiment, hedging patterns. The probe might detect these surface features rather than a
   genuine "sycophancy concept." Control experiments needed (e.g., length-matched pairs).

3. **Sample efficiency:** We have ~3,236 labeled pairs per category. After filtering by
   evaluator scores, the effective probe training set may be smaller. Should be sufficient
   for linear probes (hundreds suffice) but monitor for overfitting.

4. **Recovery model interpretation:** If DPO model shows reduced probe AUROC, is this because:
   (a) sycophancy was genuinely removed, or (b) the representation changed enough that the
   old probe direction no longer applies? Need to retrain probes on each model independently
   AND apply cross-model probes.

5. **CKA at scale:** Computing CKA for all layer pairs across 4+ models with ~1000 inputs is
   manageable but requires storing ~36 x 1000 x 4096 activations per model (~590 MB per model
   in bf16). Easily fits in memory.

6. **Causal validation:** Probing is correlational. If we find a sycophancy direction,
   activation steering along that direction should change sycophancy behavior. This bridges
   probing (Phase 5) with steering (Phase 4).
