# Mechanistic Interpretability Techniques for Understanding Sycophancy

Research compiled 2026-03-22. Covers causal/mechanistic interpretability methods relevant to understanding whether alignment interventions genuinely remove sycophancy or merely suppress its surface expression.

---

## 1. Causal Tracing / Activation Patching

### How It Works

Causal tracing (also called activation patching or interchange intervention) identifies which model components are causally responsible for a specific behavior by running a controlled experiment inside the network:

1. **Clean run**: Process the input normally, record all intermediate activations
2. **Corrupted run**: Corrupt the input (e.g., replace subject tokens with noise) so the model fails at the task
3. **Patch run**: Re-run the corrupted input but restore (patch in) specific activations from the clean run one component at a time
4. **Measure recovery**: If patching component X restores the correct behavior, that component is causally important

The corruption method matters. Common approaches: Gaussian noise added to embeddings, random token substitution, or zero ablation. The choice of corruption and metric can significantly affect results (see best practices paper below).

### What Questions It Answers

- Which layers, attention heads, or MLP blocks are causally responsible for a behavior?
- Where in the network is information about user opinions stored vs. factual knowledge?
- Does sycophancy recovery actually change the causal components, or route around them?

### Application to Sycophancy

For our study, causal tracing can answer a critical question: **do recovery methods (DPO, RLHF, CAI) change the same components that drive sycophantic behavior, or do they add new suppression circuits?**

Concrete approach:
- Create contrastive pairs: (prompt with user opinion, prompt without) where the sycophantic model flips its answer
- Trace which components mediate the opinion-sensitivity
- Compare causal traces before and after each recovery intervention
- If recovery changes the same components: genuine removal. If different components light up: suppression circuit

### Implementation Complexity

**Medium.** TransformerLens provides hook-based access to all activations. The core loop is: forward pass with hooks to cache activations, forward pass with corruption, forward pass replacing specific activations. Main cost is compute: O(N) forward passes for N components. For Qwen3-8B with 36 layers, ~200+ components to test per example.

**Scaling trick**: Attribution patching (Neel Nanda, 2023) uses gradients as a linear approximation, reducing cost to 1 backward + 2 forward passes to score all components simultaneously. Less precise but good for initial screening.

### Key Papers/References

- Meng et al., "Locating and Editing Factual Associations in GPT" (NeurIPS 2022) -- introduced causal tracing for ROME. Found MLPs at middle layers store factual knowledge. https://arxiv.org/abs/2202.05262
- Goldowsky-Dill et al., "Towards Best Practices of Activation Patching" (2023) -- systematic study of methodological choices. https://arxiv.org/abs/2309.16042
- Neel Nanda, "Attribution Patching: Activation Patching at Industrial Scale" -- gradient-based approximation. https://www.neelnanda.io/mechanistic-interpretability/attribution-patching

### Relevance to Our Study: HIGH

This is the most direct technique for answering "did recovery actually change the sycophancy mechanism or just add a mask?" Our linear probes test representational change; causal tracing tests causal change. They complement each other.

---

## 2. Attention Head Analysis

### How It Works

Transformer attention heads compute attention weights that determine how much each token position attends to every other position. For sycophancy, the key question is: which heads attend to user pressure cues (e.g., "I think the answer is X", "Are you sure?") and how does that attention influence downstream computation?

The analysis pipeline:
1. **Extract attention patterns**: Hook into attention layers, save the attention weight matrices (shape: [heads, seq_len, seq_len])
2. **Identify pressure tokens**: Mark which tokens in the input carry user opinion/pressure
3. **Measure attention to pressure**: For each head, compute how much attention flows from response-generating positions to user pressure positions
4. **Validate causally**: Use activation patching on specific heads to confirm causal role

### Visualization

Attention pattern heatmaps show which tokens attend to which. For sycophancy, you look for heads where response tokens disproportionately attend to opinion tokens. Tools: BertViz, circuitsvis (from TransformerLens ecosystem), or custom matplotlib heatmaps.

### What Questions It Answers

- Which attention heads "read" user opinions?
- Do certain heads act as "opinion amplifiers" that strengthen the influence of user preferences?
- After recovery training, do these heads change their attention patterns?

### Implementation Complexity

**Low-Medium.** Extracting attention weights is trivial with hooks. The harder part is meaningful analysis -- raw attention weights can be misleading (attention != importance). Best combined with causal validation via head-level activation patching.

### Key Papers/References

- Olsson et al., "In-context Learning and Induction Heads" (2022) -- foundational work on categorizing attention head functions
- Chen et al., "From Yes-Men to Truth-Tellers" (ICML 2024) -- specifically identified sycophancy-related attention heads via path patching

### Relevance to Our Study: MEDIUM

Useful as a diagnostic and visualization tool. Less rigorous than causal tracing alone but provides intuitive visual evidence. Best used alongside SPT-style head identification.

---

## 3. Pinpoint Tuning (SPT) -- ICML 2024

### How It Works

Supervised Pinpoint Tuning (SPT) is a two-stage approach:

**Stage 1: Identify sycophancy heads via path patching**
- Construct contrastive input pairs where sycophantic vs. honest responses differ
- For each attention head, perform a hard intervention: replace the head's output with its output from the counterfactual input
- Measure the change in logit difference between sycophantic and honest tokens
- Rank heads by their causal effect on sycophancy
- Result: only ~4% of attention heads significantly influence sycophantic behavior

**Stage 2: Fine-tune only the identified heads**
- Freeze all model parameters except the Q, K, V, O projection matrices of the identified heads
- Train on an anti-sycophancy dataset (correct answers even when user disagrees)
- Because only a tiny fraction of parameters update, general capability is preserved

### Results (from the paper)

Tested on Llama-2-7B and Llama-2-13B:
- SPT achieves higher confidence (+71.84%) and truthfulness (+67.83%) than full SFT (+61.47%, +65.17%) on Llama-2-13B
- Critical advantage: SFT decreased GSM8K by 8.57%, while SPT *improved* it by 1.59%
- SPT is orthogonal to LoRA -- combining LoRA + SPT gives an additional 15% boost over SPT alone

### What Questions It Answers

- Which specific attention heads drive sycophancy? (Gives a concrete list)
- Can we fix sycophancy by surgically modifying only those heads?
- Does surgical modification preserve general capabilities better than broad fine-tuning?

### Implementation Complexity

**Medium-High.**
- Path patching infrastructure needed (similar to causal tracing but at head granularity)
- Custom training loop that freezes all params except identified head projections
- Code available: https://github.com/yellowtownhz/sycophancy-interpretability

### Application to Our Study

SPT is directly relevant as a **fifth recovery method** or as an **analysis tool**:
- Run path patching on our SFT sycophantic model to identify the heads
- Compare: do DPO/RLHF/CAI change the same heads that SPT identifies?
- If DPO changes different components than the SPT-identified heads, it suggests DPO works via a different mechanism (possibly suppression rather than removal)
- Could also run SPT as a recovery method and compare its depth of removal

### Key Papers/References

- Chen et al., "From Yes-Men to Truth-Tellers: Addressing Sycophancy in Large Language Models with Pinpoint Tuning" (ICML 2024). https://arxiv.org/abs/2409.01658
- Code: https://github.com/yellowtownhz/sycophancy-interpretability

### Relevance to Our Study: VERY HIGH

This directly identifies the mechanistic locus of sycophancy. Even if we don't use SPT as a recovery method, the head identification procedure is invaluable for evaluating whether other recovery methods actually modify the right components.

---

## 4. CAUSM -- ICLR 2025

### How It Works

CAUSM (Causally Motivated Sycophancy Mitigation) models sycophancy through Structural Causal Models (SCMs). The core insight: sycophancy arises from spurious correlations between user preferences and model outputs in latent space.

**The causal model:**
- User preference -> (spurious path) -> Model output (sycophancy)
- Question content -> (causal path) -> Model output (correct answer)
- CAUSM aims to eliminate the spurious path while preserving the causal path

**Two variants:**

1. **CAUSM (Base) -- Pruning approach:**
   - Use a leave-one-out (LOO) measure to score how much each attention head contributes to the spurious correlation
   - Prune the top K=10 most sycophancy-contributing heads (set their outputs to zero)
   - Training-free, inference-time intervention

2. **CAUSM (CAC) -- Causal Activation Calibration:**
   - Identify a "causal direction" in activation space that separates sycophantic from honest processing
   - For K=48 heads, calibrate activations by projecting out the spurious direction and amplifying the causal direction
   - Uses a mixing coefficient lambda=0.1
   - Better OOD generalization than pruning

### Improvement Over SPT

- SPT identifies important heads but fine-tunes them with standard SFT objective
- CAUSM provides a *causal framework* for understanding *why* those heads matter (spurious correlation vs. causal processing)
- CAUSM's calibration approach modifies the direction of computation within heads, not just the head weights
- Better out-of-distribution generalization (tested on MMLU, MATH, AQuA, TriviaQA)

### Results

- CAUSM (CAC) outperforms baselines on TriviaQA (66.56% -> 69.45%) and MATH (45.18% -> 48.34%)
- Disentangled representations focus more on question tokens and less on user preference tokens
- Robust OOD generalization across sycophancy scenarios not seen during calibration

### Implementation Complexity

**Medium-High.** Requires:
- Implementing the SCM framework and LOO importance scoring
- Computing causal directions via contrastive activation analysis
- Head-level intervention infrastructure (similar to SPT)

### Application to Our Study

CAUSM's causal direction concept connects directly to our linear probes:
- Our probes detect whether sycophancy is represented in activations
- CAUSM's causal direction IS a specific linear direction related to sycophancy
- We could compare: does the probe direction align with CAUSM's causal direction?
- If DPO eliminates the probe-detectable direction but not the CAUSM causal direction (or vice versa), it reveals partial removal

### Key Papers/References

- Li et al., "CAUSM: Causally Motivated Sycophancy Mitigation for Large Language Models" (ICLR 2025). https://openreview.net/forum?id=yRKelogz5i
- PDF: https://proceedings.iclr.cc/paper_files/paper/2025/file/a52b0d191b619477cc798d544f4f0e4b-Paper-Conference.pdf

### Relevance to Our Study: HIGH

The causal framework provides a principled way to distinguish genuine removal from suppression. The causal direction concept directly complements our linear probe approach.

---

## 5. Circuit Analysis

### How It Works

Circuit analysis identifies the minimal computational subgraph (circuit) within a transformer that implements a specific behavior. A circuit consists of nodes (attention heads, MLP layers, embedding components) and edges (the residual stream connections between them).

**Core methods for circuit discovery:**

1. **Path Patching (manual):**
   - Like activation patching but tracks information flow along specific paths
   - Patch the output of component A, but only allow the effect to propagate through component B (block other paths)
   - If patching A->B changes behavior, that edge is part of the circuit
   - Precise but O(N^2) for N components

2. **ACDC (Automated Circuit DisCovery):**
   - Greedy search: ablate edges one by one, keep those whose removal changes behavior
   - Thorough but slow -- scales poorly to large models

3. **Edge Attribution Patching (EAP):**
   - Gradient-based approximation: score all edges simultaneously with 1 backward + 2 forward passes
   - Much faster but linear approximation can miss nonlinear interactions
   - Good for initial screening, validate important edges with actual patching

4. **Edge Pruning (EP):**
   - Frame circuit discovery as optimization: learn binary masks over edges
   - Gradient-based, parallelizable across GPUs
   - Scales to 13B+ models (demonstrated on CodeLlama-13B)
   - Produces smaller, more faithful circuits than ACDC or EAP

### Anthropic's Circuit Tracing (2025)

Anthropic's approach is distinct: they first decompose activations into interpretable features using cross-layer transcoders (not standard SAEs), then build attribution graphs showing how features interact:
- **Nodes** = learned features (interpretable concepts like "user is disagreeing" or "factual claim")
- **Edges** = causal influence between features across layers
- Applied to Claude 3.5 Haiku, revealed multi-step reasoning, planning in poetry generation, and feature supernodes

Open-sourced tools support Gemma-2-2B, Llama-3.2-1B, Qwen3-4B. Community re-implementations by EleutherAI (Attribute library) and Goodfire.

### What Questions It Answers

- What is the complete computational pathway from "user states opinion" to "model agrees sycophantically"?
- Which edges in the circuit are modified by recovery training?
- Is there a distinct "suppression circuit" that recovery methods add?

### Implementation Complexity

**High.** Full circuit analysis is the most labor-intensive technique:
- Path patching at edge level is O(N^2) in components
- ACDC and EP require significant infrastructure
- Anthropic's approach requires training transcoders first
- TransformerLens helps but the analysis itself is research-grade work

**Practical recommendation:** Use EAP for initial screening, validate key edges with actual path patching. Don't attempt full circuit discovery on 8B model without significant compute budget.

### Key Papers/References

- Conmy et al., "Towards Automated Circuit Discovery for Mechanistic Interpretability" (2023) -- ACDC
- Syed et al., "Attribution Patching Outperforms Automated Circuit Discovery" (2023) -- EAP
- Bhaskar et al., Edge Pruning for scalable circuit discovery (2025)
- Anthropic, "Circuit Tracing: Revealing Computational Graphs in Language Models" (2025). https://transformer-circuits.pub/2025/attribution-graphs/methods.html
- Anthropic open-source tools: https://www.anthropic.com/research/open-source-circuit-tracing

### Relevance to Our Study: MEDIUM-HIGH

Full circuit discovery is likely too expensive for our timeline. However, targeted circuit analysis (e.g., trace the sycophancy circuit using EAP, then check if recovery methods modify those edges) is feasible and highly informative.

---

## 6. Sparse Autoencoders (SAEs)

### How It Works

SAEs address the **superposition** problem: neural networks represent more concepts than they have neurons, so individual neurons are polysemantic (fire for multiple unrelated concepts). SAEs decompose activations into a larger set of sparse, monosemantic features.

**Architecture:**
```
Input: activation vector h (d_model dimensions, e.g., 4096 for Qwen3-8B)
Encoder: f = ReLU(W_enc @ h + b_enc)    # f is sparse, high-dimensional (e.g., 16384 or 65536)
Decoder: h_hat = W_dec @ f + b_dec       # reconstruct original activation
Loss: ||h - h_hat||^2 + lambda * ||f||_1  # reconstruction + sparsity
```

The sparsity penalty (L1) ensures only a few features activate for any given input. Each feature (column of W_dec) represents a direction in activation space that corresponds to an interpretable concept.

**What a "feature" looks like:**
- Feature 4721 might activate on tokens where the model is about to agree with someone
- Feature 12003 might activate when the input contains a factual claim the model knows is wrong
- Feature 8455 might activate on user expressions of displeasure or challenge

**Current best practice (as of late 2025):**
- Use **BatchTopK** or **JumpReLU** SAEs (not vanilla ReLU -- outdated)
- BatchTopK directly controls sparsity level without tuning lambda
- Wider latent spaces (more features) produce more interpretable results
- Can train a decent SAE for a real LLM on a single A100 in hours

### SAELens -- Primary Tool

SAELens (Bloom, Tigges, Duong, Chanin, 2024) is the standard library:
- Works with any PyTorch model, deep integration with TransformerLens
- Pre-trained SAEs available on HuggingFace for many models (Gemma, Llama, GPT-2, etc.)
- Companion tools: SAE-Vis (visualization), SAEBench (benchmarking)
- Training config via `LanguageModelSAERunnerConfig`
- Can train on residual stream, attention outputs, or MLP outputs at any layer

**Important note:** No pre-trained SAEs exist for Qwen3-8B as of this writing. We would need to train our own, which is feasible but adds work.

### Identifying Sycophancy-Related Features

Approach:
1. Train SAEs on residual stream activations at key layers (middle layers where sycophancy is likely encoded)
2. Collect activations for sycophantic vs. honest responses
3. Find features that differentially activate: high on sycophantic, low on honest (or vice versa)
4. Validate: ablate those features and check if sycophancy changes
5. Compare feature activation patterns before vs. after recovery methods

### What Questions It Answers

- What interpretable features does the model use when being sycophantic?
- Are there specific features for "user is challenging me" or "I should agree even though I'm wrong"?
- After DPO/RLHF recovery, are those features still present but suppressed, or genuinely gone?

### Implementation Complexity

**Medium-High.**
- Training SAEs: straightforward with SAELens, needs 1 GPU for several hours per layer
- Finding relevant features: requires contrastive analysis on curated datasets
- Validation: ablation studies add compute cost
- No pre-trained Qwen3-8B SAEs -- must train from scratch

**Practical note from the field:** Google DeepMind published "Negative results for SAEs on downstream tasks" (2025), indicating SAEs don't always help for practical alignment applications. However, for *understanding* mechanisms (our goal), they remain very useful even if they don't directly fix the behavior.

### Key Papers/References

- Cunningham et al., "Sparse Autoencoders Find Highly Interpretable Features in Language Models" (2023). https://arxiv.org/abs/2309.08600
- Bloom et al., "SAELens" (2024). https://github.com/decoderesearch/SAELens
- Adam Karvonen, "An Intuitive Explanation of Sparse Autoencoders for LLM Interpretability" (2024). https://adamkarvonen.github.io/machine_learning/2024/06/11/sae-intuitions.html
- Universal SAEs (ICML 2025) -- cross-model feature alignment

### Relevance to Our Study: MEDIUM-HIGH

SAEs provide the most interpretable decomposition of what the model is doing internally. The main barrier is needing to train SAEs for Qwen3-8B from scratch. If we invest in this, we get a powerful lens into whether recovery methods change the same features that drive sycophancy.

---

## 7. Logit Lens / Tuned Lens

### How It Works

**Logit Lens** (nostalgebraist, 2020): Take the hidden state at each intermediate layer and project it through the model's final unembedding matrix (the same matrix the final layer uses to produce logits). This gives a probability distribution over tokens at each layer, showing what the model would "predict" if computation stopped at that layer.

```python
# Pseudocode
for layer_idx in range(n_layers):
    hidden = model.get_hidden_state(input, layer=layer_idx)
    logits = model.lm_head(model.ln_f(hidden))  # project to vocab
    probs = softmax(logits)
    # Examine: what token does the model "think" at this layer?
```

**Key insight:** Models don't keep raw inputs around and gradually process them. Instead, inputs are immediately transformed into a prediction-like representation that is progressively refined. The model often has a "pretty good guess" well before the final layer.

**Tuned Lens** (Belrose et al., 2023): Addresses a limitation -- different layers use different internal representations (representational drift), so raw projection through the unembedding matrix can be unreliable. The tuned lens learns a lightweight affine transformation per layer:

```python
# Per-layer learned probe
logits_layer_i = model.lm_head(model.ln_f(A_i @ hidden_i + b_i))
```

Where A_i, b_i are learned to minimize KL divergence with the final layer's output distribution. This corrects for representational drift and gives more reliable intermediate predictions.

**LogitLens4LLMs** (2025): Extended the original logit lens to modern architectures including Llama-3, Qwen, and others. Open-sourced at https://github.com/zhenyu-02/LogitLens4LLMs.

### What Questions It Answers

- At which layer does the model "decide" to be sycophantic vs. honest?
- Does the sycophantic answer appear early and get refined, or does it emerge suddenly at a specific layer?
- After recovery training, does the layer-by-layer trajectory change?

### Application to Sycophancy

For a prompt where the model will sycophantically agree with an incorrect user opinion:
1. Run logit lens at each layer
2. Track: at which layer does P(sycophantic_token) exceed P(honest_token)?
3. Compare trajectories: baseline model vs. SFT sycophantic model vs. DPO-recovered model
4. If DPO-recovered model shows the sycophantic token emerging early then being suppressed late -- evidence of surface-level suppression, not genuine removal

This directly tests our core hypothesis about cosmetic vs. genuine recovery.

### Implementation Complexity

**Low.** This is the simplest technique in this document:
- Logit lens: just hook intermediate hidden states and multiply by unembedding matrix
- Tuned lens: train small affine probes (minutes of training)
- LogitLens4LLMs supports Qwen architecture out of the box
- No special infrastructure needed beyond standard forward pass hooks

### Key Papers/References

- nostalgebraist, "interpreting GPT: the logit lens" (LessWrong, 2020). https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
- Belrose et al., "Eliciting Latent Predictions from Transformers with the Tuned Lens" (2023). https://arxiv.org/abs/2303.08112
- LogitLens4LLMs (2025). https://arxiv.org/html/2503.11667v1

### Relevance to Our Study: HIGH

Low implementation cost, high diagnostic value. The layer-by-layer prediction trajectory directly reveals whether recovery methods prevent sycophantic predictions from forming (genuine removal) or allow them to form and then override them (suppression). Should be one of the first techniques we implement.

---

## Additional Relevant Work

### "Sycophancy Is Not One Thing" (Vennemeyer et al., 2025)

Key finding for our study: sycophantic agreement and sycophantic praise are **distinct behaviors encoded along orthogonal linear directions** in latent space. They can be independently amplified or suppressed. Genuine agreement and sycophantic agreement overlap in early layers but diverge in later layers.

Implication: our linear probes should test for multiple sycophancy sub-behaviors, not treat sycophancy as monolithic. A recovery method might eliminate one direction but not the other.

- Paper: https://arxiv.org/abs/2509.21305

### "When Truth Is Overridden" (2025)

Sycophancy is opinion-driven, not authority-driven. Models agree with incorrect user opinions regardless of claimed expertise. The model fails to represent authority internally.

Implication: our eval scenarios testing user expertise vs. opinion should focus on opinion content, not authority framing.

- Paper: https://arxiv.org/html/2508.02087v1

### Contrastive Activation Addition (CAA) -- Rimsky et al. (ACL 2024)

Steering vectors from difference-in-means of residual stream activations. For sycophancy on Llama 2, layers 15-17 are most influential. Subtracting the sycophancy vector also improves TruthfulQA. Stacks additively with fine-tuning.

- Paper: https://arxiv.org/abs/2312.06681
- Code: https://github.com/nrimsky/CAA

### Feature Guided Activation Additions (FGAA) -- ICLR 2025

Combines SAEs with CAA: uses SAE features to guide more precise activation additions. Outperforms vanilla CAA on both steering effectiveness and output coherence. Tested on Gemma-2-2B and Gemma-2-9B.

### Contrastive Causal Mediation (CCM) -- 2025

Selects steerable model components from long-form responses using causal mediation rather than correlational probes. Outperforms probe-based component selection for sycophancy steering.

---

## Technique Comparison Summary

| Technique | Complexity | Compute Cost | What It Reveals | Priority for Our Study |
|-----------|-----------|-------------|-----------------|----------------------|
| Logit Lens / Tuned Lens | Low | Minimal | Layer-by-layer prediction trajectory | **P0 -- implement first** |
| Causal Tracing | Medium | Medium | Which components are causally responsible | **P0 -- core analysis** |
| Attention Head Analysis | Low-Med | Low | Which heads attend to user pressure | P1 -- diagnostic |
| SPT Head Identification | Med-High | Medium | Specific heads driving sycophancy | **P1 -- key for comparing methods** |
| Linear Probes (already planned) | Low | Low | Whether sycophancy is representationally encoded | **P0 -- already in plan** |
| CAUSM Causal Direction | Med-High | Medium | Causal vs. spurious correlation structure | P1 -- complements probes |
| SAEs | Med-High | High (training) | Interpretable feature decomposition | P2 -- high value but high cost |
| Full Circuit Analysis | High | Very High | Complete computational subgraph | P2 -- selective use only |

## Recommended Analysis Pipeline for Our Study

1. **Linear probes** (already planned) -- detect if sycophancy is representationally present post-recovery
2. **Logit lens** -- track layer-by-layer trajectory to see if sycophancy forms then gets suppressed vs. never forms
3. **Causal tracing** -- identify which components drive sycophancy, check if recovery methods modify those components
4. **SPT-style head identification** -- get a concrete list of sycophancy heads, compare across pre/post recovery
5. **SAEs** (if time permits) -- decompose sycophancy into interpretable features for the most fine-grained analysis

This pipeline goes from cheapest/fastest to most expensive, with each layer adding confidence to the conclusion about genuine vs. cosmetic removal.
