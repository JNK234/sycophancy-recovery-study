# Experiment 004: Linear Probing v1 — FLAWED

## Status: INVALID — Redesigned as Experiment 005

## What We Did

Built a linear probing pipeline (`src/probing/`) to test whether sycophancy is still encoded internally after DPO recovery. Extracted hidden state activations from base, SFT, and DPO models on 1000 contrastive samples, trained logistic regression probes per layer.

## Results

All three models scored nearly identically (~0.90 AUROC), which should not happen — the base model was never trained to be sycophantic.

| Model | Mean AUROC | Peak AUROC | Peak Layer |
|-------|-----------|-----------|------------|
| Base | 0.898 | 0.941 | 23 |
| SFT | 0.898 | 0.942 | 23 |
| DPO | 0.889 | 0.940 | 23 |

Cross-model transfer (SFT probe on DPO): mean AUROC 0.880.
Direction similarity (SFT vs DPO): mean cosine 0.819.

## Bugs Found

### Bug 1: Wrong dtype parameter
`dtype=dtype` instead of `torch_dtype=dtype` in model loading. Silently ignored, model loads in config default.

### Bug 2: Wrong last-token position for left-padding
`attention_mask.sum(dim=1) - 1` computes right-padding formula. With left-padding, the last real token is always at `seq_len - 1`. This extracted activations from wrong positions when sequences had different lengths.

### Bug 3: Prompt leakage in train/val split
Individual samples (not prompt groups) were split. The honest version of prompt X could be in train while the sycophantic version was in val, allowing the probe to memorize prompt-specific features.

## The Fundamental Design Flaw

The probe was measuring **text comprehension** (can the model tell honest text from sycophantic text?) instead of **behavioral intent** (is the model about to be sycophantic?).

We fed pre-written honest and sycophantic responses through ALL models. Every model — including the base model — can distinguish these texts because they're linguistically different (agreeable tone, hedging, factual errors vs direct corrections). The probe detected these text features, not sycophancy tendencies.

Labels were identical across all models (determined by text content, not model behavior). A perfectly aligned model would score just as high as a sycophantic one.

## The Fix: Prompt-Only Probing

Instead of probing `prompt + response`, probe ONLY the prompt — the user's question with sycophantic pressure — BEFORE the model generates anything. Label by what the model ACTUALLY does (from existing judge evaluation results).

This way:
- SFT model: sycophantic intent should be detectable in pre-generation hidden states
- Base model: no systematic intent → probe should fail (~0.5 AUROC)
- DPO model: the question we're answering

## Learnings

1. **Linear probing methodology matters enormously.** The same technique with different data/labeling answers completely different questions. "Can the model detect sycophancy in text" vs "Is the model planning to be sycophantic" require fundamentally different experimental designs.

2. **Base model as sanity check.** If the base model scores high, something is wrong — either the probe is detecting text features (not behavior) or there's data leakage. Always check the control model first.

3. **Labels should come from model behavior, not text content.** Per-model labels from judge results are the correct approach for behavioral probing.

4. **Prompt-only extraction probes intent.** Extracting at the point of "about to generate" captures the model's decision state, not its text comprehension.
