# ABOUTME: Train linear probes on hidden state activations and evaluate.
# ABOUTME: Supports per-model training, cross-model transfer, and probe direction analysis.

from __future__ import annotations

import os
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from src.probing.config import ProbeConfig


def train_linear_probe(
    activations: np.ndarray,
    labels: np.ndarray,
    config: ProbeConfig,
) -> tuple[LogisticRegression, dict]:
    """Train a logistic regression probe on activations for one layer.

    Args:
        activations: shape [N, hidden_dim]
        labels: shape [N], binary (0=honest, 1=sycophantic)

    Returns (fitted_probe, metrics_dict).
    """
    probe = LogisticRegression(
        max_iter=config.max_iter,
        C=config.C,
        solver="lbfgs",
        random_state=42,
    )
    probe.fit(activations, labels)

    preds = probe.predict(activations)
    probs = probe.predict_proba(activations)[:, 1]

    metrics = {
        "train_accuracy": accuracy_score(labels, preds),
        "train_auroc": roc_auc_score(labels, probs),
    }
    return probe, metrics


def evaluate_probe(
    probe: LogisticRegression,
    activations: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """Evaluate a fitted probe on given activations."""
    preds = probe.predict(activations)
    probs = probe.predict_proba(activations)[:, 1]

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "auroc": float(roc_auc_score(labels, probs)),
        "f1": float(f1_score(labels, preds)),
        "n_samples": len(labels),
    }


def train_probes_all_layers(
    activations: dict[int, np.ndarray],
    labels: np.ndarray,
    config: ProbeConfig,
) -> dict[int, tuple[LogisticRegression, dict]]:
    """Train separate probe per layer. Returns {layer: (probe, metrics)}."""
    results = {}
    for layer_idx in sorted(activations.keys()):
        probe, metrics = train_linear_probe(activations[layer_idx], labels, config)
        results[layer_idx] = (probe, metrics)
    return results


def cross_model_evaluation(
    source_probes: dict[int, LogisticRegression],
    target_activations: dict[int, np.ndarray],
    target_labels: np.ndarray,
) -> dict[int, dict]:
    """Apply probes from one model to another model's activations.

    This is THE key analysis: if SFT probes classify DPO activations
    with high AUROC, sycophancy is still encoded internally.
    """
    results = {}
    for layer_idx, probe in source_probes.items():
        if layer_idx not in target_activations:
            continue
        results[layer_idx] = evaluate_probe(
            probe, target_activations[layer_idx], target_labels,
        )
    return results


def compute_direction_similarity(
    probes_a: dict[int, LogisticRegression],
    probes_b: dict[int, LogisticRegression],
) -> dict[int, float]:
    """Cosine similarity between probe weight vectors across two models.

    If weights are similar, both models encode sycophancy in the same direction.
    If weights differ, one model reorganized its representations.
    """
    similarities = {}
    for layer_idx in sorted(probes_a.keys()):
        if layer_idx not in probes_b:
            continue
        w_a = probes_a[layer_idx].coef_[0]
        w_b = probes_b[layer_idx].coef_[0]
        cos_sim = float(np.dot(w_a, w_b) / (np.linalg.norm(w_a) * np.linalg.norm(w_b) + 1e-8))
        similarities[layer_idx] = cos_sim
    return similarities


def save_probes(
    probes: dict[int, tuple[LogisticRegression, dict]],
    path: str,
) -> None:
    """Save fitted probes to pickle file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Extract just the probe objects
    probe_dict = {layer: probe for layer, (probe, _) in probes.items()}
    with open(path, "wb") as f:
        pickle.dump(probe_dict, f)


def load_probes(path: str) -> dict[int, LogisticRegression]:
    """Load saved probes from pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)
