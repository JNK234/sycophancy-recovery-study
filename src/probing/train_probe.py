# ABOUTME: Train linear probes on hidden state activations and evaluate.
# ABOUTME: Supports per-model training, cross-model transfer, and probe direction analysis.

from __future__ import annotations

import os
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from src.probing.config import ProbeConfig, BootstrapConfig, ControlConfig


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
    if len(np.unique(labels)) < 2:
        raise ValueError(f"Cannot train probe: labels are single-class "
                         f"(unique={np.unique(labels)}). Check train/val split.")

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
    if len(np.unique(labels)) < 2:
        return {"accuracy": 0.5, "auroc": 0.5, "f1": 0.0, "n_samples": len(labels)}

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


def bootstrap_evaluate_probe(
    probe: LogisticRegression,
    activations: np.ndarray,
    labels: np.ndarray,
    config: BootstrapConfig,
) -> dict:
    """Evaluate probe with stratified bootstrap CIs.

    Uses stratified resampling (positive/negative separately) to avoid
    single-class resamples and produce tighter CIs.
    """
    if len(np.unique(labels)) < 2:
        return {"accuracy": 0.5, "auroc": 0.5, "auroc_ci_lower": 0.5,
                "auroc_ci_upper": 0.5, "f1": 0.0, "n_samples": len(labels),
                "n_bootstrap": 0}

    preds = probe.predict(activations)
    probs = probe.predict_proba(activations)[:, 1]

    observed_auroc = float(roc_auc_score(labels, probs))
    observed_acc = float(accuracy_score(labels, preds))
    observed_f1 = float(f1_score(labels, preds))

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    rng = np.random.RandomState(config.seed)

    boot_aurocs = []
    for _ in range(config.n_iterations):
        # Stratified: resample within each class
        boot_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        boot_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        boot_idx = np.concatenate([boot_pos, boot_neg])

        boot_labels = labels[boot_idx]
        boot_probs = probs[boot_idx]

        if len(np.unique(boot_labels)) < 2:
            continue
        boot_aurocs.append(roc_auc_score(boot_labels, boot_probs))

    alpha = 1.0 - config.confidence_level
    lower = float(np.percentile(boot_aurocs, 100 * alpha / 2))
    upper = float(np.percentile(boot_aurocs, 100 * (1 - alpha / 2)))

    return {
        "accuracy": observed_acc,
        "auroc": observed_auroc,
        "auroc_ci_lower": lower,
        "auroc_ci_upper": upper,
        "f1": observed_f1,
        "n_samples": len(labels),
        "n_bootstrap": len(boot_aurocs),
    }


def permutation_test_auroc(
    probe: LogisticRegression,
    activations: np.ndarray,
    labels: np.ndarray,
    n_permutations: int = 1000,
    seed: int = 42,
) -> dict:
    """Permutation test: is this probe's AUROC significantly above chance (0.5)?

    Fixes probs from trained probe, permutes labels n times, computes AUROC
    for each permutation. p-value = fraction of permuted AUROCs >= observed.
    """
    probs = probe.predict_proba(activations)[:, 1]
    observed = roc_auc_score(labels, probs)

    rng = np.random.RandomState(seed)
    count_ge = 0
    for _ in range(n_permutations):
        perm_labels = rng.permutation(labels)
        if len(np.unique(perm_labels)) < 2:
            continue
        perm_auroc = roc_auc_score(perm_labels, probs)
        if perm_auroc >= observed:
            count_ge += 1

    p_value = (count_ge + 1) / (n_permutations + 1)  # +1 for observed itself

    return {
        "auroc": float(observed),
        "p_value": float(p_value),
        "n_permutations": n_permutations,
    }


def train_control_probes(
    train_acts: dict[int, np.ndarray],
    train_labels: np.ndarray,
    val_acts: dict[int, np.ndarray],
    val_labels: np.ndarray,
    probe_config: ProbeConfig,
    n_seeds: int = 10,
) -> dict[int, dict]:
    """Train probes on shuffled labels as a noise-floor sanity check.

    If control probes get >0.55 AUROC, the real probes may be fitting noise.
    Returns {layer: {mean_control_auroc, std_control_auroc}}.
    """
    layers = sorted(train_acts.keys())
    layer_aurocs = {l: [] for l in layers}

    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        shuffled_labels = rng.permutation(train_labels)

        for layer_idx in layers:
            probe = LogisticRegression(
                max_iter=probe_config.max_iter,
                C=probe_config.C,
                solver="lbfgs",
                random_state=seed,
            )
            # Guard against single-class shuffled labels
            if len(np.unique(shuffled_labels)) < 2:
                layer_aurocs[layer_idx].append(0.5)
                continue

            probe.fit(train_acts[layer_idx], shuffled_labels)
            val_probs = probe.predict_proba(val_acts[layer_idx])[:, 1]

            if len(np.unique(val_labels)) < 2:
                layer_aurocs[layer_idx].append(0.5)
                continue

            auroc = roc_auc_score(val_labels, val_probs)
            layer_aurocs[layer_idx].append(auroc)

    results = {}
    warn_layers = []
    for layer_idx in layers:
        aurocs = layer_aurocs[layer_idx]
        mean_auroc = float(np.mean(aurocs))
        std_auroc = float(np.std(aurocs))
        results[layer_idx] = {
            "mean_control_auroc": mean_auroc,
            "std_control_auroc": std_auroc,
        }
        if mean_auroc > 0.55:
            warn_layers.append((layer_idx, mean_auroc))

    if warn_layers:
        print(f"  WARNING: {len(warn_layers)} layers have control AUROC > 0.55:")
        for l, a in warn_layers[:5]:
            print(f"    layer {l}: {a:.3f}")
        print("  Consider increasing regularization (lower C) or checking data quality.")

    return results


def max_statistic_permutation_test(
    probes: dict[int, LogisticRegression],
    activations: dict[int, np.ndarray],
    labels: np.ndarray,
    n_permutations: int = 500,
    seed: int = 42,
) -> dict:
    """Test if peak AUROC across layers is significant after multiple comparisons.

    Under label permutations, computes AUROC for every layer and takes the max.
    Compares observed peak to this null distribution of maxima.
    """
    layers = sorted(probes.keys())

    # Compute observed per-layer AUROCs
    observed_aurocs = {}
    for layer_idx in layers:
        probs = probes[layer_idx].predict_proba(activations[layer_idx])[:, 1]
        observed_aurocs[layer_idx] = roc_auc_score(labels, probs)

    observed_peak = max(observed_aurocs.values())
    observed_peak_layer = max(observed_aurocs, key=observed_aurocs.get)

    # Precompute all probs (avoid repeated predict_proba in loop)
    all_probs = {l: probes[l].predict_proba(activations[l])[:, 1] for l in layers}

    rng = np.random.RandomState(seed)
    null_peaks = []
    for _ in range(n_permutations):
        perm_labels = rng.permutation(labels)
        if len(np.unique(perm_labels)) < 2:
            continue
        perm_max = max(
            roc_auc_score(perm_labels, all_probs[l]) for l in layers
        )
        null_peaks.append(perm_max)

    count_ge = sum(1 for p in null_peaks if p >= observed_peak)
    corrected_p = float((count_ge + 1) / (len(null_peaks) + 1))

    return {
        "observed_peak": float(observed_peak),
        "observed_peak_layer": int(observed_peak_layer),
        "null_peak_mean": float(np.mean(null_peaks)),
        "null_peak_95th": float(np.percentile(null_peaks, 95)),
        "corrected_p_value": corrected_p,
        "n_permutations": len(null_peaks),
    }
