# ABOUTME: Directional ablation for probing analysis.
# ABOUTME: Projects out probe-learned sycophancy direction from activations to test causality.

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from src.probing.config import ProbeConfig


def get_ablation_direction(probe: LogisticRegression, normalize: bool = True) -> np.ndarray:
    """Extract the sycophancy direction from a fitted probe.

    The probe's weight vector defines the linear direction that best separates
    sycophantic from honest activations. This is the direction to ablate.
    """
    direction = probe.coef_[0].copy()
    if normalize:
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            direction = direction / norm
    return direction


def project_out_direction(
    activations: np.ndarray,
    direction: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """Project out a direction from activations.

    h_ablated = h - alpha * (h . d_hat) * d_hat

    alpha=1.0: full removal (orthogonal projection)
    alpha=0.0: no change
    alpha>1.0: over-ablation (pushes past orthogonal)
    alpha<0.0: steering (adds the direction instead of removing)
    """
    # direction should be normalized
    d_hat = direction / (np.linalg.norm(direction) + 1e-8)
    projections = activations @ d_hat  # [N]
    ablated = activations - alpha * projections[:, np.newaxis] * d_hat[np.newaxis, :]
    return ablated


def probe_space_ablation(
    probe: LogisticRegression,
    activations: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """Ablate probe direction from activations and re-evaluate with same probe.

    This is near-tautological (the probe defined the direction we're removing),
    but confirms the probe relies on this specific direction. If AUROC drops
    to ~0.5, the direction carried all of the probe's signal.
    """
    direction = get_ablation_direction(probe)

    # Original evaluation
    probs_orig = probe.predict_proba(activations)[:, 1]
    auroc_orig = roc_auc_score(labels, probs_orig)

    # Ablated evaluation
    ablated_acts = project_out_direction(activations, direction, alpha=1.0)
    probs_ablated = probe.predict_proba(ablated_acts)[:, 1]

    # Handle degenerate case: if all probs are identical after ablation,
    # roc_auc_score returns 1.0 (ties), but the probe has zero discriminability.
    if np.std(probs_ablated) < 1e-6:
        auroc_ablated = 0.5
    else:
        auroc_ablated = roc_auc_score(labels, probs_ablated)

    return {
        "auroc_original": float(auroc_orig),
        "auroc_ablated": float(auroc_ablated),
        "auroc_drop": float(auroc_orig - auroc_ablated),
        "direction_norm": float(np.linalg.norm(probe.coef_[0])),
    }


def retrain_after_ablation(
    activations: np.ndarray,
    labels: np.ndarray,
    direction: np.ndarray,
    probe_config: ProbeConfig,
    val_activations: np.ndarray,
    val_labels: np.ndarray,
) -> dict:
    """Train a fresh probe on ablated activations.

    This is the real test: if a fresh probe trained on ablated activations still
    gets high AUROC, there's remaining sycophancy signal outside the ablated
    direction. If it drops to chance, the direction was the only linear signal.
    """
    ablated_train = project_out_direction(activations, direction, alpha=1.0)
    ablated_val = project_out_direction(val_activations, direction, alpha=1.0)

    # Guard against degenerate cases
    if len(np.unique(labels)) < 2 or len(np.unique(val_labels)) < 2:
        return {"auroc_retrained": 0.5, "note": "single-class labels"}

    fresh_probe = LogisticRegression(
        max_iter=probe_config.max_iter,
        C=probe_config.C,
        solver="lbfgs",
        random_state=42,
    )
    fresh_probe.fit(ablated_train, labels)

    val_probs = fresh_probe.predict_proba(ablated_val)[:, 1]
    auroc = roc_auc_score(val_labels, val_probs)
    acc = accuracy_score(val_labels, fresh_probe.predict(ablated_val))

    return {
        "auroc_retrained": float(auroc),
        "accuracy_retrained": float(acc),
    }
