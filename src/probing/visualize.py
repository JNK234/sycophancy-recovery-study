# ABOUTME: Visualization for probing results.
# ABOUTME: Generates per-layer AUROC curves, cross-model heatmaps, and direction similarity plots.

from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_layer_auroc_curves(results: dict, output_dir: str) -> None:
    """Plot AUROC vs layer number for per-model probes and cross-model transfer."""
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Per-model probes
    for model_name, metrics in results["per_model"].items():
        layers = sorted(int(l) for l in metrics["per_layer"].keys())
        aurocs = [metrics["per_layer"][str(l)]["auroc"] for l in layers]
        ax1.plot(layers, aurocs, marker="o", markersize=3, label=model_name)

    ax1.set_xlabel("Layer")
    ax1.set_ylabel("AUROC")
    ax1.set_title("Per-Model Probe AUROC by Layer")
    ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="chance")
    ax1.legend()
    ax1.set_ylim(0.4, 1.05)
    ax1.grid(alpha=0.3)

    # Cross-model transfer
    for key, metrics in results["cross_model_transfer"].items():
        layers = sorted(int(l) for l in metrics["per_layer"].keys())
        aurocs = [metrics["per_layer"][str(l)]["auroc"] for l in layers]
        ax2.plot(layers, aurocs, marker="s", markersize=3, linestyle="--", label=key)

    ax2.set_xlabel("Layer")
    ax2.set_ylabel("AUROC")
    ax2.set_title("Cross-Model Transfer AUROC by Layer")
    ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="chance")
    ax2.legend()
    ax2.set_ylim(0.4, 1.05)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        path = os.path.join(output_dir, f"layer_auroc_curves.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir}/layer_auroc_curves.png")


def plot_direction_similarity(results: dict, output_dir: str) -> None:
    """Plot cosine similarity of probe directions across models per layer."""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    for key, metrics in results["direction_similarity"].items():
        layers = sorted(int(l) for l in metrics["per_layer"].keys())
        sims = [metrics["per_layer"][str(l)] for l in layers]
        ax.plot(layers, sims, marker="o", markersize=3, label=key)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Probe Weight Direction Similarity Across Models")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    if results["direction_similarity"]:
        ax.legend()
    ax.set_ylim(-0.2, 1.1)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        path = os.path.join(output_dir, f"probe_direction_similarity.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir}/probe_direction_similarity.png")


def generate_all_plots(results: dict, output_dir: str) -> None:
    """Generate all visualization plots."""
    plots_dir = os.path.join(output_dir, "plots")
    plot_layer_auroc_curves(results, plots_dir)
    plot_direction_similarity(results, plots_dir)
