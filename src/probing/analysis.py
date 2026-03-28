# ABOUTME: Orchestrate full probing analysis across multiple models.
# ABOUTME: Runs per-model probes, cross-model transfer, direction similarity, and reporting.

from __future__ import annotations

import json
import os

import numpy as np

from src.probing.config import ProbingConfig
from src.probing.extraction import load_activations
from src.probing.train_probe import (
    train_probes_all_layers,
    evaluate_probe,
    cross_model_evaluation,
    compute_direction_similarity,
    save_probes,
    load_probes,
)


def run_full_analysis(
    config: ProbingConfig,
    train_indices: list[int],
    val_indices: list[int],
    reuse_probes: bool = False,
) -> dict:
    """Run all probe experiments with per-model labels.

    Each model has its own labels (from judge verdicts), so probes are trained
    on each model's actual sycophantic behavior, not shared text labels.

    If reuse_probes=True, loads existing probe files from disk instead of
    retraining. This ensures consistent transfer numbers across runs — the
    reference probe stays frozen when evaluating new models.
    """
    act_dir = os.path.join(config.output_dir, "activations")

    # Load all activations
    model_data = {}
    for model_entry in config.models:
        path = os.path.join(act_dir, f"{model_entry.name}.pt")
        data = load_activations(path)
        model_data[model_entry.name] = data
        print(f"  Loaded activations: {model_entry.name} "
              f"({data['metadata']['num_samples']} samples, "
              f"{data['metadata']['num_layers']} layers)")

    results = {
        "per_model": {},
        "cross_model_transfer": {},
        "direction_similarity": {},
    }

    # ── Experiment 1: Per-model probes ──
    print("\n--- Experiment 1: Per-model probes ---")
    all_probes = {}

    for model_name, data in model_data.items():
        activations = data["activations"]
        labels = data["labels"]

        train_acts = {l: a[train_indices] for l, a in activations.items()}
        val_acts = {l: a[val_indices] for l, a in activations.items()}
        train_labels = labels[train_indices]
        val_labels = labels[val_indices]

        syc_rate = train_labels.mean()
        probe_path = os.path.join(config.output_dir, "probes", f"{model_name}_probes.pkl")

        # Load existing probes from disk if reuse_probes and file exists
        if reuse_probes and os.path.exists(probe_path):
            loaded = load_probes(probe_path)
            all_probes[model_name] = loaded
            print(f"  {model_name}: loaded frozen probe from {probe_path} "
                  f"({syc_rate:.1%} sycophantic)")

            val_metrics = {}
            for layer_idx, probe in loaded.items():
                val_result = evaluate_probe(probe, val_acts[layer_idx], val_labels)
                val_metrics[layer_idx] = val_result
        else:
            print(f"  {model_name}: training with {len(train_labels)} samples "
                  f"({syc_rate:.1%} sycophantic)")

            probes = train_probes_all_layers(train_acts, train_labels, config.probe)
            all_probes[model_name] = {l: probe for l, (probe, _) in probes.items()}

            val_metrics = {}
            for layer_idx, (probe, _) in probes.items():
                val_result = evaluate_probe(probe, val_acts[layer_idx], val_labels)
                val_metrics[layer_idx] = val_result

            save_probes(probes, probe_path)

        aurocs = [m["auroc"] for m in val_metrics.values()]
        peak_layer = max(val_metrics, key=lambda l: val_metrics[l]["auroc"])

        results["per_model"][model_name] = {
            "per_layer": {str(l): m for l, m in val_metrics.items()},
            "mean_auroc": float(np.mean(aurocs)),
            "peak_auroc": float(max(aurocs)),
            "peak_layer": int(peak_layer),
            "label_balance": float(syc_rate),
        }

        print(f"  {model_name}: mean_auroc={np.mean(aurocs):.3f}, "
              f"peak_auroc={max(aurocs):.3f} (layer {peak_layer})")

    # ── Experiment 2: Cross-model transfer ──
    print("\n--- Experiment 2: Cross-model transfer ---")
    ref_name = config.reference_model
    if ref_name in all_probes:
        ref_probes = all_probes[ref_name]

        for model_name, data in model_data.items():
            if model_name == ref_name:
                continue

            activations = data["activations"]
            labels = data["labels"]
            val_acts = {l: a[val_indices] for l, a in activations.items()}
            val_labels = labels[val_indices]

            transfer_metrics = cross_model_evaluation(ref_probes, val_acts, val_labels)

            aurocs = [m["auroc"] for m in transfer_metrics.values()]
            peak_layer = max(transfer_metrics, key=lambda l: transfer_metrics[l]["auroc"])

            key = f"{ref_name}_probe_on_{model_name}"
            results["cross_model_transfer"][key] = {
                "per_layer": {str(l): m for l, m in transfer_metrics.items()},
                "mean_auroc": float(np.mean(aurocs)),
                "peak_auroc": float(max(aurocs)),
                "peak_layer": int(peak_layer),
            }

            print(f"  {key}: mean_auroc={np.mean(aurocs):.3f}, "
                  f"peak_auroc={max(aurocs):.3f} (layer {peak_layer})")

    # ── Experiment 3: Direction similarity ──
    print("\n--- Experiment 3: Probe direction similarity ---")
    model_names = list(all_probes.keys())

    for i, name_a in enumerate(model_names):
        for name_b in model_names[i + 1:]:
            sims = compute_direction_similarity(all_probes[name_a], all_probes[name_b])
            key = f"{name_a}_vs_{name_b}"
            results["direction_similarity"][key] = {
                "per_layer": {str(l): float(s) for l, s in sims.items()},
                "mean_cosine": float(np.mean(list(sims.values()))),
            }
            print(f"  {key}: mean_cosine={np.mean(list(sims.values())):.3f}")

    return results


def save_results(results: dict, config: ProbingConfig) -> None:
    """Save results to git-tracked results directory."""
    os.makedirs(config.results_dir, exist_ok=True)

    for key in ["per_model", "cross_model_transfer", "direction_similarity"]:
        path = os.path.join(config.results_dir, f"{key}.json")
        with open(path, "w") as f:
            json.dump(results[key], f, indent=2)
        print(f"  Saved: {path}")

    summary = _build_summary(results, config)
    path = os.path.join(config.results_dir, "summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {path}")

    config.save_yaml(os.path.join(config.results_dir, "config.yaml"))


def _build_summary(results: dict, config: ProbingConfig) -> dict:
    """Build headline summary from detailed results."""
    summary = {"per_model": {}, "cross_model_transfer": {}, "direction_similarity": {}}

    for model_name, metrics in results["per_model"].items():
        summary["per_model"][model_name] = {
            "mean_auroc": metrics["mean_auroc"],
            "peak_auroc": metrics["peak_auroc"],
            "peak_layer": metrics["peak_layer"],
            "label_balance": metrics["label_balance"],
        }

    for key, metrics in results["cross_model_transfer"].items():
        summary["cross_model_transfer"][key] = {
            "mean_auroc": metrics["mean_auroc"],
            "peak_auroc": metrics["peak_auroc"],
        }

    for key, metrics in results["direction_similarity"].items():
        summary["direction_similarity"][key] = {
            "mean_cosine": metrics["mean_cosine"],
        }

    return summary


def print_report(results: dict, config_name: str) -> None:
    """Print formatted probing report to console."""
    print(f"\n{'=' * 70}")
    print(f"  PROBING REPORT: {config_name}")
    print(f"{'=' * 70}")

    print(f"\n  Per-Model Probes (own train/val):")
    for model_name, metrics in results["per_model"].items():
        print(f"    {model_name:12s}: mean_auroc={metrics['mean_auroc']:.3f}  "
              f"peak_auroc={metrics['peak_auroc']:.3f} (layer {metrics['peak_layer']})  "
              f"syc_rate={metrics['label_balance']:.1%}")

    print(f"\n  Cross-Model Transfer:")
    for key, metrics in results["cross_model_transfer"].items():
        print(f"    {key}: mean_auroc={metrics['mean_auroc']:.3f}  "
              f"peak_auroc={metrics['peak_auroc']:.3f}")

    print(f"\n  Probe Direction Similarity:")
    for key, metrics in results["direction_similarity"].items():
        print(f"    {key}: mean_cosine={metrics['mean_cosine']:.3f}")

    print(f"{'=' * 70}")
