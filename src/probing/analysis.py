# ABOUTME: Orchestrate full probing analysis across multiple models.
# ABOUTME: Runs per-model probes, controls, cross-model transfer, direction similarity, and reporting.

from __future__ import annotations

import json
import os

import numpy as np

from src.probing.config import ProbingConfig
from src.probing.extraction import load_activations
from src.probing.train_probe import (
    train_probes_all_layers,
    evaluate_probe,
    bootstrap_evaluate_probe,
    permutation_test_auroc,
    train_control_probes,
    max_statistic_permutation_test,
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
    use_bootstrap = config.bootstrap.enabled

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

        # Load existing probes if reuse_probes and file exists.
        if reuse_probes and os.path.exists(probe_path):
            loaded = load_probes(probe_path)
            all_probes[model_name] = loaded
            print(f"  {model_name}: loaded frozen probe from {probe_path} "
                  f"({syc_rate:.1%} sycophantic)")

            val_metrics = {}
            for layer_idx, probe in loaded.items():
                if use_bootstrap:
                    val_result = bootstrap_evaluate_probe(
                        probe, val_acts[layer_idx], val_labels, config.bootstrap)
                else:
                    val_result = evaluate_probe(probe, val_acts[layer_idx], val_labels)
                val_metrics[layer_idx] = val_result
        else:
            print(f"  {model_name}: training with {len(train_labels)} samples "
                  f"({syc_rate:.1%} sycophantic)")

            probes = train_probes_all_layers(train_acts, train_labels, config.probe)
            all_probes[model_name] = {l: probe for l, (probe, _) in probes.items()}

            val_metrics = {}
            for layer_idx, (probe, _) in probes.items():
                if use_bootstrap:
                    val_result = bootstrap_evaluate_probe(
                        probe, val_acts[layer_idx], val_labels, config.bootstrap)
                else:
                    val_result = evaluate_probe(probe, val_acts[layer_idx], val_labels)
                val_metrics[layer_idx] = val_result

            save_probes(probes, probe_path)

        aurocs = [m["auroc"] for m in val_metrics.values()]
        peak_layer = max(val_metrics, key=lambda l: val_metrics[l]["auroc"])

        model_result = {
            "per_layer": {str(l): m for l, m in val_metrics.items()},
            "mean_auroc": float(np.mean(aurocs)),
            "peak_auroc": float(max(aurocs)),
            "peak_layer": int(peak_layer),
            "label_balance": float(syc_rate),
        }

        # Add peak analysis if bootstrap is enabled
        if use_bootstrap:
            # Count layers above chance (uncorrected — descriptive, not inferential)
            n_above = sum(
                1 for m in val_metrics.values()
                if m.get("auroc_ci_lower", 0) > 0.5
            )
            model_result["n_layers_above_chance"] = n_above
            model_result["peak_mean_delta"] = float(max(aurocs) - np.mean(aurocs))

        results["per_model"][model_name] = model_result

        ci_str = ""
        if use_bootstrap:
            peak_m = val_metrics[peak_layer]
            ci_str = f" [{peak_m.get('auroc_ci_lower', 0):.3f}, {peak_m.get('auroc_ci_upper', 0):.3f}]"

        print(f"  {model_name}: mean_auroc={np.mean(aurocs):.3f}, "
              f"peak_auroc={max(aurocs):.3f}{ci_str} (layer {peak_layer})")

    # ── Experiment 1.5: Random-label control ──
    if config.control.enabled:
        print("\n--- Experiment 1.5: Random-label control ---")
        ref_name = config.reference_model
        if ref_name in model_data:
            ref_data = model_data[ref_name]
            ref_acts = ref_data["activations"]
            ref_labels = ref_data["labels"]
            ref_train_acts = {l: a[train_indices] for l, a in ref_acts.items()}
            ref_val_acts = {l: a[val_indices] for l, a in ref_acts.items()}
            ref_train_labels = ref_labels[train_indices]
            ref_val_labels = ref_labels[val_indices]

            control_results = train_control_probes(
                ref_train_acts, ref_train_labels,
                ref_val_acts, ref_val_labels,
                config.probe, config.control.n_seeds,
            )

            # Compute aggregate stats
            mean_controls = [v["mean_control_auroc"] for v in control_results.values()]
            results["control"] = {
                "reference_model": ref_name,
                "per_layer": {str(l): v for l, v in control_results.items()},
                "overall_mean": float(np.mean(mean_controls)),
                "overall_std": float(np.std(mean_controls)),
                "n_seeds": config.control.n_seeds,
            }
            print(f"  Control AUROC (shuffled labels): "
                  f"mean={np.mean(mean_controls):.3f} +/- {np.std(mean_controls):.3f}")

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

            # Per-layer evaluation with optional bootstrap
            transfer_metrics = {}
            for layer_idx, probe in ref_probes.items():
                if layer_idx not in val_acts:
                    continue
                if use_bootstrap:
                    transfer_metrics[layer_idx] = bootstrap_evaluate_probe(
                        probe, val_acts[layer_idx], val_labels, config.bootstrap)
                else:
                    transfer_metrics[layer_idx] = evaluate_probe(
                        probe, val_acts[layer_idx], val_labels)

            aurocs = [m["auroc"] for m in transfer_metrics.values()]
            peak_layer = max(transfer_metrics, key=lambda l: transfer_metrics[l]["auroc"])

            key = f"{ref_name}_probe_on_{model_name}"
            transfer_result = {
                "per_layer": {str(l): m for l, m in transfer_metrics.items()},
                "mean_auroc": float(np.mean(aurocs)),
                "peak_auroc": float(max(aurocs)),
                "peak_layer": int(peak_layer),
            }

            # Permutation test on mean AUROC for transfer
            if use_bootstrap:
                # Run permutation test at peak layer for transfer
                perm_result = permutation_test_auroc(
                    ref_probes[peak_layer], val_acts[peak_layer],
                    val_labels, n_permutations=config.bootstrap.n_iterations,
                    seed=config.bootstrap.seed,
                )
                transfer_result["peak_p_value"] = perm_result["p_value"]

                # Max-statistic correction for peak across layers
                peak_correction = max_statistic_permutation_test(
                    ref_probes, val_acts, val_labels,
                    n_permutations=config.bootstrap.n_iterations,
                    seed=config.bootstrap.seed,
                )
                transfer_result["peak_corrected_p"] = peak_correction["corrected_p_value"]
                transfer_result["null_peak_95th"] = peak_correction["null_peak_95th"]

            results["cross_model_transfer"][key] = transfer_result

            ci_str = ""
            p_str = ""
            if use_bootstrap:
                peak_m = transfer_metrics[peak_layer]
                ci_str = f" [{peak_m.get('auroc_ci_lower', 0):.3f}, {peak_m.get('auroc_ci_upper', 0):.3f}]"
                p_str = f" p={transfer_result.get('peak_p_value', 'N/A'):.3f}"

            print(f"  {key}: mean_auroc={np.mean(aurocs):.3f}, "
                  f"peak_auroc={max(aurocs):.3f}{ci_str}{p_str} (layer {peak_layer})")

    # ── Experiment 3: Probe-space ablation ──
    print("\n--- Experiment 3: Probe-space ablation ---")
    from src.probing.ablation import (
        get_ablation_direction, probe_space_ablation, retrain_after_ablation
    )

    results["ablation"] = {}
    for model_name, data in model_data.items():
        if model_name not in all_probes:
            continue

        activations = data["activations"]
        labels = data["labels"]
        train_acts = {l: a[train_indices] for l, a in activations.items()}
        val_acts = {l: a[val_indices] for l, a in activations.items()}
        train_labels = labels[train_indices]
        val_labels = labels[val_indices]

        # Get this model's peak layer, validate probe exists
        peak_layer = results["per_model"][model_name]["peak_layer"]
        if peak_layer not in all_probes[model_name]:
            print(f"  {model_name}: skipping (no probe at peak layer {peak_layer})")
            continue
        probe = all_probes[model_name][peak_layer]
        direction = get_ablation_direction(probe)

        # Probe-space ablation (same probe, ablated activations)
        abl_result = probe_space_ablation(
            probe, val_acts[peak_layer], val_labels)

        # Retrain after ablation (fresh probe on ablated activations)
        retrain_result = retrain_after_ablation(
            train_acts[peak_layer], train_labels,
            direction, config.probe,
            val_acts[peak_layer], val_labels,
        )

        results["ablation"][model_name] = {
            "peak_layer": peak_layer,
            **abl_result,
            **retrain_result,
        }

        print(f"  {model_name} (L{peak_layer}): "
              f"orig={abl_result['auroc_original']:.3f} → "
              f"ablated={abl_result['auroc_ablated']:.3f} "
              f"(drop={abl_result['auroc_drop']:.3f}), "
              f"retrained={retrain_result['auroc_retrained']:.3f}")

    # ── Experiment 4: Direction similarity ──
    print("\n--- Experiment 4: Probe direction similarity ---")
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

    # Save all result sections (handles new sections like "control" generically)
    for key, data in results.items():
        path = os.path.join(config.results_dir, f"{key}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved: {path}")

    summary = _build_summary(results, config)
    path = os.path.join(config.results_dir, "summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {path}")

    config.save_yaml(os.path.join(config.results_dir, "config.yaml"))


def _build_summary(results: dict, config: ProbingConfig) -> dict:
    """Build headline summary from detailed results."""
    summary = {}

    # Per-model
    summary["per_model"] = {}
    for model_name, metrics in results["per_model"].items():
        entry = {
            "mean_auroc": metrics["mean_auroc"],
            "peak_auroc": metrics["peak_auroc"],
            "peak_layer": metrics["peak_layer"],
            "label_balance": metrics["label_balance"],
        }
        # Include CI and peak analysis if available
        if "n_layers_above_chance" in metrics:
            entry["n_layers_above_chance"] = metrics["n_layers_above_chance"]
            entry["peak_mean_delta"] = metrics["peak_mean_delta"]
        summary["per_model"][model_name] = entry

    # Cross-model transfer
    summary["cross_model_transfer"] = {}
    for key, metrics in results["cross_model_transfer"].items():
        entry = {
            "mean_auroc": metrics["mean_auroc"],
            "peak_auroc": metrics["peak_auroc"],
        }
        if "peak_p_value" in metrics:
            entry["peak_p_value"] = metrics["peak_p_value"]
        if "peak_corrected_p" in metrics:
            entry["peak_corrected_p"] = metrics["peak_corrected_p"]
        summary["cross_model_transfer"][key] = entry

    # Direction similarity
    summary["direction_similarity"] = {}
    for key, metrics in results["direction_similarity"].items():
        summary["direction_similarity"][key] = {
            "mean_cosine": metrics["mean_cosine"],
        }

    # Ablation (if present)
    if "ablation" in results:
        summary["ablation"] = {}
        for model_name, abl in results["ablation"].items():
            summary["ablation"][model_name] = {
                "peak_layer": abl["peak_layer"],
                "auroc_original": abl["auroc_original"],
                "auroc_ablated": abl["auroc_ablated"],
                "auroc_drop": abl["auroc_drop"],
                "auroc_retrained": abl["auroc_retrained"],
            }

    # Control (if present)
    if "control" in results:
        summary["control"] = {
            "reference_model": results["control"]["reference_model"],
            "overall_mean": results["control"]["overall_mean"],
            "overall_std": results["control"]["overall_std"],
        }

    return summary


def print_report(results: dict, config_name: str) -> None:
    """Print formatted probing report to console."""
    print(f"\n{'=' * 70}")
    print(f"  PROBING REPORT: {config_name}")
    print(f"{'=' * 70}")

    # Per-model probes
    print(f"\n  Per-Model Probes (own train/val):")
    for model_name, metrics in results["per_model"].items():
        ci_str = ""
        peak_layer_data = metrics.get("per_layer", {}).get(str(metrics["peak_layer"]), {})
        if "auroc_ci_lower" in peak_layer_data:
            ci_str = f" [{peak_layer_data['auroc_ci_lower']:.3f}, {peak_layer_data['auroc_ci_upper']:.3f}]"

        print(f"    {model_name:12s}: mean={metrics['mean_auroc']:.3f}  "
              f"peak={metrics['peak_auroc']:.3f}{ci_str} (L{metrics['peak_layer']})  "
              f"syc={metrics['label_balance']:.1%}")

    # Peak layer analysis
    has_peak_data = any("peak_mean_delta" in m for m in results["per_model"].values())
    if has_peak_data:
        print(f"\n  Peak Layer Analysis:")
        for model_name, metrics in results["per_model"].items():
            delta = metrics.get("peak_mean_delta", 0)
            n_above = metrics.get("n_layers_above_chance", "?")
            flag = " *** PEAK >> MEAN" if delta > 0.10 else ""
            print(f"    {model_name:12s}: peak-mean delta={delta:+.3f}  "
                  f"layers_above_chance={n_above}{flag}")

    # Control
    if "control" in results:
        ctrl = results["control"]
        print(f"\n  Random-Label Control ({ctrl.get('reference_model', '?')}):")
        print(f"    mean={ctrl['overall_mean']:.3f} +/- {ctrl['overall_std']:.3f}")
        if ctrl["overall_mean"] > 0.55:
            print(f"    WARNING: Control > 0.55 — probes may be fitting noise!")

    # Cross-model transfer
    print(f"\n  Cross-Model Transfer:")
    for key, metrics in results["cross_model_transfer"].items():
        p_str = ""
        if "peak_p_value" in metrics:
            p_str = f"  p={metrics['peak_p_value']:.3f}"
        corr_str = ""
        if "peak_corrected_p" in metrics:
            corr_str = f"  corrected_p={metrics['peak_corrected_p']:.3f}"
        print(f"    {key}: mean={metrics['mean_auroc']:.3f}  "
              f"peak={metrics['peak_auroc']:.3f}{p_str}{corr_str}")

    # Ablation
    if "ablation" in results:
        print(f"\n  Probe-Space Ablation (peak layer):")
        for model_name, abl in results["ablation"].items():
            print(f"    {model_name:12s} (L{abl['peak_layer']}): "
                  f"orig={abl['auroc_original']:.3f} → "
                  f"ablated={abl['auroc_ablated']:.3f} "
                  f"(drop={abl['auroc_drop']:.3f}), "
                  f"retrained={abl['auroc_retrained']:.3f}")

    # Direction similarity
    print(f"\n  Probe Direction Similarity:")
    for key, metrics in results["direction_similarity"].items():
        print(f"    {key}: mean_cosine={metrics['mean_cosine']:.3f}")

    print(f"{'=' * 70}")
