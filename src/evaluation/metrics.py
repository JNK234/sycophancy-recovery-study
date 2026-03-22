# ABOUTME: Aggregate metrics computation across all datasets.
# ABOUTME: Delegates to per-evaluator compute_metrics() and computes summary statistics.

from __future__ import annotations

import json
import os
from typing import Any

from src.evaluation import get_evaluator


def compute_all_metrics(
    eval_results: dict[str, dict],
    output_dir: str,
) -> dict[str, Any]:
    """Compute and save metrics for all evaluated datasets.

    Args:
        eval_results: Dict mapping dataset_type -> {generations, judgments}.
        output_dir: Directory to save metric JSON files.

    Returns:
        Summary dict with per-dataset and aggregate metrics.
    """
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    all_metrics = {}

    for dataset_type, data in eval_results.items():
        evaluator_cls = get_evaluator(dataset_type)
        evaluator = evaluator_cls()

        metrics = evaluator.compute_metrics(
            judgments=data["judgments"],
            generations=data["generations"],
        )
        all_metrics[dataset_type] = metrics

        # Save per-dataset metrics
        metrics_path = os.path.join(metrics_dir, f"{dataset_type}.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  [{dataset_type}] Metrics saved -> {metrics_path}")

    # Compute aggregate summary
    summary = _build_summary(all_metrics)
    summary_path = os.path.join(metrics_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved -> {summary_path}")

    return summary


def _build_summary(all_metrics: dict[str, Any]) -> dict[str, Any]:
    """Build aggregate summary from per-dataset metrics."""
    summary = {"per_dataset": {}}

    # Answer dataset headline metrics
    if "answer" in all_metrics:
        am = all_metrics["answer"]
        summary["per_dataset"]["answer"] = {
            "plain_accuracy": am.get("plain_accuracy", 0),
            "sycophancy_rate": am.get("sycophancy_rate", 0),
            "sycophancy_gap": am.get("sycophancy_gap", 0),
        }

    # Are-you-sure headline metrics
    if "are_you_sure" in all_metrics:
        ays = all_metrics["are_you_sure"]
        summary["per_dataset"]["are_you_sure"] = {
            "pass1_accuracy": ays.get("pass1_accuracy", 0),
            "flip_rate": ays.get("flip_rate", 0),
            "stubbornness_rate": ays.get("stubbornness_rate", 0),
        }

    # Feedback headline metrics
    if "feedback" in all_metrics:
        fm = all_metrics["feedback"]
        summary["per_dataset"]["feedback"] = {
            "overall_sycophancy_rate": fm.get("overall_sycophancy_rate", 0),
        }
        # Add per-sub-dataset sycophancy rates
        for sub_ds in ["math", "arguments", "poems"]:
            key = f"{sub_ds}/all"
            if key in fm:
                summary["per_dataset"]["feedback"][f"{sub_ds}_sycophancy_rate"] = (
                    fm[key].get("sycophancy_rate", 0)
                )

    # Aggregate sycophancy score (average across datasets)
    syc_rates = []
    if "answer" in summary["per_dataset"]:
        syc_rates.append(summary["per_dataset"]["answer"]["sycophancy_rate"])
    if "are_you_sure" in summary["per_dataset"]:
        syc_rates.append(summary["per_dataset"]["are_you_sure"]["flip_rate"])
    if "feedback" in summary["per_dataset"]:
        syc_rates.append(summary["per_dataset"]["feedback"]["overall_sycophancy_rate"])

    summary["aggregate_sycophancy"] = (
        sum(syc_rates) / len(syc_rates) if syc_rates else 0
    )

    # Full detailed metrics
    summary["detailed"] = all_metrics

    return summary
