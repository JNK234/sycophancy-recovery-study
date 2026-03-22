# ABOUTME: Print formatted evaluation summary and save final report JSON.
# ABOUTME: Provides human-readable console output of key metrics.

from __future__ import annotations

import json
import os
from typing import Any


def print_report(summary: dict[str, Any], config_name: str) -> None:
    """Print a formatted evaluation report to console."""
    print("\n" + "=" * 70)
    print(f"  EVALUATION REPORT: {config_name}")
    print("=" * 70)

    per_ds = summary.get("per_dataset", {})

    # Answer dataset
    if "answer" in per_ds:
        am = per_ds["answer"]
        print(f"\n  Answer Dataset:")
        print(f"    Plain accuracy:   {am['plain_accuracy']:.3f}")
        print(f"    Sycophancy rate:  {am['sycophancy_rate']:.3f}")
        print(f"    Sycophancy gap:   {am['sycophancy_gap']:.3f}")

    # Are-you-sure dataset
    if "are_you_sure" in per_ds:
        ays = per_ds["are_you_sure"]
        print(f"\n  Are-You-Sure Dataset:")
        print(f"    Pass 1 accuracy:  {ays['pass1_accuracy']:.3f}")
        print(f"    Flip rate:        {ays['flip_rate']:.3f}")
        print(f"    Stubbornness:     {ays['stubbornness_rate']:.3f}")

    # Feedback dataset
    if "feedback" in per_ds:
        fm = per_ds["feedback"]
        print(f"\n  Feedback Dataset:")
        print(f"    Overall syc rate: {fm['overall_sycophancy_rate']:.3f}")
        for sub_ds in ["math", "arguments", "poems"]:
            key = f"{sub_ds}_sycophancy_rate"
            if key in fm:
                print(f"    {sub_ds:12s} syc:  {fm[key]:.3f}")

    # Aggregate
    agg = summary.get("aggregate_sycophancy", 0)
    print(f"\n  Aggregate Sycophancy Score: {agg:.3f}")
    print("=" * 70 + "\n")


def save_report(summary: dict[str, Any], output_dir: str) -> str:
    """Save the full summary report as JSON."""
    report_path = os.path.join(output_dir, "metrics", "summary.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    return report_path
