#!/usr/bin/env python3
# ABOUTME: CLI entrypoint for running linear probing analysis.
# ABOUTME: Orchestrates: prompt loading -> activation extraction -> probe training -> analysis.

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.probing.config import ProbingConfig


def main():
    parser = argparse.ArgumentParser(
        description="Run linear probing analysis on model activations."
    )
    parser.add_argument("config", help="Path to probing config YAML")
    parser.add_argument(
        "--skip-extraction", action="store_true",
        help="Skip activation extraction, load saved .pt files",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Only process these model names (e.g., --models sft dpo)",
    )
    parser.add_argument(
        "--visualize-only", action="store_true",
        help="Only regenerate plots from saved results",
    )
    parser.add_argument(
        "--reuse-probes", action="store_true",
        help="Load existing probes from disk instead of retraining. "
             "Ensures frozen reference probe for consistent transfer numbers.",
    )
    args = parser.parse_args()

    config = ProbingConfig.from_yaml(args.config)

    if args.models:
        config.models = [m for m in config.models if m.name in args.models]
        print(f"Filtered to models: {[m.name for m in config.models]}")

    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    config.save_yaml(os.path.join(config.output_dir, "config.yaml"))

    if args.visualize_only:
        _visualize_only(config)
        return

    # ── Step 1: Load prompts + per-model behavior labels ──
    print("\n--- Step 1: Loading prompts and behavior labels ---")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.models[0].name_or_path,
        cache_dir=config.models[0].cache_dir,
    )

    from src.probing.data_prep import load_prompts_and_labels
    dataset = load_prompts_and_labels(config.data, config.models, tokenizer)

    # ── Step 2: Activation extraction ──
    if not args.skip_extraction:
        print("\n--- Step 2: Extracting activations ---")
        from src.probing.extraction import extract_and_save

        for model_entry in config.models:
            labels = dataset.model_labels[model_entry.name]
            extract_and_save(
                model_entry=model_entry,
                texts=dataset.all_prompts,
                labels=labels.tolist(),
                config=config.extraction,
                output_dir=os.path.join(config.output_dir, "activations"),
            )
    else:
        print("\n--- Step 2: Skipped (--skip-extraction) ---")

    # ── Step 3: Probe training + analysis ──
    print("\n--- Step 3: Training probes and running analysis ---")
    from src.probing.analysis import run_full_analysis, save_results, print_report

    results = run_full_analysis(config, dataset.train_indices, dataset.val_indices,
                               reuse_probes=args.reuse_probes)

    # ── Step 4: Save results ──
    print("\n--- Step 4: Saving results ---")
    save_results(results, config)

    # ── Step 5: Visualize ──
    print("\n--- Step 5: Generating plots ---")
    from src.probing.visualize import generate_all_plots
    generate_all_plots(results, config.results_dir)

    print_report(results, config.name)


def _visualize_only(config: ProbingConfig):
    """Regenerate plots from saved JSON results."""
    import json
    from src.probing.visualize import generate_all_plots

    results = {}
    for key in ["per_model", "cross_model_transfer", "direction_similarity"]:
        path = os.path.join(config.results_dir, f"{key}.json")
        with open(path) as f:
            results[key] = json.load(f)

    generate_all_plots(results, config.results_dir)
    print("Plots regenerated.")


if __name__ == "__main__":
    main()
