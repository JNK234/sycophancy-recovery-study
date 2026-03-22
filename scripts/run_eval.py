# ABOUTME: CLI entrypoint for running the sycophancy evaluation system.
# ABOUTME: Orchestrates: load config -> generate responses -> judge -> metrics -> report.

from __future__ import annotations

import argparse
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.config import EvalConfig
from evaluation.datasets import load_eval_dataset
from evaluation import get_evaluator
from evaluation.generate import run_generation_pass, run_challenge_generation
from evaluation.judge import run_judge_pass
from evaluation.metrics import compute_all_metrics
from evaluation.report import print_report


def main():
    parser = argparse.ArgumentParser(description="Run sycophancy evaluation")
    parser.add_argument("config", help="Path to eval config YAML")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip Pass 1, load existing generations")
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip Pass 2, load existing judgments")
    args = parser.parse_args()

    config = EvalConfig.from_yaml(args.config)
    print(f"Evaluation: {config.name}")
    print(f"Output: {config.output_dir}")

    # Save config copy
    os.makedirs(config.output_dir, exist_ok=True)
    config.save_yaml(os.path.join(config.output_dir, "config.yaml"))

    eval_results = {}
    subject_llm = None
    judge_llm = None

    # Always load datasets (needed for judge prompts even when skipping generation)
    dataset_cache = {}
    for ds_entry in config.datasets:
        data = load_eval_dataset(
            path=ds_entry.path,
            max_samples=ds_entry.max_samples,
            seen_question_file=ds_entry.seen_question_file,
            seed=config.seed,
        )
        dataset_cache[ds_entry.type] = data
        print(f"  [{ds_entry.type}] Loaded {len(data)} samples "
              f"({sum(1 for r in data if r.get('seen'))} seen)")

    # === PASS 1: Generation ===
    if not args.skip_generation:
        print("\n--- Pass 1: Generating responses ---")
        for ds_entry in config.datasets:
            data = dataset_cache[ds_entry.type]
            evaluator = get_evaluator(ds_entry.type)()
            prompts = evaluator.build_generation_prompts(data)

            gen_path = os.path.join(config.output_dir, "generations", f"{ds_entry.type}.jsonl")
            generations, subject_llm = run_generation_pass(
                config, ds_entry.type, prompts, gen_path, llm=subject_llm,
            )

            # Handle are_you_sure 2-pass
            if ds_entry.type == "are_you_sure":
                challenge_prompts = evaluator.build_challenge_prompts(generations)
                challenge_gens = run_challenge_generation(
                    config, challenge_prompts, gen_path, subject_llm,
                )
                generations = generations + challenge_gens

            eval_results[ds_entry.type] = {"generations": generations, "data": data}

        # Free subject model GPU memory
        del subject_llm
        subject_llm = None
        _force_gpu_cleanup()
    else:
        # Reload generations from saved JSONL and merge with original data
        print("\n--- Skipping Pass 1: Loading saved generations ---")
        for ds_entry in config.datasets:
            data = dataset_cache[ds_entry.type]
            gen_path = os.path.join(config.output_dir, "generations", f"{ds_entry.type}.jsonl")
            generations = _load_generations_with_base(gen_path, data)
            print(f"  [{ds_entry.type}] Loaded {len(generations)} saved generations")
            eval_results[ds_entry.type] = {"generations": generations, "data": data}

    # === PASS 2: Judge scoring ===
    if not args.skip_judge:
        print("\n--- Pass 2: Judge scoring ---")
        for ds_entry in config.datasets:
            if ds_entry.type not in eval_results:
                continue

            evaluator = get_evaluator(ds_entry.type)()
            generations = eval_results[ds_entry.type]["generations"]
            judge_prompts = evaluator.build_judge_prompts(generations)

            judge_path = os.path.join(config.output_dir, "judgments", f"{ds_entry.type}.jsonl")
            judgments, judge_llm = run_judge_pass(
                config, ds_entry.type, judge_prompts, judge_path, llm=judge_llm,
            )
            eval_results[ds_entry.type]["judgments"] = judgments

        del judge_llm
        judge_llm = None
        _force_gpu_cleanup()
    else:
        # Reload judgments from saved JSONL
        print("\n--- Skipping Pass 2: Loading saved judgments ---")
        for ds_entry in config.datasets:
            if ds_entry.type not in eval_results:
                continue
            judge_path = os.path.join(config.output_dir, "judgments", f"{ds_entry.type}.jsonl")
            judgments = _load_jsonl(judge_path)
            print(f"  [{ds_entry.type}] Loaded {len(judgments)} saved judgments")
            eval_results[ds_entry.type]["judgments"] = judgments

    # === Metrics & Report ===
    print("\n--- Computing metrics ---")
    summary = compute_all_metrics(eval_results, config.output_dir)
    print_report(summary, config.name)


def _load_generations_with_base(gen_path: str, data: list[dict]) -> list[dict]:
    """Load saved generation JSONL and merge back the 'base' field from original data."""
    data_map = {i: row for i, row in enumerate(data)}
    generations = []
    with open(gen_path) as f:
        for line in f:
            if not line.strip():
                continue
            gen = json.loads(line)
            idx = gen["idx"]
            # Merge base data from original dataset
            if idx in data_map:
                gen["base"] = data_map[idx].get("base", {})
            # Reconstruct fields needed by evaluators
            if "template_type" not in gen:
                gen["template_type"] = ""
            generations.append(gen)
    return generations


def _load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _force_gpu_cleanup():
    """Force GPU memory cleanup between passes."""
    try:
        import torch
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


if __name__ == "__main__":
    main()
