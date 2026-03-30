# ABOUTME: Shared eval orchestration used by both CLI and post-training eval.
# ABOUTME: Single source of truth for the 2-pass eval pipeline (generate + judge).

from __future__ import annotations

import json
import os
from typing import Optional

from src.evaluation.config import EvalConfig
from src.evaluation.datasets import load_eval_dataset
from src.evaluation import get_evaluator
from src.evaluation.generate import run_generation_pass, run_challenge_generation
from src.evaluation.judge import run_judge_pass
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.report import print_report


def run_evaluation(
    config: EvalConfig,
    skip_generation: bool = False,
    skip_judge: bool = False,
) -> dict:
    """Run the full 2-pass evaluation pipeline.

    Args:
        config: Evaluation configuration.
        skip_generation: If True, load existing generations from disk.
        skip_judge: If True, load existing judgments from disk.

    Returns:
        Summary metrics dict from compute_all_metrics.
    """
    os.makedirs(config.output_dir, exist_ok=True)
    config.save_yaml(os.path.join(config.output_dir, "config.yaml"))

    eval_results = {}
    subject_llm = None
    judge_llm = None

    # Load all datasets upfront (needed for judge prompts even when skipping generation)
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
    if not skip_generation:
        print("\n--- Pass 1: Generating responses ---")
        for ds_entry in config.datasets:
            data = dataset_cache[ds_entry.type]
            evaluator = get_evaluator(ds_entry.type)()
            prompts = evaluator.build_generation_prompts(data)

            gen_path = os.path.join(config.output_dir, "generations", f"{ds_entry.type}.jsonl")
            generations, subject_llm = run_generation_pass(
                config, ds_entry.type, prompts, gen_path, llm=subject_llm,
            )

            if ds_entry.type == "are_you_sure":
                challenge_prompts = evaluator.build_challenge_prompts(generations)
                challenge_gens = run_challenge_generation(
                    config, challenge_prompts, gen_path, subject_llm,
                )
                generations = generations + challenge_gens

            eval_results[ds_entry.type] = {"generations": generations, "data": data}

        del subject_llm
        subject_llm = None
        _force_gpu_cleanup()
    else:
        print("\n--- Skipping Pass 1: Loading saved generations ---")
        for ds_entry in config.datasets:
            data = dataset_cache[ds_entry.type]
            gen_path = os.path.join(config.output_dir, "generations", f"{ds_entry.type}.jsonl")
            generations = _load_generations_with_base(gen_path, data)
            print(f"  [{ds_entry.type}] Loaded {len(generations)} saved generations")
            eval_results[ds_entry.type] = {"generations": generations, "data": data}

    # === PASS 2: Judge scoring ===
    if not skip_judge:
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

    return summary


def _load_generations_with_base(gen_path: str, data: list[dict]) -> list[dict]:
    """Load saved generation JSONL and merge back the 'base' field from original data.

    Uses prompt_id for matching when available, falls back to idx for old artifacts.
    """
    id_map = {row["prompt_id"]: row for row in data if row.get("prompt_id")}
    idx_map = {i: row for i, row in enumerate(data)}

    generations = []
    with open(gen_path) as f:
        for line in f:
            if not line.strip():
                continue
            gen = json.loads(line)
            source_row = None
            if gen.get("prompt_id") and gen["prompt_id"] in id_map:
                source_row = id_map[gen["prompt_id"]]
            elif gen["idx"] in idx_map:
                source_row = idx_map[gen["idx"]]

            if source_row:
                gen["base"] = source_row.get("base", {})
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
