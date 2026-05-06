#!/usr/bin/env python3
# ABOUTME: CLI entrypoint for the Self-Refine pretest diagnostic.
# ABOUTME: Tests whether M_syc (SFT-sycophantic) can self-critique and revise its own sycophancy.

from __future__ import annotations

import argparse
import os
import sys

# Add project root to path so `src.*` imports resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generation.self_refine_pretest import run_pretest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Self-Refine pretest: can M_syc self-critique its own sycophancy?"
    )
    parser.add_argument(
        "--n-per-dataset",
        type=int,
        default=25,
        help="Number of prompts to sample from each of answer.jsonl and feedback.jsonl (default: 25 -> 50 total)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling and principle selection",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/self_refine_pretest",
        help="Where to save artifacts (subject_outputs.jsonl, judgments_*.jsonl, summary.json)",
    )
    parser.add_argument(
        "--answer-path",
        type=str,
        default="evals/sycophancy-eval/datasets/answer.jsonl",
    )
    parser.add_argument(
        "--feedback-path",
        type=str,
        default="evals/sycophancy-eval/datasets/feedback.jsonl",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/scratch/wnn7240/huggingface_cache",
    )
    args = parser.parse_args()

    run_pretest(
        n_per_dataset=args.n_per_dataset,
        seed=args.seed,
        output_dir=args.output_dir,
        answer_path=args.answer_path,
        feedback_path=args.feedback_path,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
