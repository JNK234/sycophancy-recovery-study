# ABOUTME: CLI entrypoint for running the sycophancy evaluation system.
# ABOUTME: Thin wrapper that parses args and delegates to the shared eval runner.

from __future__ import annotations

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.config import EvalConfig
from src.evaluation.runner import run_evaluation


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

    run_evaluation(
        config,
        skip_generation=args.skip_generation,
        skip_judge=args.skip_judge,
    )


if __name__ == "__main__":
    main()
