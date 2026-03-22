#!/usr/bin/env python3
# ABOUTME: CLI entrypoint for running training experiments.
# ABOUTME: Reads YAML config and dispatches to the appropriate trainer class.

from __future__ import annotations

import argparse
import importlib
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.config_schema import ExperimentConfig


TRAINERS = {
    "sft": "src.training.sft_trainer.SFTSycophancyTrainer",
    "dpo": "src.training.dpo_trainer.DPORecoveryTrainer",
}


def _import_trainer(dotted_path: str):
    """Import trainer class from dotted module path."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def main():
    parser = argparse.ArgumentParser(
        description="Run a training experiment from a YAML config."
    )
    parser.add_argument(
        "--config", required=True, help="Path to experiment YAML config"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation on an existing merged model",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Only merge an existing LoRA adapter",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from a checkpoint directory (e.g., /path/to/checkpoint-147)",
    )
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)

    method = config.experiment.method
    if method not in TRAINERS:
        print(f"Error: Unknown method '{method}'. Available: {list(TRAINERS.keys())}")
        sys.exit(1)

    trainer_cls = _import_trainer(TRAINERS[method])
    trainer = trainer_cls(config)

    if args.eval_only:
        trainer.evaluate()
    elif args.merge_only:
        trainer.merge()
    else:
        trainer.run(resume_from_checkpoint=args.resume)


if __name__ == "__main__":
    main()
