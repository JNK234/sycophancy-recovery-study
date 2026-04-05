#!/usr/bin/env python3
# ABOUTME: CLI entrypoint for training a reward model on DPO preference pairs.
# ABOUTME: Prerequisite for GRPO training — produces a scalar scorer for completions.

from __future__ import annotations

import argparse
import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from src.training.config_schema import ExperimentConfig
from src.training.reward_model import train_reward_model


def main():
    parser = argparse.ArgumentParser(description="Train reward model for GRPO")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    rm_path = train_reward_model(config)
    print(f"\nReward model saved to: {rm_path}")


if __name__ == "__main__":
    main()
