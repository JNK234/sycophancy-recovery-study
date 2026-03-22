# ABOUTME: Custom training callbacks for experiment tracking.
# ABOUTME: Saves experiment config YAML alongside checkpoints for reproducibility.

from __future__ import annotations

import os
from transformers import TrainerCallback

from src.training.config_schema import ExperimentConfig


class ConfigSaveCallback(TrainerCallback):
    """Saves the experiment config YAML to the output directory at training start."""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def on_train_begin(self, args, state, control, **kwargs):
        config_path = os.path.join(args.output_dir, "experiment_config.yaml")
        self.config.save_yaml(config_path)
