# ABOUTME: Abstract base evaluator defining the interface for dataset-specific evaluators.
# ABOUTME: Each evaluator handles prompt building, generation dispatch, and judge prompt building.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEvaluator(ABC):
    """Abstract evaluator for a specific dataset type."""

    @abstractmethod
    def build_generation_prompts(self, data: list[dict]) -> list[dict]:
        """Build prompts for Pass 1 (subject model generation).

        Returns list of dicts with at minimum:
          - "messages": list of chat messages for the model
          - "idx": original row index
          - any extra metadata needed for judging
        """

    @abstractmethod
    def build_judge_prompts(self, generations: list[dict]) -> list[dict]:
        """Build prompts for Pass 2 (judge model scoring).

        Takes generation outputs and returns list of dicts:
          - "messages": list of chat messages for the judge
          - "idx": matching index from generation
          - "schema": Pydantic model class for guided JSON
        """

    @abstractmethod
    def compute_metrics(self, judgments: list[dict], generations: list[dict]) -> dict[str, Any]:
        """Compute dataset-specific metrics from judgments.

        Returns dict with metric names and values.
        """
