# ABOUTME: Package init for evaluation system with evaluator registry.
# ABOUTME: Provides factory function to get dataset-specific evaluators.

from evaluation.evaluators.answer_evaluator import AnswerEvaluator
from evaluation.evaluators.are_you_sure_evaluator import AreYouSureEvaluator
from evaluation.evaluators.feedback_evaluator import FeedbackEvaluator

EVALUATOR_REGISTRY = {
    "answer": AnswerEvaluator,
    "are_you_sure": AreYouSureEvaluator,
    "feedback": FeedbackEvaluator,
}


def get_evaluator(dataset_type: str):
    """Return evaluator class for the given dataset type."""
    if dataset_type not in EVALUATOR_REGISTRY:
        raise ValueError(
            f"Unknown dataset type '{dataset_type}'. "
            f"Must be one of {list(EVALUATOR_REGISTRY.keys())}"
        )
    return EVALUATOR_REGISTRY[dataset_type]
