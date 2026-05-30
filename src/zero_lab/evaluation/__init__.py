"""Evaluation harness for fixed-seed AlphaZero baselines."""

from zero_lab.evaluation.agents import (
    EvaluationAgent,
    RandomLegalMoveAgent,
    UniformSearchAgent,
)
from zero_lab.evaluation.matches import MatchConfig, MatchResult, play_match, run_head_to_head
from zero_lab.evaluation.reports import (
    DEFAULT_EVALUATION_LIMITATIONS,
    EvaluationReport,
    MatchScore,
    summarize_match_results,
)

__all__ = [
    "DEFAULT_EVALUATION_LIMITATIONS",
    "EvaluationAgent",
    "EvaluationReport",
    "MatchConfig",
    "MatchResult",
    "MatchScore",
    "RandomLegalMoveAgent",
    "UniformSearchAgent",
    "play_match",
    "run_head_to_head",
    "summarize_match_results",
]
