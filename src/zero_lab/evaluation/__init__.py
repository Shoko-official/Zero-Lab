"""Evaluation harness for fixed-seed AlphaZero baselines."""

from zero_lab.evaluation.agents import (
    EvaluationAgent,
    RandomLegalMoveAgent,
    UniformSearchAgent,
)
from zero_lab.evaluation.checkpoints import (
    AlphaZeroCheckpoint,
    CheckpointAgent,
    CheckpointComparison,
    compare_alpha_zero_checkpoints,
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
    "AlphaZeroCheckpoint",
    "CheckpointAgent",
    "CheckpointComparison",
    "EvaluationAgent",
    "EvaluationReport",
    "MatchConfig",
    "MatchResult",
    "MatchScore",
    "RandomLegalMoveAgent",
    "UniformSearchAgent",
    "compare_alpha_zero_checkpoints",
    "play_match",
    "run_head_to_head",
    "summarize_match_results",
]
