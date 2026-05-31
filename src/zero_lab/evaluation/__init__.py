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
from zero_lab.evaluation.chess import (
    ChessGameRecord,
    ChessMatchConfig,
    ChessMoveRecord,
    run_chess_match,
    run_chess_matches,
)
from zero_lab.evaluation.elo import EloConfidenceInterval, estimate_elo_confidence_interval
from zero_lab.evaluation.matches import MatchConfig, MatchResult, play_match, run_head_to_head
from zero_lab.evaluation.promotion import (
    DEFAULT_PROMOTION_SEED_POLICY,
    PROMOTION_SCHEMA_VERSION,
    AlphaZeroPromotionReport,
    PromotionConfig,
    build_alpha_zero_promotion_report,
)
from zero_lab.evaluation.reports import (
    DEFAULT_EVALUATION_LIMITATIONS,
    EvaluationReport,
    MatchScore,
    summarize_match_results,
)

__all__ = [
    "DEFAULT_EVALUATION_LIMITATIONS",
    "DEFAULT_PROMOTION_SEED_POLICY",
    "PROMOTION_SCHEMA_VERSION",
    "AlphaZeroCheckpoint",
    "AlphaZeroPromotionReport",
    "ChessGameRecord",
    "ChessMatchConfig",
    "ChessMoveRecord",
    "CheckpointAgent",
    "CheckpointComparison",
    "EloConfidenceInterval",
    "EvaluationAgent",
    "EvaluationReport",
    "MatchConfig",
    "MatchResult",
    "MatchScore",
    "PromotionConfig",
    "RandomLegalMoveAgent",
    "UniformSearchAgent",
    "compare_alpha_zero_checkpoints",
    "build_alpha_zero_promotion_report",
    "estimate_elo_confidence_interval",
    "play_match",
    "run_chess_match",
    "run_chess_matches",
    "run_head_to_head",
    "summarize_match_results",
]
