"""Promotion report schema for AlphaZero checkpoint comparisons."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from zero_lab.evaluation.checkpoints import CheckpointComparison
from zero_lab.evaluation.elo import EloConfidenceInterval, estimate_elo_confidence_interval
from zero_lab.evaluation.matches import MatchConfig
from zero_lab.evaluation.reports import EvaluationReport, summarize_match_results

PROMOTION_SCHEMA_VERSION = 1
DEFAULT_PROMOTION_SEED_POLICY = "sequential_match_seed"


@dataclass(frozen=True, slots=True)
class PromotionConfig:
    match_config: MatchConfig = field(default_factory=MatchConfig)
    confidence_level: float = 0.95
    promotion_elo_threshold: float = 0.0
    seed_policy: str = DEFAULT_PROMOTION_SEED_POLICY

    def __post_init__(self) -> None:
        if not self.seed_policy:
            raise ValueError("seed_policy must not be empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "confidence_level": self.confidence_level,
            "match": self.match_config.to_dict(),
            "promotion_elo_threshold": self.promotion_elo_threshold,
            "seed_policy": self.seed_policy,
        }


@dataclass(frozen=True, slots=True)
class AlphaZeroPromotionReport:
    champion_candidate: CheckpointComparison
    config: PromotionConfig
    match_report: EvaluationReport
    candidate_elo_interval: EloConfidenceInterval
    schema_version: int = PROMOTION_SCHEMA_VERSION

    @property
    def decision(self) -> str:
        if self.candidate_elo_interval.lower >= self.config.promotion_elo_threshold:
            return "promote"
        return "hold"

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate": self.champion_candidate.candidate.to_dict(),
            "candidate_elo_confidence_interval": self.candidate_elo_interval.to_dict(),
            "champion": self.champion_candidate.champion.to_dict(),
            "config": self.config.to_dict(),
            "promotion": {
                "decision": self.decision,
                "elo_threshold": self.config.promotion_elo_threshold,
            },
            "results": self.match_report.to_dict(),
            "schema_version": self.schema_version,
            "seed_policy": {
                "base_seed": self.config.match_config.seed,
                "description": "match seed = base seed + match index",
                "name": self.config.seed_policy,
            },
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def build_alpha_zero_promotion_report(
    comparison: CheckpointComparison,
    *,
    config: PromotionConfig,
) -> AlphaZeroPromotionReport:
    match_report = summarize_match_results(
        comparison.results,
        config=config.match_config,
    )
    candidate_score = match_report.scores[comparison.candidate.name]
    candidate_interval = estimate_elo_confidence_interval(
        wins=candidate_score.wins,
        losses=candidate_score.losses,
        draws=candidate_score.draws,
        confidence_level=config.confidence_level,
    )
    return AlphaZeroPromotionReport(
        champion_candidate=comparison,
        config=config,
        match_report=match_report,
        candidate_elo_interval=candidate_interval,
    )
