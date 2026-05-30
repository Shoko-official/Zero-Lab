"""JSON-ready summaries for evaluation matches."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from zero_lab.evaluation.matches import MatchConfig, MatchResult

DEFAULT_EVALUATION_LIMITATIONS = (
    "Elo ratings are intentionally out of scope for this report.",
    "SPRT is intentionally out of scope for this report.",
    "Results cover fixed-seed baseline matches only.",
)


@dataclass(frozen=True, slots=True)
class MatchScore:
    wins: int = 0
    losses: int = 0
    draws: int = 0
    unfinished: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "draws": self.draws,
            "losses": self.losses,
            "unfinished": self.unfinished,
            "wins": self.wins,
        }


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    config: MatchConfig
    games: tuple[str, ...]
    seeds: tuple[int, ...]
    scores: dict[str, MatchScore]
    matches: tuple[MatchResult, ...]
    limitations: tuple[str, ...] = DEFAULT_EVALUATION_LIMITATIONS

    def to_dict(self) -> dict[str, object]:
        return {
            "config": self.config.to_dict(),
            "games": list(self.games),
            "limitations": list(self.limitations),
            "matches": [match.to_dict() for match in self.matches],
            "scores": {
                agent: score.to_dict()
                for agent, score in sorted(self.scores.items())
            },
            "seeds": list(self.seeds),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def summarize_match_results(
    results: Sequence[MatchResult],
    *,
    config: MatchConfig,
    limitations: Sequence[str] = DEFAULT_EVALUATION_LIMITATIONS,
) -> EvaluationReport:
    if not results:
        raise ValueError("results must not be empty")

    score_values = {
        agent: {"draws": 0, "losses": 0, "unfinished": 0, "wins": 0}
        for agent in _agent_names(results)
    }
    for result in results:
        first_score = score_values[result.first_agent]
        second_score = score_values[result.second_agent]
        if not result.terminal:
            first_score["unfinished"] += 1
            second_score["unfinished"] += 1
        elif result.outcome_for_first == 0:
            first_score["draws"] += 1
            second_score["draws"] += 1
        elif result.outcome_for_first == 1:
            first_score["wins"] += 1
            second_score["losses"] += 1
        elif result.outcome_for_first == -1:
            first_score["losses"] += 1
            second_score["wins"] += 1
        else:
            raise ValueError("terminal result must have an outcome")

    scores = {
        agent: MatchScore(
            draws=values["draws"],
            losses=values["losses"],
            unfinished=values["unfinished"],
            wins=values["wins"],
        )
        for agent, values in score_values.items()
    }
    return EvaluationReport(
        config=config,
        games=tuple(dict.fromkeys(result.game for result in results)),
        seeds=tuple(result.seed for result in results),
        scores=scores,
        matches=tuple(results),
        limitations=tuple(limitations),
    )


def _agent_names(results: Sequence[MatchResult]) -> tuple[str, ...]:
    names: list[str] = []
    for result in results:
        if result.first_agent not in names:
            names.append(result.first_agent)
        if result.second_agent not in names:
            names.append(result.second_agent)
    return tuple(names)
