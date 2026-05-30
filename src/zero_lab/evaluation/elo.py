"""Approximate Elo confidence intervals for promotion reports."""

from __future__ import annotations

import math
from dataclasses import dataclass

_EPSILON = 1e-6
_Z_BY_CONFIDENCE = {
    0.90: 1.6448536269514722,
    0.95: 1.959963984540054,
    0.99: 2.5758293035489004,
}


@dataclass(frozen=True, slots=True)
class EloConfidenceInterval:
    games: int
    score_rate: float
    elo: float
    lower: float
    upper: float
    confidence_level: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "confidence_level": self.confidence_level,
            "elo": self.elo,
            "games": self.games,
            "lower": self.lower,
            "score_rate": self.score_rate,
            "upper": self.upper,
        }


def estimate_elo_confidence_interval(
    *,
    wins: int,
    losses: int,
    draws: int = 0,
    confidence_level: float = 0.95,
) -> EloConfidenceInterval:
    _validate_count("wins", wins)
    _validate_count("losses", losses)
    _validate_count("draws", draws)
    games = wins + losses + draws
    if games <= 0:
        raise ValueError("at least one game is required")
    if confidence_level not in _Z_BY_CONFIDENCE:
        raise ValueError("confidence_level must be one of 0.90, 0.95, or 0.99")

    score_rate = (wins + 0.5 * draws) / games
    lower_rate, upper_rate = _wilson_interval(
        score_rate=score_rate,
        games=games,
        z=_Z_BY_CONFIDENCE[confidence_level],
    )
    return EloConfidenceInterval(
        games=games,
        score_rate=score_rate,
        elo=_elo_from_score_rate(score_rate),
        lower=_elo_from_score_rate(lower_rate),
        upper=_elo_from_score_rate(upper_rate),
        confidence_level=confidence_level,
    )


def _validate_count(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _wilson_interval(*, score_rate: float, games: int, z: float) -> tuple[float, float]:
    z_squared = z * z
    denominator = 1.0 + z_squared / games
    center = (score_rate + z_squared / (2.0 * games)) / denominator
    margin = (
        z
        * math.sqrt(score_rate * (1.0 - score_rate) / games + z_squared / (4.0 * games * games))
        / denominator
    )
    return _clamp_rate(center - margin), _clamp_rate(center + margin)


def _elo_from_score_rate(score_rate: float) -> float:
    rate = _clamp_rate(score_rate)
    return 400.0 * math.log10(rate / (1.0 - rate))


def _clamp_rate(score_rate: float) -> float:
    return min(max(score_rate, _EPSILON), 1.0 - _EPSILON)
