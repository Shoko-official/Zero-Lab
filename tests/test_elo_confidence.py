from __future__ import annotations

import pytest

from zero_lab.evaluation import estimate_elo_confidence_interval


def test_elo_confidence_interval_estimates_candidate_advantage() -> None:
    interval = estimate_elo_confidence_interval(wins=6, losses=2, draws=2)

    assert interval.games == 10
    assert interval.score_rate == pytest.approx(0.7)
    assert interval.elo > 0.0
    assert interval.lower < interval.elo < interval.upper


def test_elo_confidence_interval_handles_perfect_score() -> None:
    interval = estimate_elo_confidence_interval(wins=4, losses=0, draws=0)

    assert interval.games == 4
    assert interval.lower < interval.upper
    assert interval.upper < 2400.0


def test_elo_confidence_interval_rejects_invalid_counts() -> None:
    with pytest.raises(ValueError, match="wins"):
        estimate_elo_confidence_interval(wins=-1, losses=0)


def test_elo_confidence_interval_requires_supported_confidence_level() -> None:
    with pytest.raises(ValueError, match="confidence_level"):
        estimate_elo_confidence_interval(wins=1, losses=1, confidence_level=0.8)
