from __future__ import annotations

import pytest

from zero_lab.evaluation import (
    AlphaZeroCheckpoint,
    MatchConfig,
    RandomLegalMoveAgent,
    UniformSearchAgent,
    compare_alpha_zero_checkpoints,
)
from zero_lab.games.toy import TicTacToeGame


def test_checkpoint_comparison_runs_champion_against_candidate() -> None:
    champion = AlphaZeroCheckpoint(
        name="champion",
        uri="checkpoints/champion.pt",
        commit_hash="abc1234",
    )
    candidate = AlphaZeroCheckpoint(
        name="candidate",
        uri="checkpoints/candidate.pt",
        commit_hash="def5678",
    )

    comparison = compare_alpha_zero_checkpoints(
        champion=champion,
        candidate=candidate,
        champion_agent=RandomLegalMoveAgent(),
        candidate_agent=UniformSearchAgent(simulations=4),
        games=(TicTacToeGame(),),
        config=MatchConfig(seed=41, games_per_side=1),
    )

    assert comparison.champion == champion
    assert comparison.candidate == candidate
    assert [result.seed for result in comparison.results] == [41, 42]
    assert [(result.first_agent, result.second_agent) for result in comparison.results] == [
        ("champion", "candidate"),
        ("candidate", "champion"),
    ]


def test_checkpoint_requires_commit_hash() -> None:
    with pytest.raises(ValueError, match="commit_hash"):
        AlphaZeroCheckpoint(name="candidate", uri="checkpoints/candidate.pt", commit_hash="")


def test_checkpoint_comparison_requires_distinct_names() -> None:
    checkpoint = AlphaZeroCheckpoint(
        name="same",
        uri="checkpoints/same.pt",
        commit_hash="abc1234",
    )

    with pytest.raises(ValueError, match="must differ"):
        compare_alpha_zero_checkpoints(
            champion=checkpoint,
            candidate=checkpoint,
            champion_agent=RandomLegalMoveAgent(),
            candidate_agent=UniformSearchAgent(simulations=4),
            games=(TicTacToeGame(),),
            config=MatchConfig(seed=0),
        )
