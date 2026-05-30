from __future__ import annotations

from zero_lab.evaluation import (
    MatchConfig,
    RandomLegalMoveAgent,
    UniformSearchAgent,
    run_head_to_head,
)
from zero_lab.games.toy import ConnectFourGame, TicTacToeGame


def test_head_to_head_runner_alternates_sides_and_seeds() -> None:
    results = run_head_to_head(
        games=(TicTacToeGame(), ConnectFourGame()),
        first_agent=RandomLegalMoveAgent(),
        second_agent=UniformSearchAgent(simulations=4),
        config=MatchConfig(seed=11, games_per_side=1),
    )

    assert [result.seed for result in results] == [11, 12, 13, 14]
    assert [(result.first_agent, result.second_agent) for result in results] == [
        ("random_legal", "uniform_search"),
        ("uniform_search", "random_legal"),
        ("random_legal", "uniform_search"),
        ("uniform_search", "random_legal"),
    ]
    assert [result.game for result in results] == [
        "tic_tac_toe",
        "tic_tac_toe",
        "connect_four",
        "connect_four",
    ]


def test_head_to_head_runner_is_deterministic_with_fixed_seed() -> None:
    config = MatchConfig(seed=23, games_per_side=2)
    first = run_head_to_head(
        games=(TicTacToeGame(),),
        first_agent=RandomLegalMoveAgent(),
        second_agent=UniformSearchAgent(simulations=4),
        config=config,
    )
    second = run_head_to_head(
        games=(TicTacToeGame(),),
        first_agent=RandomLegalMoveAgent(),
        second_agent=UniformSearchAgent(simulations=4),
        config=config,
    )

    assert first == second
    assert all(result.terminal for result in first)
