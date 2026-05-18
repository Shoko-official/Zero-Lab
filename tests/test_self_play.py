from __future__ import annotations

from zero_lab.games.toy import TicTacToeGame
from zero_lab.search import MCTSSearchConfig
from zero_lab.search.alpha_zero import UniformEvaluator
from zero_lab.self_play import AlphaZeroSelfPlay, SelfPlayConfig


def test_self_play_generates_valid_episode() -> None:
    runner = AlphaZeroSelfPlay(
        UniformEvaluator(),
        MCTSSearchConfig(simulations=8),
        SelfPlayConfig(max_moves=9, temperature=0.0),
    )

    episode = runner.play(TicTacToeGame(), seed=7)

    assert episode.game == "tic_tac_toe"
    assert episode.length > 0
    assert all(abs(sum(step.policy) - 1.0) < 1e-9 for step in episode.steps)
    assert all(-1.0 <= step.value_target <= 1.0 for step in episode.steps)


def test_self_play_is_deterministic_with_fixed_seed() -> None:
    runner = AlphaZeroSelfPlay(
        UniformEvaluator(),
        MCTSSearchConfig(simulations=4),
        SelfPlayConfig(max_moves=5, temperature=1.0),
    )

    first = runner.play(TicTacToeGame(), seed=11)
    second = runner.play(TicTacToeGame(), seed=11)

    assert first == second
