from __future__ import annotations

import random

import pytest

from zero_lab.evaluation import RandomLegalMoveAgent, UniformSearchAgent
from zero_lab.games.toy import TicTacToeState


def test_random_legal_move_agent_uses_seeded_rng() -> None:
    state = TicTacToeState().apply(4)
    first_rng = random.Random(17)
    second_rng = random.Random(17)
    agent = RandomLegalMoveAgent()

    first_actions = [agent.select_action(state, rng=first_rng) for _ in range(5)]
    second_actions = [agent.select_action(state, rng=second_rng) for _ in range(5)]

    assert first_actions == second_actions
    assert set(first_actions).issubset(state.legal_actions())


def test_random_legal_move_agent_rejects_terminal_state() -> None:
    state = TicTacToeState()
    for action in (0, 3, 1, 4, 2):
        state = state.apply(action)

    with pytest.raises(ValueError, match="without legal actions"):
        RandomLegalMoveAgent().select_action(state, rng=random.Random(0))


def test_uniform_search_agent_finds_immediate_tic_tac_toe_win() -> None:
    state = TicTacToeState()
    for action in (0, 3, 1, 4):
        state = state.apply(action)

    action = UniformSearchAgent(simulations=32).select_action(state, rng=random.Random(0))

    assert action == 2


def test_uniform_search_agent_validates_temperature() -> None:
    with pytest.raises(ValueError, match="temperature"):
        UniformSearchAgent(temperature=-0.1)
