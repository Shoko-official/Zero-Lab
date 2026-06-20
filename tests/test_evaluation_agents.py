from __future__ import annotations

import random

import pytest

from zero_lab.evaluation import AlphaZeroSearchAgent, RandomLegalMoveAgent, UniformSearchAgent
from zero_lab.games import GameState
from zero_lab.games.toy import TicTacToeState
from zero_lab.search.alpha_zero import Evaluation


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


def test_alpha_zero_search_agent_uses_custom_evaluator() -> None:
    class CenterBiasedEvaluator:
        def evaluate(self, state: GameState) -> Evaluation:
            policy = {action: 1.0 for action in state.legal_actions()}
            policy[4] = 10.0
            return Evaluation(policy=policy, value=0.0)

    action = AlphaZeroSearchAgent(
        CenterBiasedEvaluator(),
        simulations=8,
    ).select_action(TicTacToeState(), rng=random.Random(0))

    assert action == 4


def test_alpha_zero_search_agent_validates_temperature() -> None:
    class NeutralEvaluator:
        def evaluate(self, state: GameState) -> Evaluation:
            return Evaluation(policy={action: 1.0 for action in state.legal_actions()}, value=0.0)

    with pytest.raises(ValueError, match="temperature"):
        AlphaZeroSearchAgent(NeutralEvaluator(), temperature=-0.1)


def test_uniform_search_agent_validates_temperature() -> None:
    with pytest.raises(ValueError, match="temperature"):
        UniformSearchAgent(temperature=-0.1)
