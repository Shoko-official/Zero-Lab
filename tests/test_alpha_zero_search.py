from __future__ import annotations

import random

import pytest

from zero_lab.games.toy.tic_tac_toe import TicTacToeState
from zero_lab.models import AlphaZeroBatch, AlphaZeroModel, AlphaZeroOutput
from zero_lab.search.alpha_zero import (
    AlphaZeroSearch,
    BatchedModelEvaluator,
    MCTSSearchConfig,
    ModelEvaluator,
    UniformEvaluator,
    evaluate_batch,
    select_action_by_temperature,
    visit_count_policy,
)
from zero_lab.search.alpha_zero.model_evaluator import softmax_policy


class FixedLogitModel:
    def predict(self, batch: AlphaZeroBatch) -> AlphaZeroOutput:
        logits = tuple(tuple(float(action) for action in range(batch.shape.action_size)))
        return AlphaZeroOutput.from_sequences(
            policy_logits=(logits,),
            values=(0.0,),
            batch=batch,
        )


class RecordingBatchModel:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def predict(self, batch: AlphaZeroBatch) -> AlphaZeroOutput:
        self.batch_sizes.append(batch.shape.batch_size)
        logits = tuple(
            tuple(float(action) for action in range(batch.shape.action_size))
            for _ in range(batch.shape.batch_size)
        )
        values = tuple(0.25 for _ in range(batch.shape.batch_size))
        return AlphaZeroOutput.from_sequences(
            policy_logits=logits,
            values=values,
            batch=batch,
        )


def test_fixed_logit_model_satisfies_protocol() -> None:
    model: AlphaZeroModel = FixedLogitModel()
    batch = AlphaZeroBatch.from_sequences(
        observations=((0, 0, 0),),
        legal_action_masks=((True, False, True),),
        action_size=3,
    )

    output = model.predict(batch)

    assert output.policy_value.values == (0.0,)


def test_softmax_policy_filters_illegal_actions() -> None:
    policy = softmax_policy((100.0, 0.0, 1.0), (1, 2))

    assert set(policy) == {1, 2}
    assert policy[2] > policy[1]
    assert sum(policy.values()) == pytest.approx(1.0)


def test_model_evaluator_uses_legal_action_mask() -> None:
    state = TicTacToeState().apply(0)
    evaluation = ModelEvaluator(FixedLogitModel()).evaluate(state)

    assert 0 not in evaluation.policy
    assert set(evaluation.policy) == set(state.legal_actions())
    assert sum(evaluation.policy.values()) == pytest.approx(1.0)


def test_model_evaluator_batches_non_terminal_states() -> None:
    model = RecordingBatchModel()
    states = (TicTacToeState(), TicTacToeState().apply(0))

    evaluations = evaluate_batch(BatchedModelEvaluator(model), states)

    assert model.batch_sizes == [2]
    assert len(evaluations) == 2
    assert set(evaluations[1].policy) == set(states[1].legal_actions())
    assert evaluations[0].value == pytest.approx(0.25)


def test_model_evaluator_skips_terminal_states_in_batch() -> None:
    model = RecordingBatchModel()
    state = TicTacToeState()
    for action in (0, 3, 1, 4, 2):
        state = state.apply(action)

    evaluations = evaluate_batch(ModelEvaluator(model), (state,))

    assert model.batch_sizes == []
    assert evaluations[0].policy == {}
    assert evaluations[0].value == pytest.approx(-1.0)


def test_visit_count_policy_normalizes_counts() -> None:
    policy = visit_count_policy({0: 2, 2: 6}, action_size=4)

    assert policy == (0.25, 0.0, 0.75, 0.0)


def test_select_action_by_zero_temperature_uses_visit_argmax() -> None:
    assert select_action_by_temperature({0: 3, 1: 7, 2: 7}, temperature=0.0) == 1


def test_select_action_by_temperature_is_seedable() -> None:
    rng = random.Random(7)

    action = select_action_by_temperature({0: 1, 1: 100}, temperature=1.0, rng=rng)

    assert action == 1


def test_search_returns_empty_result_for_terminal_root() -> None:
    state = TicTacToeState()
    for action in (0, 3, 1, 4, 2):
        state = state.apply(action)

    result = AlphaZeroSearch(
        UniformEvaluator(),
        MCTSSearchConfig(simulations=8),
    ).run(state)

    assert result.visit_counts == {}
    assert result.policy == {}


def test_search_finds_immediate_tic_tac_toe_win() -> None:
    state = TicTacToeState()
    for action in (0, 3, 1, 4):
        state = state.apply(action)

    result = AlphaZeroSearch(
        UniformEvaluator(),
        MCTSSearchConfig(simulations=32),
    ).run(state)

    assert result.best_action == 2
    assert result.action_values[2] == pytest.approx(1.0)
    assert sum(result.visit_counts.values()) == 32
