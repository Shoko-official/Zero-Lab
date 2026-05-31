"""AlphaZero evaluator adapters."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import cast

from zero_lab.games import GameState
from zero_lab.models import AlphaZeroBatch, AlphaZeroModel
from zero_lab.search.alpha_zero.evaluator import (
    AlphaZeroBatchEvaluator,
    AlphaZeroEvaluator,
    Evaluation,
)


@dataclass(frozen=True, slots=True)
class ModelEvaluator:
    model: AlphaZeroModel

    def evaluate(self, state: GameState) -> Evaluation:
        return self.evaluate_batch((state,))[0]

    def evaluate_batch(self, states: Sequence[GameState]) -> tuple[Evaluation, ...]:
        states_tuple = tuple(states)
        if not states_tuple:
            return ()

        evaluations: list[Evaluation | None] = [None] * len(states_tuple)
        pending_positions: list[int] = []
        pending_states: list[GameState] = []
        for position, state in enumerate(states_tuple):
            if state.is_terminal:
                evaluations[position] = terminal_evaluation(state)
            else:
                pending_positions.append(position)
                pending_states.append(state)

        if pending_states:
            action_size = pending_states[0].action_size
            observation_size = len(pending_states[0].canonical_observation())
            for state in pending_states:
                if state.action_size != action_size:
                    raise ValueError("batched model evaluation requires matching action_size")
                if len(state.canonical_observation()) != observation_size:
                    raise ValueError("batched model evaluation requires matching observation size")

            batch = AlphaZeroBatch.from_sequences(
                observations=tuple(state.canonical_observation() for state in pending_states),
                legal_action_masks=tuple(state.legal_action_mask() for state in pending_states),
                action_size=action_size,
            )
            output = self.model.predict(batch)
            for batch_index, position in enumerate(pending_positions):
                state = pending_states[batch_index]
                logits = output.policy_value.policy_logits[batch_index]
                evaluations[position] = Evaluation(
                    policy=softmax_policy(logits, state.legal_actions()),
                    value=output.policy_value.values[batch_index],
                )

        return tuple(_require_evaluation(evaluation) for evaluation in evaluations)


def terminal_evaluation(state: GameState) -> Evaluation:
    if not state.is_terminal:
        raise ValueError("state must be terminal")
    outcome = state.outcome_for(state.current_player)
    return Evaluation(policy={}, value=0.0 if outcome is None else float(outcome))


@dataclass(frozen=True, slots=True)
class BatchedModelEvaluator:
    model: AlphaZeroModel

    def evaluate(self, state: GameState) -> Evaluation:
        return ModelEvaluator(self.model).evaluate(state)

    def evaluate_batch(self, states: Sequence[GameState]) -> tuple[Evaluation, ...]:
        return ModelEvaluator(self.model).evaluate_batch(states)


@dataclass(frozen=True, slots=True)
class UniformEvaluator:
    value: float = 0.0

    def evaluate(self, state: GameState) -> Evaluation:
        if state.is_terminal:
            return terminal_evaluation(state)
        return Evaluation(policy=uniform_policy(state.legal_actions()), value=self.value)

    def evaluate_batch(self, states: Sequence[GameState]) -> tuple[Evaluation, ...]:
        return tuple(self.evaluate(state) for state in states)


def evaluate_batch(
    evaluator: AlphaZeroEvaluator,
    states: Sequence[GameState],
) -> tuple[Evaluation, ...]:
    states_tuple = tuple(states)
    if not states_tuple:
        return ()

    batch_method = getattr(evaluator, "evaluate_batch", None)
    if callable(batch_method):
        return tuple(cast(AlphaZeroBatchEvaluator, evaluator).evaluate_batch(states_tuple))
    return tuple(evaluator.evaluate(state) for state in states_tuple)


def softmax_policy(logits: Iterable[float], legal_actions: Iterable[int]) -> dict[int, float]:
    logits_by_action = tuple(float(value) for value in logits)
    legal = tuple(legal_actions)
    if not legal:
        return {}

    selected_logits = tuple(logits_by_action[action] for action in legal)
    if not all(math.isfinite(value) for value in selected_logits):
        return uniform_policy(legal)

    max_logit = max(selected_logits)
    weights = tuple(math.exp(value - max_logit) for value in selected_logits)
    total = sum(weights)
    if total <= 0.0 or not math.isfinite(total):
        return uniform_policy(legal)

    return {action: weight / total for action, weight in zip(legal, weights, strict=True)}


def uniform_policy(legal_actions: Iterable[int]) -> dict[int, float]:
    legal = tuple(legal_actions)
    if not legal:
        return {}
    probability = 1.0 / len(legal)
    return {action: probability for action in legal}


def normalize_policy(policy: dict[int, float], legal_actions: Iterable[int]) -> dict[int, float]:
    legal = tuple(legal_actions)
    if not legal:
        return {}

    normalized = {action: max(0.0, float(policy.get(action, 0.0))) for action in legal}
    total = sum(normalized.values())
    if total <= 0.0 or not math.isfinite(total):
        return uniform_policy(legal)
    return {action: value / total for action, value in normalized.items()}


def as_evaluator(evaluator: AlphaZeroEvaluator | AlphaZeroModel) -> AlphaZeroEvaluator:
    if hasattr(evaluator, "evaluate"):
        return cast(AlphaZeroEvaluator, evaluator)
    return ModelEvaluator(evaluator)


def _require_evaluation(evaluation: Evaluation | None) -> Evaluation:
    if evaluation is None:
        raise RuntimeError("missing evaluation")
    return evaluation
