"""AlphaZero evaluator adapters."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from zero_lab.games import GameState
from zero_lab.models import AlphaZeroBatch, AlphaZeroModel
from zero_lab.search.alpha_zero.evaluator import AlphaZeroEvaluator, Evaluation


@dataclass(frozen=True, slots=True)
class ModelEvaluator:
    model: AlphaZeroModel

    def evaluate(self, state: GameState) -> Evaluation:
        if state.is_terminal:
            outcome = state.outcome_for(state.current_player)
            return Evaluation(policy={}, value=0.0 if outcome is None else float(outcome))

        batch = AlphaZeroBatch.from_sequences(
            observations=(state.canonical_observation(),),
            legal_action_masks=(state.legal_action_mask(),),
            action_size=state.action_size,
        )
        output = self.model.predict(batch)
        logits = output.policy_value.policy_logits[0]
        policy = softmax_policy(logits, state.legal_actions())
        value = output.policy_value.values[0]
        return Evaluation(policy=policy, value=value)


@dataclass(frozen=True, slots=True)
class UniformEvaluator:
    value: float = 0.0

    def evaluate(self, state: GameState) -> Evaluation:
        if state.is_terminal:
            outcome = state.outcome_for(state.current_player)
            return Evaluation(policy={}, value=0.0 if outcome is None else float(outcome))
        return Evaluation(policy=uniform_policy(state.legal_actions()), value=self.value)


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
        return evaluator
    return ModelEvaluator(evaluator)
