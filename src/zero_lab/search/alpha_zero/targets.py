"""Training target helpers derived from MCTS visits."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping


def visit_count_policy(visit_counts: Mapping[int, int], action_size: int) -> tuple[float, ...]:
    if isinstance(action_size, bool) or action_size <= 0:
        raise ValueError("action_size must be a positive integer")

    policy = [0.0] * action_size
    total_visits = 0
    for action, visits in visit_counts.items():
        if isinstance(action, bool) or action < 0 or action >= action_size:
            raise ValueError("visit_counts contains an action outside the action space")
        if isinstance(visits, bool) or visits < 0:
            raise ValueError("visit counts must be non-negative integers")
        total_visits += visits

    if total_visits == 0:
        raise ValueError("visit_counts must contain at least one visit")

    for action, visits in visit_counts.items():
        policy[action] = visits / total_visits
    return tuple(policy)


def select_action_by_temperature(
    visit_counts: Mapping[int, int],
    *,
    temperature: float,
    rng: random.Random | None = None,
) -> int:
    if not visit_counts:
        raise ValueError("visit_counts must not be empty")
    if temperature < 0.0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and non-negative")

    if temperature == 0.0:
        return max(visit_counts, key=lambda action: (visit_counts[action], -action))

    actions = tuple(sorted(visit_counts))
    weights = tuple(float(visit_counts[action]) ** (1.0 / temperature) for action in actions)
    total = sum(weights)
    if total <= 0.0 or not math.isfinite(total):
        raise ValueError("visit_counts must contain positive finite sampling weight")

    generator = random if rng is None else rng
    threshold = generator.random() * total
    cumulative = 0.0
    for action, weight in zip(actions, weights, strict=True):
        cumulative += weight
        if threshold <= cumulative:
            return action

    return actions[-1]
