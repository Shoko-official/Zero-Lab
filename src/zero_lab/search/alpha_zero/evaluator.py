"""Evaluator contract used by AlphaZero search."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from zero_lab.games import GameState


@dataclass(frozen=True, slots=True)
class Evaluation:
    policy: Mapping[int, float]
    value: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise ValueError("value must be finite")
        for action, prior in self.policy.items():
            if isinstance(action, bool) or not isinstance(action, int):
                raise ValueError("policy actions must be integers")
            if not math.isfinite(prior) or prior < 0.0:
                raise ValueError("policy priors must be finite non-negative values")


class AlphaZeroEvaluator(Protocol):
    def evaluate(self, state: GameState) -> Evaluation:
        pass


class AlphaZeroBatchEvaluator(AlphaZeroEvaluator, Protocol):
    def evaluate_batch(self, states: Sequence[GameState]) -> tuple[Evaluation, ...]:
        pass
