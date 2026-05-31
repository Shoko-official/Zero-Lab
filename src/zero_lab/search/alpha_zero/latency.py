"""Latency measurement helpers for AlphaZero batched inference."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from zero_lab.games import GameState
from zero_lab.search.alpha_zero.evaluator import AlphaZeroEvaluator
from zero_lab.search.alpha_zero.model_evaluator import evaluate_batch


@dataclass(frozen=True, slots=True)
class BatchedInferenceLatency:
    batch_size: int
    repeats: int
    warmup: int
    evaluations: int
    total_seconds: float
    mean_seconds: float
    min_seconds: float
    max_seconds: float
    per_state_seconds: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "batch_size": self.batch_size,
            "evaluations": self.evaluations,
            "max_seconds": self.max_seconds,
            "mean_seconds": self.mean_seconds,
            "min_seconds": self.min_seconds,
            "per_state_seconds": self.per_state_seconds,
            "repeats": self.repeats,
            "total_seconds": self.total_seconds,
            "warmup": self.warmup,
        }


def measure_batched_inference_latency(
    evaluator: AlphaZeroEvaluator,
    states: Sequence[GameState],
    *,
    repeats: int = 10,
    warmup: int = 1,
    clock: Callable[[], float] = time.perf_counter,
) -> BatchedInferenceLatency:
    states_tuple = tuple(states)
    if not states_tuple:
        raise ValueError("states must not be empty")
    if isinstance(repeats, bool) or repeats <= 0:
        raise ValueError("repeats must be a positive integer")
    if isinstance(warmup, bool) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")

    for _ in range(warmup):
        evaluate_batch(evaluator, states_tuple)

    durations: list[float] = []
    for _ in range(repeats):
        started = clock()
        evaluate_batch(evaluator, states_tuple)
        durations.append(clock() - started)

    total_seconds = sum(durations)
    mean_seconds = total_seconds / repeats
    batch_size = len(states_tuple)
    return BatchedInferenceLatency(
        batch_size=batch_size,
        repeats=repeats,
        warmup=warmup,
        evaluations=batch_size * repeats,
        total_seconds=total_seconds,
        mean_seconds=mean_seconds,
        min_seconds=min(durations),
        max_seconds=max(durations),
        per_state_seconds=mean_seconds / batch_size,
    )
