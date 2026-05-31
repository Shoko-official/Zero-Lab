from __future__ import annotations

import pytest

from zero_lab.games.toy import TicTacToeState
from zero_lab.search.alpha_zero import UniformEvaluator, measure_batched_inference_latency


class StepClock:
    def __init__(self, step: float) -> None:
        self.current = 0.0
        self.step = step

    def __call__(self) -> float:
        self.current += self.step
        return self.current


def test_batched_inference_latency_reports_batch_timing() -> None:
    states = (TicTacToeState(), TicTacToeState().apply(0))

    report = measure_batched_inference_latency(
        UniformEvaluator(),
        states,
        repeats=3,
        warmup=1,
        clock=StepClock(0.01),
    )

    assert report.batch_size == 2
    assert report.repeats == 3
    assert report.warmup == 1
    assert report.evaluations == 6
    assert report.mean_seconds == pytest.approx(0.01)
    assert report.per_state_seconds == pytest.approx(0.005)
    assert report.to_dict()["batch_size"] == 2


def test_batched_inference_latency_rejects_empty_states() -> None:
    with pytest.raises(ValueError, match="states"):
        measure_batched_inference_latency(UniformEvaluator(), ())


def test_batched_inference_latency_rejects_invalid_repeats() -> None:
    with pytest.raises(ValueError, match="repeats"):
        measure_batched_inference_latency(UniformEvaluator(), (TicTacToeState(),), repeats=0)
