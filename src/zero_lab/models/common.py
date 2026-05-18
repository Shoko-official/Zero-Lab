"""Shared model contract primitives."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BatchShape:
    batch_size: int
    action_size: int
    observation_size: int

    def __post_init__(self) -> None:
        _require_positive_integer(self.batch_size, "batch_size")
        _require_positive_integer(self.action_size, "action_size")
        _require_positive_integer(self.observation_size, "observation_size")


@dataclass(frozen=True, slots=True)
class PolicyValueOutput:
    policy_logits: tuple[tuple[float, ...], ...]
    values: tuple[float, ...]

    def validate(self, *, batch_size: int, action_size: int) -> None:
        _require_positive_integer(batch_size, "batch_size")
        _require_positive_integer(action_size, "action_size")

        if len(self.policy_logits) != batch_size:
            raise ValueError("policy_logits batch dimension does not match batch_size")
        if len(self.values) != batch_size:
            raise ValueError("values batch dimension does not match batch_size")

        for row in self.policy_logits:
            if len(row) != action_size:
                raise ValueError("policy_logits action dimension does not match action_size")
            _validate_finite_values(row, "policy_logits")

        _validate_finite_values(self.values, "values")


def validate_observation_batch(
    observations: Sequence[Sequence[float | int]],
    *,
    shape: BatchShape,
) -> tuple[tuple[float, ...], ...]:
    if len(observations) != shape.batch_size:
        raise ValueError("observation batch dimension does not match batch_size")

    normalized: list[tuple[float, ...]] = []
    for observation in observations:
        if len(observation) != shape.observation_size:
            raise ValueError("observation dimension does not match observation_size")
        row = tuple(float(value) for value in observation)
        _validate_finite_values(row, "observations")
        normalized.append(row)

    return tuple(normalized)


def validate_policy_mask(
    legal_action_masks: Sequence[Sequence[bool]],
    *,
    batch_size: int,
    action_size: int,
) -> tuple[tuple[bool, ...], ...]:
    _require_positive_integer(batch_size, "batch_size")
    _require_positive_integer(action_size, "action_size")

    if len(legal_action_masks) != batch_size:
        raise ValueError("legal_action_masks batch dimension does not match batch_size")

    normalized: list[tuple[bool, ...]] = []
    for mask in legal_action_masks:
        if len(mask) != action_size:
            raise ValueError("legal_action_masks action dimension does not match action_size")
        if not any(mask):
            raise ValueError("legal_action_masks must contain at least one legal action")
        normalized.append(tuple(mask))

    return tuple(normalized)


def normalize_policy_value_output(
    policy_logits: Sequence[Sequence[float | int]],
    values: Sequence[float | int],
) -> PolicyValueOutput:
    return PolicyValueOutput(
        policy_logits=tuple(tuple(float(value) for value in row) for row in policy_logits),
        values=tuple(float(value) for value in values),
    )


def _require_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_finite_values(values: Sequence[float], name: str) -> None:
    for value in values:
        if not math.isfinite(value):
            raise ValueError(f"{name} must contain only finite values")
