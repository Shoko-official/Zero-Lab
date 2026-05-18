"""AlphaZero model contract."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from zero_lab.models.common import (
    BatchShape,
    PolicyValueOutput,
    normalize_policy_value_output,
    validate_observation_batch,
    validate_policy_mask,
)


@dataclass(frozen=True, slots=True)
class AlphaZeroBatch:
    observations: tuple[tuple[float, ...], ...]
    legal_action_masks: tuple[tuple[bool, ...], ...]
    shape: BatchShape

    @classmethod
    def from_sequences(
        cls,
        *,
        observations: Sequence[Sequence[float | int]],
        legal_action_masks: Sequence[Sequence[bool]],
        action_size: int,
    ) -> AlphaZeroBatch:
        if not observations:
            raise ValueError("observations must not be empty")

        shape = BatchShape(
            batch_size=len(observations),
            action_size=action_size,
            observation_size=len(observations[0]),
        )
        return cls(
            observations=validate_observation_batch(observations, shape=shape),
            legal_action_masks=validate_policy_mask(
                legal_action_masks,
                batch_size=shape.batch_size,
                action_size=shape.action_size,
            ),
            shape=shape,
        )


@dataclass(frozen=True, slots=True)
class AlphaZeroOutput:
    policy_value: PolicyValueOutput

    @classmethod
    def from_sequences(
        cls,
        *,
        policy_logits: Sequence[Sequence[float | int]],
        values: Sequence[float | int],
        batch: AlphaZeroBatch,
    ) -> AlphaZeroOutput:
        policy_value = normalize_policy_value_output(policy_logits, values)
        policy_value.validate(
            batch_size=batch.shape.batch_size,
            action_size=batch.shape.action_size,
        )
        return cls(policy_value=policy_value)


class AlphaZeroModel(Protocol):
    def predict(self, batch: AlphaZeroBatch) -> AlphaZeroOutput:
        pass
