"""MuZero model contract."""

from __future__ import annotations

import math
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
class MuZeroInitialOutput:
    latent_states: tuple[tuple[float, ...], ...]
    policy_value: PolicyValueOutput

    @classmethod
    def from_sequences(
        cls,
        *,
        latent_states: Sequence[Sequence[float | int]],
        policy_logits: Sequence[Sequence[float | int]],
        values: Sequence[float | int],
        batch_size: int,
        action_size: int,
        latent_size: int,
    ) -> MuZeroInitialOutput:
        states = _validate_latent_batch(
            latent_states,
            batch_size=batch_size,
            latent_size=latent_size,
        )
        policy_value = normalize_policy_value_output(policy_logits, values)
        policy_value.validate(batch_size=batch_size, action_size=action_size)
        return cls(latent_states=states, policy_value=policy_value)


@dataclass(frozen=True, slots=True)
class MuZeroRecurrentOutput:
    latent_states: tuple[tuple[float, ...], ...]
    policy_value: PolicyValueOutput
    rewards: tuple[float, ...]

    @classmethod
    def from_sequences(
        cls,
        *,
        latent_states: Sequence[Sequence[float | int]],
        policy_logits: Sequence[Sequence[float | int]],
        values: Sequence[float | int],
        rewards: Sequence[float | int],
        batch_size: int,
        action_size: int,
        latent_size: int,
    ) -> MuZeroRecurrentOutput:
        states = _validate_latent_batch(
            latent_states,
            batch_size=batch_size,
            latent_size=latent_size,
        )
        policy_value = normalize_policy_value_output(policy_logits, values)
        policy_value.validate(batch_size=batch_size, action_size=action_size)

        if len(rewards) != batch_size:
            raise ValueError("rewards batch dimension does not match batch_size")
        reward_values = tuple(float(value) for value in rewards)
        _validate_finite_values(reward_values, "rewards")

        return cls(
            latent_states=states,
            policy_value=policy_value,
            rewards=reward_values,
        )


@dataclass(frozen=True, slots=True)
class RecurrentBatch:
    latent_states: tuple[tuple[float, ...], ...]
    actions: tuple[int, ...]
    legal_action_masks: tuple[tuple[bool, ...], ...]
    batch_size: int
    action_size: int
    latent_size: int

    @classmethod
    def from_sequences(
        cls,
        *,
        latent_states: Sequence[Sequence[float | int]],
        actions: Sequence[int],
        legal_action_masks: Sequence[Sequence[bool]],
        action_size: int,
    ) -> RecurrentBatch:
        if not latent_states:
            raise ValueError("latent_states must not be empty")

        batch_size = len(latent_states)
        latent_size = len(latent_states[0])
        states = _validate_latent_batch(
            latent_states,
            batch_size=batch_size,
            latent_size=latent_size,
        )

        if len(actions) != batch_size:
            raise ValueError("actions batch dimension does not match batch_size")

        normalized_actions: list[int] = []
        for action in actions:
            if isinstance(action, bool) or not isinstance(action, int):
                raise ValueError("actions must be integers")
            if action < 0 or action >= action_size:
                raise ValueError("actions must be inside the action space")
            normalized_actions.append(action)

        masks = validate_policy_mask(
            legal_action_masks,
            batch_size=batch_size,
            action_size=action_size,
        )

        return cls(
            latent_states=states,
            actions=tuple(normalized_actions),
            legal_action_masks=masks,
            batch_size=batch_size,
            action_size=action_size,
            latent_size=latent_size,
        )


class MuZeroModel(Protocol):
    def initial_inference(
        self,
        observations: Sequence[Sequence[float | int]],
        legal_action_masks: Sequence[Sequence[bool]],
    ) -> MuZeroInitialOutput:
        pass

    def recurrent_inference(self, batch: RecurrentBatch) -> MuZeroRecurrentOutput:
        pass


def validate_initial_observations(
    observations: Sequence[Sequence[float | int]],
    *,
    action_size: int,
    legal_action_masks: Sequence[Sequence[bool]],
) -> tuple[tuple[float, ...], ...]:
    if not observations:
        raise ValueError("observations must not be empty")

    shape = BatchShape(
        batch_size=len(observations),
        action_size=action_size,
        observation_size=len(observations[0]),
    )
    validate_policy_mask(
        legal_action_masks,
        batch_size=shape.batch_size,
        action_size=shape.action_size,
    )
    return validate_observation_batch(observations, shape=shape)


def _validate_latent_batch(
    latent_states: Sequence[Sequence[float | int]],
    *,
    batch_size: int,
    latent_size: int,
) -> tuple[tuple[float, ...], ...]:
    if len(latent_states) != batch_size:
        raise ValueError("latent_states batch dimension does not match batch_size")

    normalized: list[tuple[float, ...]] = []
    for state in latent_states:
        if len(state) != latent_size:
            raise ValueError("latent_states dimension does not match latent_size")
        row = tuple(float(value) for value in state)
        _validate_finite_values(row, "latent_states")
        normalized.append(row)

    return tuple(normalized)


def _validate_finite_values(values: Sequence[float], name: str) -> None:
    for value in values:
        if not math.isfinite(value):
            raise ValueError(f"{name} must contain only finite values")
