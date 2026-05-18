"""Framework-light AlphaZero training batch contracts."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from zero_lab.games.base import FIRST_PLAYER, SECOND_PLAYER
from zero_lab.models import AlphaZeroBatch
from zero_lab.models.common import BatchShape, validate_observation_batch, validate_policy_mask


@dataclass(frozen=True, slots=True)
class AlphaZeroTrainingBatch:
    observations: tuple[tuple[float, ...], ...]
    legal_action_masks: tuple[tuple[bool, ...], ...]
    target_policies: tuple[tuple[float, ...], ...]
    target_values: tuple[float, ...]
    selected_actions: tuple[int, ...]
    current_players: tuple[int, ...]
    shape: BatchShape

    @classmethod
    def from_sequences(
        cls,
        *,
        observations: Sequence[Sequence[float | int]],
        legal_action_masks: Sequence[Sequence[bool]],
        target_policies: Sequence[Sequence[float | int]],
        target_values: Sequence[float | int],
        selected_actions: Sequence[int],
        current_players: Sequence[int],
        action_size: int,
    ) -> AlphaZeroTrainingBatch:
        if not observations:
            raise ValueError("observations must not be empty")

        shape = BatchShape(
            batch_size=len(observations),
            action_size=action_size,
            observation_size=len(observations[0]),
        )
        masks = validate_policy_mask(
            legal_action_masks,
            batch_size=shape.batch_size,
            action_size=shape.action_size,
        )
        policies = _validate_target_policies(
            target_policies,
            batch_size=shape.batch_size,
            action_size=shape.action_size,
        )
        _validate_policy_targets_are_legal(policies, masks)

        return cls(
            observations=validate_observation_batch(observations, shape=shape),
            legal_action_masks=masks,
            target_policies=policies,
            target_values=_validate_target_values(target_values, batch_size=shape.batch_size),
            selected_actions=_validate_selected_actions(
                selected_actions,
                batch_size=shape.batch_size,
                action_size=shape.action_size,
            ),
            current_players=_validate_current_players(
                current_players,
                batch_size=shape.batch_size,
            ),
            shape=shape,
        )

    def to_model_batch(self) -> AlphaZeroBatch:
        return AlphaZeroBatch(
            observations=self.observations,
            legal_action_masks=self.legal_action_masks,
            shape=self.shape,
        )


def _validate_target_policies(
    policies: Sequence[Sequence[float | int]],
    *,
    batch_size: int,
    action_size: int,
) -> tuple[tuple[float, ...], ...]:
    if len(policies) != batch_size:
        raise ValueError("target_policies batch dimension does not match batch_size")

    normalized: list[tuple[float, ...]] = []
    for policy in policies:
        if len(policy) != action_size:
            raise ValueError("target_policies action dimension does not match action_size")
        row = tuple(float(value) for value in policy)
        if any(not math.isfinite(value) for value in row):
            raise ValueError("target_policies must contain only finite values")
        if any(value < 0.0 for value in row):
            raise ValueError("target_policies must contain only non-negative values")
        if abs(sum(row) - 1.0) > 1e-6:
            raise ValueError("target_policies must sum to 1")
        normalized.append(row)

    return tuple(normalized)


def _validate_policy_targets_are_legal(
    policies: Sequence[Sequence[float]],
    legal_action_masks: Sequence[Sequence[bool]],
) -> None:
    for policy, mask in zip(policies, legal_action_masks, strict=True):
        for value, is_legal in zip(policy, mask, strict=True):
            if value > 0.0 and not is_legal:
                raise ValueError("target_policies must not assign mass to illegal actions")


def _validate_target_values(
    values: Sequence[float | int],
    *,
    batch_size: int,
) -> tuple[float, ...]:
    if len(values) != batch_size:
        raise ValueError("target_values batch dimension does not match batch_size")

    normalized = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in normalized):
        raise ValueError("target_values must contain only finite values")
    if any(abs(value) > 1.0 for value in normalized):
        raise ValueError("target_values must be in [-1, 1]")
    return normalized


def _validate_selected_actions(
    actions: Sequence[int],
    *,
    batch_size: int,
    action_size: int,
) -> tuple[int, ...]:
    if len(actions) != batch_size:
        raise ValueError("selected_actions batch dimension does not match batch_size")

    for action in actions:
        if isinstance(action, bool) or not isinstance(action, int):
            raise ValueError("selected_actions must contain only integers")
        if action < 0 or action >= action_size:
            raise ValueError("selected_actions must be within the action space")
    return tuple(actions)


def _validate_current_players(
    players: Sequence[int],
    *,
    batch_size: int,
) -> tuple[int, ...]:
    if len(players) != batch_size:
        raise ValueError("current_players batch dimension does not match batch_size")

    for player in players:
        if isinstance(player, bool) or player not in (FIRST_PLAYER, SECOND_PLAYER):
            raise ValueError("current_players must contain only 1 or -1")
    return tuple(players)
