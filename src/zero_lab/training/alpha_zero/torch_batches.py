"""PyTorch tensor adapters for AlphaZero training batches."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from zero_lab.models.common import BatchShape
from zero_lab.training.alpha_zero.batches import AlphaZeroTrainingBatch


@dataclass(frozen=True, slots=True)
class TorchAlphaZeroTrainingBatch:
    observations: torch.Tensor
    legal_action_masks: torch.Tensor
    target_policies: torch.Tensor
    target_values: torch.Tensor
    selected_actions: torch.Tensor
    current_players: torch.Tensor
    shape: BatchShape

    def __post_init__(self) -> None:
        _require_shape(self.observations, (self.shape.batch_size, self.shape.observation_size))
        _require_shape(self.legal_action_masks, (self.shape.batch_size, self.shape.action_size))
        _require_shape(self.target_policies, (self.shape.batch_size, self.shape.action_size))
        _require_shape(self.target_values, (self.shape.batch_size,))
        _require_shape(self.selected_actions, (self.shape.batch_size,))
        _require_shape(self.current_players, (self.shape.batch_size,))
        _require_dtype(self.target_policies, self.observations.dtype, "target_policies")
        _require_dtype(self.target_values, self.observations.dtype, "target_values")
        _require_dtype(self.legal_action_masks, torch.bool, "legal_action_masks")
        _require_dtype(self.selected_actions, torch.long, "selected_actions")
        _require_dtype(self.current_players, torch.long, "current_players")
        _require_device(self.target_policies, self.observations.device, "target_policies")
        _require_device(self.target_values, self.observations.device, "target_values")
        _require_device(self.legal_action_masks, self.observations.device, "legal_action_masks")
        _require_device(self.selected_actions, self.observations.device, "selected_actions")
        _require_device(self.current_players, self.observations.device, "current_players")

    @property
    def device(self) -> torch.device:
        return self.observations.device

    @property
    def dtype(self) -> torch.dtype:
        return self.observations.dtype


def as_torch_alpha_zero_training_batch(
    batch: AlphaZeroTrainingBatch,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> TorchAlphaZeroTrainingBatch:
    target_device = None if device is None else torch.device(device)
    observations = torch.as_tensor(batch.observations, dtype=dtype, device=target_device)
    target_policies = torch.as_tensor(batch.target_policies, dtype=dtype, device=target_device)
    target_values = torch.as_tensor(batch.target_values, dtype=dtype, device=target_device)
    legal_action_masks = torch.as_tensor(
        batch.legal_action_masks,
        dtype=torch.bool,
        device=target_device,
    )
    selected_actions = torch.as_tensor(
        batch.selected_actions,
        dtype=torch.long,
        device=target_device,
    )
    current_players = torch.as_tensor(
        batch.current_players,
        dtype=torch.long,
        device=target_device,
    )

    return TorchAlphaZeroTrainingBatch(
        observations=observations,
        legal_action_masks=legal_action_masks,
        target_policies=target_policies,
        target_values=target_values,
        selected_actions=selected_actions,
        current_players=current_players,
        shape=batch.shape,
    )


def _require_shape(tensor: torch.Tensor, expected_shape: tuple[int, ...]) -> None:
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"tensor shape must be {expected_shape}")


def _require_dtype(tensor: torch.Tensor, expected_dtype: torch.dtype, name: str) -> None:
    if tensor.dtype != expected_dtype:
        raise TypeError(f"{name} must use dtype {expected_dtype}")


def _require_device(tensor: torch.Tensor, expected_device: torch.device, name: str) -> None:
    if tensor.device != expected_device:
        raise ValueError(f"{name} must be on device {expected_device}")
