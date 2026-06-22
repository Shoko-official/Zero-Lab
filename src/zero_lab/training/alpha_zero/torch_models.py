"""Small PyTorch modules for AlphaZero training smoke paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from zero_lab.training.alpha_zero.loops import (
    AlphaZeroCheckpointMetadata,
    load_alpha_zero_model_checkpoint,
)
from zero_lab.training.alpha_zero.torch_evaluator import TorchAlphaZeroEvaluator


class LinearAlphaZeroModel(torch.nn.Module):
    def __init__(self, observation_size: int, action_size: int) -> None:
        super().__init__()
        _require_positive_integer(observation_size, "observation_size")
        _require_positive_integer(action_size, "action_size")
        self.policy = torch.nn.Linear(observation_size, action_size)
        self.value = torch.nn.Linear(observation_size, 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.policy(observations), self.value(observations).squeeze(dim=1)


@dataclass(frozen=True, slots=True)
class LoadedLinearAlphaZeroEvaluator:
    model: LinearAlphaZeroModel
    evaluator: TorchAlphaZeroEvaluator
    metadata: AlphaZeroCheckpointMetadata


def load_linear_alpha_zero_evaluator_checkpoint(
    path: Path | str,
    *,
    observation_size: int,
    action_size: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> LoadedLinearAlphaZeroEvaluator:
    model = LinearAlphaZeroModel(observation_size=observation_size, action_size=action_size)
    metadata = load_alpha_zero_model_checkpoint(Path(path), model=model)
    target_device = None if device is None else torch.device(device)
    if target_device is None:
        model.to(dtype=dtype)
    else:
        model.to(device=target_device, dtype=dtype)
    evaluator = TorchAlphaZeroEvaluator(model, device=target_device, dtype=dtype)
    return LoadedLinearAlphaZeroEvaluator(
        model=model,
        evaluator=evaluator,
        metadata=metadata,
    )


def _require_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
