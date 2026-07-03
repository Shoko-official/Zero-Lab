"""Small PyTorch modules for AlphaZero training smoke paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from zero_lab.training.alpha_zero.loops import (
    AlphaZeroCheckpointMetadata,
    load_alpha_zero_model_checkpoint,
)
from zero_lab.training.alpha_zero.torch_evaluator import TorchAlphaZeroEvaluator

AlphaZeroModelName = Literal["linear", "mlp"]


class LinearAlphaZeroModel(torch.nn.Module):
    def __init__(self, observation_size: int, action_size: int) -> None:
        super().__init__()
        _require_positive_integer(observation_size, "observation_size")
        _require_positive_integer(action_size, "action_size")
        self.policy = torch.nn.Linear(observation_size, action_size)
        self.value = torch.nn.Linear(observation_size, 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.policy(observations), self.value(observations).squeeze(dim=1)


class MLPAlphaZeroModel(torch.nn.Module):
    def __init__(self, observation_size: int, action_size: int, hidden_size: int = 128) -> None:
        super().__init__()
        _require_positive_integer(observation_size, "observation_size")
        _require_positive_integer(action_size, "action_size")
        _require_positive_integer(hidden_size, "hidden_size")
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(observation_size, hidden_size),
            torch.nn.ReLU(),
        )
        self.policy = torch.nn.Linear(hidden_size, action_size)
        self.value = torch.nn.Linear(hidden_size, 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(observations)
        return self.policy(features), self.value(features).squeeze(dim=1)


AlphaZeroTorchModel = LinearAlphaZeroModel | MLPAlphaZeroModel


@dataclass(frozen=True, slots=True)
class LoadedAlphaZeroEvaluator:
    model_name: AlphaZeroModelName
    model: AlphaZeroTorchModel
    evaluator: TorchAlphaZeroEvaluator
    metadata: AlphaZeroCheckpointMetadata


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
    loaded = load_alpha_zero_evaluator_checkpoint(
        path,
        model="linear",
        observation_size=observation_size,
        action_size=action_size,
        device=device,
        dtype=dtype,
    )
    if not isinstance(loaded.model, LinearAlphaZeroModel):
        raise TypeError("linear checkpoint loader restored unexpected model type")
    return LoadedLinearAlphaZeroEvaluator(
        model=loaded.model,
        evaluator=loaded.evaluator,
        metadata=loaded.metadata,
    )


def load_alpha_zero_evaluator_checkpoint(
    path: Path | str,
    *,
    model: AlphaZeroModelName,
    observation_size: int,
    action_size: int,
    hidden_size: int = 128,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> LoadedAlphaZeroEvaluator:
    model_name, torch_model = _build_alpha_zero_model(
        model,
        observation_size=observation_size,
        action_size=action_size,
        hidden_size=hidden_size,
    )
    metadata = load_alpha_zero_model_checkpoint(Path(path), model=torch_model)
    target_device = None if device is None else torch.device(device)
    if target_device is None:
        torch_model.to(dtype=dtype)
    else:
        torch_model.to(device=target_device, dtype=dtype)
    evaluator = TorchAlphaZeroEvaluator(torch_model, device=target_device, dtype=dtype)
    return LoadedAlphaZeroEvaluator(
        model_name=model_name,
        model=torch_model,
        evaluator=evaluator,
        metadata=metadata,
    )


def _build_alpha_zero_model(
    model: AlphaZeroModelName,
    *,
    observation_size: int,
    action_size: int,
    hidden_size: int,
) -> tuple[AlphaZeroModelName, AlphaZeroTorchModel]:
    if model == "linear":
        return model, LinearAlphaZeroModel(
            observation_size=observation_size,
            action_size=action_size,
        )
    if model == "mlp":
        return model, MLPAlphaZeroModel(
            observation_size=observation_size,
            action_size=action_size,
            hidden_size=hidden_size,
        )
    raise ValueError("model must be one of: linear, mlp")


def _require_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
