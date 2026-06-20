"""Optimizer-backed AlphaZero training steps."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch

from zero_lab.training.alpha_zero.losses import AlphaZeroLossConfig, alpha_zero_loss
from zero_lab.training.alpha_zero.torch_batches import TorchAlphaZeroTrainingBatch


@dataclass(frozen=True, slots=True)
class AlphaZeroTrainStepConfig:
    loss: AlphaZeroLossConfig = AlphaZeroLossConfig()
    max_grad_norm: float | None = None

    def __post_init__(self) -> None:
        if self.max_grad_norm is not None:
            _require_positive(self.max_grad_norm, "max_grad_norm")


@dataclass(frozen=True, slots=True)
class AlphaZeroTrainStepResult:
    total_loss: float
    policy_loss: float
    value_loss: float
    grad_norm: float | None
    batch_size: int
    action_size: int


def run_alpha_zero_training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: TorchAlphaZeroTrainingBatch,
    *,
    config: AlphaZeroTrainStepConfig | None = None,
) -> AlphaZeroTrainStepResult:
    resolved_config = AlphaZeroTrainStepConfig() if config is None else config

    model.train()
    optimizer.zero_grad(set_to_none=True)

    policy_logits, predicted_values = _unpack_model_output(cast(object, model(batch.observations)))
    losses = alpha_zero_loss(
        policy_logits,
        _normalize_predicted_values(predicted_values, batch.shape.batch_size),
        batch,
        config=resolved_config.loss,
    )
    torch.autograd.backward(losses.total_loss)

    grad_norm = None
    if resolved_config.max_grad_norm is not None:
        grad_norm = _scalar(
            torch.nn.utils.clip_grad_norm_(model.parameters(), resolved_config.max_grad_norm),
            "grad_norm",
        )

    optimizer.step()

    return AlphaZeroTrainStepResult(
        total_loss=_scalar(losses.total_loss, "total_loss"),
        policy_loss=_scalar(losses.policy_loss, "policy_loss"),
        value_loss=_scalar(losses.value_loss, "value_loss"),
        grad_norm=grad_norm,
        batch_size=batch.shape.batch_size,
        action_size=batch.shape.action_size,
    )


def _unpack_model_output(output: object) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(output, tuple) or len(output) != 2:
        raise TypeError("model output must be a tuple of policy logits and predicted values")

    policy_logits, predicted_values = output
    if not isinstance(policy_logits, torch.Tensor):
        raise TypeError("policy logits must be a tensor")
    if not isinstance(predicted_values, torch.Tensor):
        raise TypeError("predicted values must be a tensor")
    return policy_logits, predicted_values


def _normalize_predicted_values(predicted_values: torch.Tensor, batch_size: int) -> torch.Tensor:
    if predicted_values.ndim == 2 and tuple(predicted_values.shape) == (batch_size, 1):
        return predicted_values.squeeze(dim=1)
    return predicted_values


def _scalar(tensor: torch.Tensor, name: str) -> float:
    value = float(tensor.detach().cpu().item())
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _require_positive(value: float, name: str) -> None:
    if isinstance(value, bool) or value <= 0.0:
        raise ValueError(f"{name} must be positive")
