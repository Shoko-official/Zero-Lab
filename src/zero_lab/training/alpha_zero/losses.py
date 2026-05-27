"""AlphaZero policy and value losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from zero_lab.training.alpha_zero.torch_batches import TorchAlphaZeroTrainingBatch

ValueLossKind = Literal["mse", "huber"]


@dataclass(frozen=True, slots=True)
class AlphaZeroLossConfig:
    policy_weight: float = 1.0
    value_weight: float = 1.0
    value_loss: ValueLossKind = "mse"
    huber_delta: float = 1.0

    def __post_init__(self) -> None:
        _require_non_negative(self.policy_weight, "policy_weight")
        _require_non_negative(self.value_weight, "value_weight")
        _require_positive(self.huber_delta, "huber_delta")
        if self.value_loss not in ("mse", "huber"):
            raise ValueError("value_loss must be 'mse' or 'huber'")


@dataclass(frozen=True, slots=True)
class AlphaZeroLoss:
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    total_loss: torch.Tensor


def alpha_zero_policy_loss(
    policy_logits: torch.Tensor,
    target_policies: torch.Tensor,
) -> torch.Tensor:
    _require_float_tensor(policy_logits, "policy_logits")
    _require_float_tensor(target_policies, "target_policies")
    if policy_logits.ndim != 2:
        raise ValueError("policy_logits must be a rank-2 tensor")
    if target_policies.shape != policy_logits.shape:
        raise ValueError("target_policies shape must match policy_logits")
    if target_policies.device != policy_logits.device:
        raise ValueError("target_policies must be on the same device as policy_logits")
    _require_finite(policy_logits, "policy_logits")
    _require_finite(target_policies, "target_policies")
    _require_probability_targets(target_policies)
    return F.cross_entropy(policy_logits, target_policies, reduction="mean")


def alpha_zero_value_loss(
    predicted_values: torch.Tensor,
    target_values: torch.Tensor,
    *,
    loss_kind: ValueLossKind = "mse",
    huber_delta: float = 1.0,
) -> torch.Tensor:
    _require_float_tensor(predicted_values, "predicted_values")
    _require_float_tensor(target_values, "target_values")
    if predicted_values.ndim != 1:
        raise ValueError("predicted_values must be a rank-1 tensor")
    if target_values.shape != predicted_values.shape:
        raise ValueError("target_values shape must match predicted_values")
    if target_values.device != predicted_values.device:
        raise ValueError("target_values must be on the same device as predicted_values")
    _require_finite(predicted_values, "predicted_values")
    _require_finite(target_values, "target_values")
    if loss_kind == "mse":
        return F.mse_loss(predicted_values, target_values, reduction="mean")
    if loss_kind == "huber":
        _require_positive(huber_delta, "huber_delta")
        return F.huber_loss(
            predicted_values,
            target_values,
            reduction="mean",
            delta=huber_delta,
        )
    raise ValueError("loss_kind must be 'mse' or 'huber'")


def alpha_zero_loss(
    policy_logits: torch.Tensor,
    predicted_values: torch.Tensor,
    batch: TorchAlphaZeroTrainingBatch,
    *,
    config: AlphaZeroLossConfig | None = None,
) -> AlphaZeroLoss:
    resolved_config = AlphaZeroLossConfig() if config is None else config
    policy = alpha_zero_policy_loss(policy_logits, batch.target_policies)
    value = alpha_zero_value_loss(
        predicted_values,
        batch.target_values,
        loss_kind=resolved_config.value_loss,
        huber_delta=resolved_config.huber_delta,
    )
    total = resolved_config.policy_weight * policy + resolved_config.value_weight * value
    return AlphaZeroLoss(
        policy_loss=policy,
        value_loss=value,
        total_loss=total,
    )


def _require_float_tensor(tensor: torch.Tensor, name: str) -> None:
    if not torch.is_floating_point(tensor):
        raise TypeError(f"{name} must be a floating point tensor")


def _require_finite(tensor: torch.Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} must contain only finite values")


def _require_probability_targets(targets: torch.Tensor) -> None:
    if (targets < 0.0).any().item():
        raise ValueError("target_policies must contain only non-negative values")
    row_sums = targets.sum(dim=1)
    expected = torch.ones_like(row_sums)
    if not torch.allclose(row_sums, expected, atol=1e-6, rtol=0.0):
        raise ValueError("target_policies must sum to 1")


def _require_non_negative(value: float, name: str) -> None:
    if isinstance(value, bool) or value < 0.0:
        raise ValueError(f"{name} must be non-negative")


def _require_positive(value: float, name: str) -> None:
    if isinstance(value, bool) or value <= 0.0:
        raise ValueError(f"{name} must be positive")
