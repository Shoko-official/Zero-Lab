"""AlphaZero policy and value losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

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


def _require_float_tensor(tensor: torch.Tensor, name: str) -> None:
    if not torch.is_floating_point(tensor):
        raise TypeError(f"{name} must be a floating point tensor")


def _require_finite(tensor: torch.Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} must contain only finite values")


def _require_non_negative(value: float, name: str) -> None:
    if isinstance(value, bool) or value < 0.0:
        raise ValueError(f"{name} must be non-negative")


def _require_positive(value: float, name: str) -> None:
    if isinstance(value, bool) or value <= 0.0:
        raise ValueError(f"{name} must be positive")
