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
