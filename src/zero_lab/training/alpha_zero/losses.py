"""AlphaZero policy and value losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

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


def _require_non_negative(value: float, name: str) -> None:
    if isinstance(value, bool) or value < 0.0:
        raise ValueError(f"{name} must be non-negative")


def _require_positive(value: float, name: str) -> None:
    if isinstance(value, bool) or value <= 0.0:
        raise ValueError(f"{name} must be positive")
