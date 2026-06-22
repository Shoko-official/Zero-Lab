"""Small PyTorch modules for AlphaZero training smoke paths."""

from __future__ import annotations

import torch


class LinearAlphaZeroModel(torch.nn.Module):
    def __init__(self, observation_size: int, action_size: int) -> None:
        super().__init__()
        _require_positive_integer(observation_size, "observation_size")
        _require_positive_integer(action_size, "action_size")
        self.policy = torch.nn.Linear(observation_size, action_size)
        self.value = torch.nn.Linear(observation_size, 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.policy(observations), self.value(observations).squeeze(dim=1)


def _require_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
