"""Configuration for AlphaZero-style MCTS."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MCTSSearchConfig:
    simulations: int = 64
    pb_c_base: float = 19652.0
    pb_c_init: float = 1.25

    def __post_init__(self) -> None:
        if isinstance(self.simulations, bool) or self.simulations <= 0:
            raise ValueError("simulations must be a positive integer")
        if self.pb_c_base <= 0.0:
            raise ValueError("pb_c_base must be positive")
        if self.pb_c_init < 0.0:
            raise ValueError("pb_c_init must be non-negative")
