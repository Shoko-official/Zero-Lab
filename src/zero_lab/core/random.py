"""Randomness controls for deterministic local runs."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class SeedReport:
    seed: int
    seeded_modules: tuple[str, ...]


def seed_python(seed: int) -> SeedReport:
    if isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    return SeedReport(seed=seed, seeded_modules=("python.random",))
