from __future__ import annotations

import random

from zero_lab.core.random import seed_python


def test_seed_python_makes_random_module_deterministic() -> None:
    seed_python(42)
    first = random.random()

    seed_python(42)
    second = random.random()

    assert first == second
