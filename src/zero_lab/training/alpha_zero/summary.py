"""Replay batch summary helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from zero_lab.games import GameRules
from zero_lab.replay import AlphaZeroSample, iter_alpha_zero_samples
from zero_lab.training.alpha_zero.replay import build_alpha_zero_training_batch


@dataclass(frozen=True, slots=True)
class AlphaZeroBatchSummary:
    source_samples: int
    emitted_samples: int
    batches: int
    batch_size: int
    drop_remainder: bool
    remainder_samples: int
    action_sizes: tuple[int, ...]
    observation_sizes: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "action_sizes": list(self.action_sizes),
            "batch_size": self.batch_size,
            "batches": self.batches,
            "drop_remainder": self.drop_remainder,
            "emitted_samples": self.emitted_samples,
            "observation_sizes": list(self.observation_sizes),
            "remainder_samples": self.remainder_samples,
            "source_samples": self.source_samples,
        }


def summarize_alpha_zero_training_batches(
    paths: Path | Sequence[Path],
    *,
    games: Mapping[str, GameRules],
    batch_size: int,
    drop_remainder: bool = False,
) -> AlphaZeroBatchSummary:
    if isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    source_samples = 0
    emitted_samples = 0
    batches = 0
    remainder_samples = 0
    action_sizes: set[int] = set()
    observation_sizes: set[int] = set()
    batch_samples: list[AlphaZeroSample] = []

    for sample in iter_alpha_zero_samples(paths):
        source_samples += 1
        batch_samples.append(sample)
        if len(batch_samples) == batch_size:
            batch = build_alpha_zero_training_batch(batch_samples, games=games)
            batches += 1
            emitted_samples += batch.shape.batch_size
            action_sizes.add(batch.shape.action_size)
            observation_sizes.add(batch.shape.observation_size)
            batch_samples = []

    if batch_samples:
        if drop_remainder:
            remainder_samples = len(batch_samples)
        else:
            batch = build_alpha_zero_training_batch(batch_samples, games=games)
            batches += 1
            emitted_samples += batch.shape.batch_size
            action_sizes.add(batch.shape.action_size)
            observation_sizes.add(batch.shape.observation_size)

    return AlphaZeroBatchSummary(
        source_samples=source_samples,
        emitted_samples=emitted_samples,
        batches=batches,
        batch_size=batch_size,
        drop_remainder=drop_remainder,
        remainder_samples=remainder_samples,
        action_sizes=tuple(sorted(action_sizes)),
        observation_sizes=tuple(sorted(observation_sizes)),
    )
