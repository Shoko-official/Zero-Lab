"""Streaming dataset helpers for AlphaZero training targets."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from zero_lab.replay.jsonl import read_episodes


@dataclass(frozen=True, slots=True)
class AlphaZeroSample:
    """Single training sample derived from a replay episode step."""

    action: int
    current_player: int
    game: str
    policy: tuple[float, ...]
    state: str
    value_target: float


def iter_alpha_zero_samples(paths: Path | Sequence[Path]) -> Iterator[AlphaZeroSample]:
    """Stream AlphaZero samples from one or more replay JSONL files."""

    replay_paths = (paths,) if isinstance(paths, Path) else paths
    for path in replay_paths:
        for episode in read_episodes(path):
            for step in episode.steps:
                yield AlphaZeroSample(
                    action=step.action,
                    current_player=step.current_player,
                    game=episode.game,
                    policy=step.policy,
                    state=step.state,
                    value_target=step.value_target,
                )
