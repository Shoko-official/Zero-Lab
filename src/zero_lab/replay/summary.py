"""Replay summary helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from zero_lab.replay.jsonl import read_episodes


@dataclass(frozen=True, slots=True)
class ReplaySummary:
    episodes: int
    steps: int
    games: dict[str, int]
    outcomes: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "episodes": self.episodes,
            "games": self.games,
            "outcomes": self.outcomes,
            "steps": self.steps,
        }


def summarize_replay(path: Path) -> ReplaySummary:
    games: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    episode_count = 0
    step_count = 0

    for episode in read_episodes(path):
        episode_count += 1
        step_count += episode.length
        games[episode.game] += 1
        outcomes[str(episode.outcome)] += 1

    return ReplaySummary(
        episodes=episode_count,
        games=dict(sorted(games.items())),
        outcomes=dict(sorted(outcomes.items())),
        steps=step_count,
    )
