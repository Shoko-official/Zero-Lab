"""JSONL replay storage."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from zero_lab.replay.records import EpisodeRecord


def append_episode(path: Path, episode: EpisodeRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(episode.to_dict(), separators=(",", ":"), sort_keys=True))
        handle.write("\n")


def read_episodes(path: Path) -> Iterator[EpisodeRecord]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                loaded = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on line {line_number}") from error
            if not isinstance(loaded, dict):
                raise ValueError(f"episode on line {line_number} must be a JSON object")
            yield EpisodeRecord.from_dict(loaded)
