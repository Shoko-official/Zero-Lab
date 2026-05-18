"""Versioned replay records for self-play episodes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Self, cast

from zero_lab.games.base import FIRST_PLAYER, SECOND_PLAYER

REPLAY_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class EpisodeStep:
    state: str
    current_player: int
    action: int
    policy: tuple[float, ...]
    value_target: float

    def __post_init__(self) -> None:
        if not self.state:
            raise ValueError("state must not be empty")
        if self.current_player not in (FIRST_PLAYER, SECOND_PLAYER):
            raise ValueError("current_player must be 1 or -1")
        if isinstance(self.action, bool) or self.action < 0:
            raise ValueError("action must be a non-negative integer")
        if not self.policy:
            raise ValueError("policy must not be empty")
        if abs(self.value_target) > 1.0:
            raise ValueError("value_target must be in [-1, 1]")

        policy_total = sum(self.policy)
        if any(value < 0.0 for value in self.policy):
            raise ValueError("policy values must be non-negative")
        if abs(policy_total - 1.0) > 1e-6:
            raise ValueError("policy must sum to 1")

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "current_player": self.current_player,
            "policy": list(self.policy),
            "state": self.state,
            "value_target": self.value_target,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> Self:
        return cls(
            action=_read_int(values, "action"),
            current_player=_read_int(values, "current_player"),
            policy=tuple(_read_float_sequence(values, "policy")),
            state=_read_string(values, "state"),
            value_target=_read_float(values, "value_target"),
        )


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    game: str
    terminal_state: str
    outcome: int | None
    steps: tuple[EpisodeStep, ...]
    schema_version: int = REPLAY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REPLAY_SCHEMA_VERSION:
            raise ValueError("unsupported replay schema version")
        if not self.game:
            raise ValueError("game must not be empty")
        if not self.terminal_state:
            raise ValueError("terminal_state must not be empty")
        if self.outcome is not None and self.outcome not in (-1, 0, 1):
            raise ValueError("outcome must be -1, 0, 1, or None")

    @property
    def length(self) -> int:
        return len(self.steps)

    def to_dict(self) -> dict[str, object]:
        return {
            "game": self.game,
            "outcome": self.outcome,
            "schema_version": self.schema_version,
            "steps": [step.to_dict() for step in self.steps],
            "terminal_state": self.terminal_state,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> Self:
        steps = values.get("steps")
        if not isinstance(steps, list):
            raise ValueError("steps must be a list")

        outcome = values.get("outcome")
        if outcome is not None and (isinstance(outcome, bool) or not isinstance(outcome, int)):
            raise ValueError("outcome must be an integer or null")

        return cls(
            game=_read_string(values, "game"),
            outcome=outcome,
            schema_version=_read_int(values, "schema_version"),
            steps=tuple(EpisodeStep.from_dict(_read_mapping(step)) for step in steps),
            terminal_state=_read_string(values, "terminal_state"),
        )


def _read_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError("expected mapping")
    return cast(Mapping[str, object], value)


def _read_string(values: Mapping[str, object], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _read_int(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _read_float(values: Mapping[str, object], key: str) -> float:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number")
    return float(value)


def _read_float_sequence(values: Mapping[str, object], key: str) -> Sequence[float]:
    value = values.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"{key} must contain only numbers")
    return tuple(float(item) for item in value)
