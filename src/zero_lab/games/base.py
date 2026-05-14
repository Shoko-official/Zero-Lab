"""Contracts and helpers for deterministic turn-based games."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, cast, runtime_checkable

FIRST_PLAYER = 1
SECOND_PLAYER = -1
EMPTY = 0


class GameError(ValueError):
    """Base error for game-state failures."""


class IllegalMoveError(GameError):
    """Raised when an action cannot be applied to a state."""


class StateDecodingError(GameError):
    """Raised when serialized state data does not match the game schema."""


@runtime_checkable
class GameState(Protocol):
    @property
    def current_player(self) -> int:
        pass

    @property
    def action_size(self) -> int:
        pass

    @property
    def is_terminal(self) -> bool:
        pass

    def legal_action_mask(self) -> tuple[bool, ...]:
        pass

    def legal_actions(self) -> tuple[int, ...]:
        pass

    def apply(self, action: int) -> GameState:
        pass

    def canonical_observation(self) -> tuple[int, ...]:
        pass

    def outcome_for(self, player: int) -> int | None:
        pass

    def serialize(self) -> str:
        pass


class GameRules(Protocol):
    @property
    def name(self) -> str:
        pass

    @property
    def action_size(self) -> int:
        pass

    def reset(self, seed: int | None = None) -> GameState:
        pass

    def deserialize(self, serialized: str) -> GameState:
        pass


def validate_player(player: int) -> None:
    if isinstance(player, bool) or player not in (FIRST_PLAYER, SECOND_PLAYER):
        raise ValueError("player must be 1 or -1")


def opponent(player: int) -> int:
    validate_player(player)
    return -player


def validate_action(action: int, action_size: int) -> None:
    if isinstance(action, bool) or not isinstance(action, int):
        raise IllegalMoveError("action must be an integer")
    if action < 0 or action >= action_size:
        raise IllegalMoveError(f"action must be in [0, {action_size})")


def validate_board(values: Sequence[object], expected_size: int) -> tuple[int, ...]:
    if len(values) != expected_size:
        raise ValueError(f"board must contain {expected_size} cells")

    board: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("board cells must be integers")
        if value not in (SECOND_PLAYER, EMPTY, FIRST_PLAYER):
            raise ValueError("board cells must be -1, 0, or 1")
        board.append(value)

    return tuple(board)


def legal_actions_from_mask(mask: Sequence[bool]) -> tuple[int, ...]:
    return tuple(index for index, is_legal in enumerate(mask) if is_legal)


def encode_state_payload(
    *,
    board: Sequence[int],
    current_player: int,
    game: str,
    version: int = 1,
) -> str:
    validate_player(current_player)
    payload = {
        "board": list(board),
        "current_player": current_player,
        "game": game,
        "version": version,
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def decode_state_payload(
    serialized: str,
    *,
    expected_game: str,
    expected_version: int = 1,
) -> Mapping[str, object]:
    try:
        loaded: Any = json.loads(serialized)
    except json.JSONDecodeError as error:
        raise StateDecodingError("state is not valid JSON") from error

    if not isinstance(loaded, dict):
        raise StateDecodingError("state payload must be a JSON object")
    if loaded.get("game") != expected_game:
        raise StateDecodingError("state payload has the wrong game")
    if loaded.get("version") != expected_version:
        raise StateDecodingError("state payload has the wrong version")

    return cast(Mapping[str, object], loaded)


def read_int_field(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise StateDecodingError(f"{key} must be an integer")
    return value


def read_board_field(
    payload: Mapping[str, object],
    key: str,
    expected_size: int,
) -> tuple[int, ...]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise StateDecodingError(f"{key} must be an array")
    return validate_board(value, expected_size)


def read_string_field(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise StateDecodingError(f"{key} must be a string")
    return value
