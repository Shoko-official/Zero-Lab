"""Connect Four adapter."""

from __future__ import annotations

from dataclasses import dataclass

from zero_lab.games.base import (
    EMPTY,
    FIRST_PLAYER,
    IllegalMoveError,
    decode_state_payload,
    encode_state_payload,
    legal_actions_from_mask,
    opponent,
    read_board_field,
    read_int_field,
    validate_action,
    validate_board,
    validate_player,
)

GAME_NAME = "connect_four"
ROWS = 6
COLUMNS = 7
BOARD_SIZE = ROWS * COLUMNS
DIRECTIONS = (
    (0, 1),
    (1, 0),
    (1, 1),
    (1, -1),
)


@dataclass(frozen=True, slots=True)
class ConnectFourState:
    board: tuple[int, ...] = (EMPTY,) * BOARD_SIZE
    current_player: int = FIRST_PLAYER

    def __post_init__(self) -> None:
        object.__setattr__(self, "board", validate_board(self.board, BOARD_SIZE))
        validate_player(self.current_player)

    @property
    def action_size(self) -> int:
        return COLUMNS

    @property
    def is_terminal(self) -> bool:
        return _winner(self.board) is not None or all(cell != EMPTY for cell in self.board)

    def legal_action_mask(self) -> tuple[bool, ...]:
        if self.is_terminal:
            return (False,) * COLUMNS
        return tuple(self.board[_index(0, column)] == EMPTY for column in range(COLUMNS))

    def legal_actions(self) -> tuple[int, ...]:
        return legal_actions_from_mask(self.legal_action_mask())

    def apply(self, action: int) -> ConnectFourState:
        validate_action(action, COLUMNS)
        if self.is_terminal:
            raise IllegalMoveError("cannot apply an action to a terminal state")
        if self.board[_index(0, action)] != EMPTY:
            raise IllegalMoveError("cannot apply an action to a full column")

        next_board = list(self.board)
        row = _lowest_empty_row(self.board, action)
        next_board[_index(row, action)] = self.current_player
        return ConnectFourState(
            board=tuple(next_board),
            current_player=opponent(self.current_player),
        )

    def canonical_observation(self) -> tuple[int, ...]:
        return tuple(cell * self.current_player for cell in self.board)

    def outcome_for(self, player: int) -> int | None:
        validate_player(player)
        winner = _winner(self.board)
        if winner is None and not self.is_terminal:
            return None
        if winner is None:
            return 0
        return 1 if winner == player else -1

    def serialize(self) -> str:
        return encode_state_payload(
            board=self.board,
            current_player=self.current_player,
            game=GAME_NAME,
        )

    @classmethod
    def deserialize(cls, serialized: str) -> ConnectFourState:
        payload = decode_state_payload(serialized, expected_game=GAME_NAME)
        current_player = read_int_field(payload, "current_player")
        board = read_board_field(payload, "board", BOARD_SIZE)
        return cls(board=board, current_player=current_player)


@dataclass(frozen=True, slots=True)
class ConnectFourGame:
    name: str = GAME_NAME
    action_size: int = COLUMNS

    def reset(self, seed: int | None = None) -> ConnectFourState:
        if seed is not None and (isinstance(seed, bool) or seed < 0):
            raise ValueError("seed must be a non-negative integer")
        return ConnectFourState()

    def deserialize(self, serialized: str) -> ConnectFourState:
        return ConnectFourState.deserialize(serialized)


def _index(row: int, column: int) -> int:
    return row * COLUMNS + column


def _lowest_empty_row(board: tuple[int, ...], column: int) -> int:
    for row in range(ROWS - 1, -1, -1):
        if board[_index(row, column)] == EMPTY:
            return row
    raise IllegalMoveError("column is full")


def _winner(board: tuple[int, ...]) -> int | None:
    for row in range(ROWS):
        for column in range(COLUMNS):
            value = board[_index(row, column)]
            if value == EMPTY:
                continue
            if _has_line_from(board, row, column, value):
                return value
    return None


def _has_line_from(board: tuple[int, ...], row: int, column: int, value: int) -> bool:
    for row_delta, column_delta in DIRECTIONS:
        count = 0
        for offset in range(4):
            next_row = row + row_delta * offset
            next_column = column + column_delta * offset
            if not _contains(next_row, next_column):
                break
            if board[_index(next_row, next_column)] != value:
                break
            count += 1
        if count == 4:
            return True
    return False


def _contains(row: int, column: int) -> bool:
    return 0 <= row < ROWS and 0 <= column < COLUMNS
