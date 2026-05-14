"""Tic Tac Toe adapter."""

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

GAME_NAME = "tic_tac_toe"
BOARD_SIZE = 9
WIN_LINES = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)


@dataclass(frozen=True, slots=True)
class TicTacToeState:
    board: tuple[int, ...] = (EMPTY,) * BOARD_SIZE
    current_player: int = FIRST_PLAYER

    def __post_init__(self) -> None:
        object.__setattr__(self, "board", validate_board(self.board, BOARD_SIZE))
        validate_player(self.current_player)

    @property
    def action_size(self) -> int:
        return BOARD_SIZE

    @property
    def is_terminal(self) -> bool:
        return _winner(self.board) is not None or all(cell != EMPTY for cell in self.board)

    def legal_action_mask(self) -> tuple[bool, ...]:
        if self.is_terminal:
            return (False,) * BOARD_SIZE
        return tuple(cell == EMPTY for cell in self.board)

    def legal_actions(self) -> tuple[int, ...]:
        return legal_actions_from_mask(self.legal_action_mask())

    def apply(self, action: int) -> TicTacToeState:
        validate_action(action, BOARD_SIZE)
        if self.is_terminal:
            raise IllegalMoveError("cannot apply an action to a terminal state")
        if self.board[action] != EMPTY:
            raise IllegalMoveError("cannot apply an action to an occupied cell")

        next_board = list(self.board)
        next_board[action] = self.current_player
        return TicTacToeState(board=tuple(next_board), current_player=opponent(self.current_player))

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
    def deserialize(cls, serialized: str) -> TicTacToeState:
        payload = decode_state_payload(serialized, expected_game=GAME_NAME)
        current_player = read_int_field(payload, "current_player")
        board = read_board_field(payload, "board", BOARD_SIZE)
        return cls(board=board, current_player=current_player)


@dataclass(frozen=True, slots=True)
class TicTacToeGame:
    name: str = GAME_NAME
    action_size: int = BOARD_SIZE

    def reset(self, seed: int | None = None) -> TicTacToeState:
        if seed is not None and (isinstance(seed, bool) or seed < 0):
            raise ValueError("seed must be a non-negative integer")
        return TicTacToeState()

    def deserialize(self, serialized: str) -> TicTacToeState:
        return TicTacToeState.deserialize(serialized)


def _winner(board: tuple[int, ...]) -> int | None:
    for first, second, third in WIN_LINES:
        value = board[first]
        if value != EMPTY and value == board[second] == board[third]:
            return value
    return None
