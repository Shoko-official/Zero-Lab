"""Chess adapter backed by python-chess rules."""

from __future__ import annotations

import json
from dataclasses import dataclass

import chess

from zero_lab.games.base import (
    FIRST_PLAYER,
    SECOND_PLAYER,
    IllegalMoveError,
    StateDecodingError,
    decode_state_payload,
    legal_actions_from_mask,
    read_string_field,
    validate_action,
    validate_player,
)

GAME_NAME = "chess"
ACTION_PLANES = 73
ACTION_SIZE = 64 * ACTION_PLANES
QUEEN_MOVE_PLANES = 56
KNIGHT_MOVE_PLANES = 8
UNDERPROMOTION_START_PLANE = QUEEN_MOVE_PLANES + KNIGHT_MOVE_PLANES

_QUEEN_DIRECTIONS = (
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
)
_DIRECTION_TO_INDEX = {direction: index for index, direction in enumerate(_QUEEN_DIRECTIONS)}
_KNIGHT_DELTAS = (
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
)
_KNIGHT_DELTA_TO_INDEX = {delta: index for index, delta in enumerate(_KNIGHT_DELTAS)}
_UNDERPROMOTION_PIECES = {
    chess.KNIGHT: 0,
    chess.BISHOP: 1,
    chess.ROOK: 2,
}
_UNDERPROMOTION_INDEX_TO_PIECE = {
    index: piece for piece, index in _UNDERPROMOTION_PIECES.items()
}
_RELATIVE_PROMOTION_FILES = (-1, 0, 1)
_PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.ROOK: 4,
    chess.QUEEN: 5,
    chess.KING: 6,
}


@dataclass(frozen=True, slots=True)
class ChessState:
    fen: str = chess.STARTING_FEN

    def __post_init__(self) -> None:
        board = self.to_board()
        object.__setattr__(self, "fen", board.fen())

    @property
    def current_player(self) -> int:
        return FIRST_PLAYER if self.to_board().turn == chess.WHITE else SECOND_PLAYER

    @property
    def action_size(self) -> int:
        return ACTION_SIZE

    @property
    def is_terminal(self) -> bool:
        return self.to_board().is_game_over(claim_draw=True)

    def legal_action_mask(self) -> tuple[bool, ...]:
        board = self.to_board()
        mask = [False] * ACTION_SIZE
        if board.is_game_over(claim_draw=True):
            return tuple(mask)

        for move in board.legal_moves:
            mask[action_from_move(board, move)] = True

        return tuple(mask)

    def legal_actions(self) -> tuple[int, ...]:
        return legal_actions_from_mask(self.legal_action_mask())

    def apply(self, action: int) -> ChessState:
        board = self.to_board()
        if board.is_game_over(claim_draw=True):
            raise IllegalMoveError("cannot apply an action to a terminal state")

        move = move_from_action(board, action)
        board.push(move)
        return ChessState(fen=board.fen())

    def canonical_observation(self) -> tuple[int, ...]:
        board = self.to_board()
        values: list[int] = []

        for rank in range(8):
            for file in range(8):
                square = _canonical_square(file=file, rank=rank, turn=board.turn)
                piece = board.piece_at(square)
                if piece is None:
                    values.append(0)
                    continue

                sign = 1 if piece.color == board.turn else -1
                values.append(sign * _PIECE_VALUES[piece.piece_type])

        return tuple(values)

    def outcome_for(self, player: int) -> int | None:
        validate_player(player)
        outcome = self.to_board().outcome(claim_draw=True)
        if outcome is None:
            return None
        if outcome.winner is None:
            return 0

        winner = FIRST_PLAYER if outcome.winner == chess.WHITE else SECOND_PLAYER
        return 1 if winner == player else -1

    def serialize(self) -> str:
        payload = {
            "fen": self.fen,
            "game": GAME_NAME,
            "version": 1,
        }
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)

    def to_board(self) -> chess.Board:
        try:
            return chess.Board(self.fen)
        except ValueError as error:
            raise StateDecodingError("invalid chess FEN") from error

    @classmethod
    def deserialize(cls, serialized: str) -> ChessState:
        payload = decode_state_payload(serialized, expected_game=GAME_NAME)
        return cls(fen=read_string_field(payload, "fen"))


@dataclass(frozen=True, slots=True)
class ChessGame:
    name: str = GAME_NAME
    action_size: int = ACTION_SIZE

    def reset(self, seed: int | None = None) -> ChessState:
        if seed is not None and (isinstance(seed, bool) or seed < 0):
            raise ValueError("seed must be a non-negative integer")
        return ChessState()

    def deserialize(self, serialized: str) -> ChessState:
        return ChessState.deserialize(serialized)


def action_from_move(board: chess.Board, move: chess.Move) -> int:
    if move not in board.legal_moves:
        raise IllegalMoveError("move is not legal in the current position")

    plane = _plane_from_move(board, move)
    return move.from_square * ACTION_PLANES + plane


def move_from_action(board: chess.Board, action: int) -> chess.Move:
    validate_action(action, ACTION_SIZE)
    from_square = action // ACTION_PLANES
    plane = action % ACTION_PLANES

    if plane < QUEEN_MOVE_PLANES:
        move = _queen_move_from_plane(board, from_square, plane)
    elif plane < UNDERPROMOTION_START_PLANE:
        move = _knight_move_from_plane(from_square, plane)
    else:
        move = _underpromotion_move_from_plane(board, from_square, plane)

    if move not in board.legal_moves:
        raise IllegalMoveError("action does not map to a legal chess move")

    return move


def _plane_from_move(board: chess.Board, move: chess.Move) -> int:
    if move.promotion in _UNDERPROMOTION_PIECES:
        return _underpromotion_plane(board, move)
    if _is_knight_move(move):
        return _knight_plane(move)
    return _queen_plane(move)


def _queen_plane(move: chess.Move) -> int:
    from_file = chess.square_file(move.from_square)
    from_rank = chess.square_rank(move.from_square)
    to_file = chess.square_file(move.to_square)
    to_rank = chess.square_rank(move.to_square)

    file_delta = to_file - from_file
    rank_delta = to_rank - from_rank
    distance = max(abs(file_delta), abs(rank_delta))

    if distance == 0 or distance > 7:
        raise IllegalMoveError("move cannot be encoded as a queen-like move")

    direction = (
        _unit(file_delta),
        _unit(rank_delta),
    )
    direction_index = _DIRECTION_TO_INDEX.get(direction)
    if direction_index is None:
        raise IllegalMoveError("move cannot be encoded as a queen-like move")

    return direction_index * 7 + distance - 1


def _is_knight_move(move: chess.Move) -> bool:
    from_file = chess.square_file(move.from_square)
    from_rank = chess.square_rank(move.from_square)
    to_file = chess.square_file(move.to_square)
    to_rank = chess.square_rank(move.to_square)
    return (abs(to_file - from_file), abs(to_rank - from_rank)) in ((1, 2), (2, 1))


def _knight_plane(move: chess.Move) -> int:
    from_file = chess.square_file(move.from_square)
    from_rank = chess.square_rank(move.from_square)
    to_file = chess.square_file(move.to_square)
    to_rank = chess.square_rank(move.to_square)
    delta = (to_file - from_file, to_rank - from_rank)
    try:
        return QUEEN_MOVE_PLANES + _KNIGHT_DELTA_TO_INDEX[delta]
    except KeyError as error:
        raise IllegalMoveError("move cannot be encoded as a knight move") from error


def _underpromotion_plane(board: chess.Board, move: chess.Move) -> int:
    piece = board.piece_at(move.from_square)
    if piece is None or piece.piece_type != chess.PAWN:
        raise IllegalMoveError("underpromotion must start from a pawn")

    from_file = chess.square_file(move.from_square)
    to_file = chess.square_file(move.to_square)
    file_delta = to_file - from_file
    relative_file_delta = file_delta if piece.color == chess.WHITE else -file_delta

    promotion = move.promotion
    if promotion is None:
        raise IllegalMoveError("underpromotion move must include a promotion piece")

    try:
        direction_index = _RELATIVE_PROMOTION_FILES.index(relative_file_delta)
        piece_index = _UNDERPROMOTION_PIECES[promotion]
    except (KeyError, ValueError) as error:
        raise IllegalMoveError("move cannot be encoded as an underpromotion") from error

    return UNDERPROMOTION_START_PLANE + direction_index * 3 + piece_index


def _queen_move_from_plane(board: chess.Board, from_square: int, plane: int) -> chess.Move:
    direction_index = plane // 7
    distance = plane % 7 + 1
    file_delta, rank_delta = _QUEEN_DIRECTIONS[direction_index]

    from_file = chess.square_file(from_square)
    from_rank = chess.square_rank(from_square)
    to_file = from_file + file_delta * distance
    to_rank = from_rank + rank_delta * distance

    if not _contains_square(to_file, to_rank):
        raise IllegalMoveError("action points outside the board")

    to_square = chess.square(to_file, to_rank)
    promotion = _queen_promotion_for(board, from_square, to_square)
    return chess.Move(from_square, to_square, promotion=promotion)


def _knight_move_from_plane(from_square: int, plane: int) -> chess.Move:
    knight_plane = plane - QUEEN_MOVE_PLANES
    file_delta, rank_delta = _KNIGHT_DELTAS[knight_plane]
    from_file = chess.square_file(from_square)
    from_rank = chess.square_rank(from_square)
    to_file = from_file + file_delta
    to_rank = from_rank + rank_delta

    if not _contains_square(to_file, to_rank):
        raise IllegalMoveError("knight action points outside the board")

    return chess.Move(from_square, chess.square(to_file, to_rank))


def _underpromotion_move_from_plane(
    board: chess.Board,
    from_square: int,
    plane: int,
) -> chess.Move:
    piece = board.piece_at(from_square)
    if piece is None or piece.piece_type != chess.PAWN:
        raise IllegalMoveError("underpromotion action must start from a pawn")

    promotion_plane = plane - UNDERPROMOTION_START_PLANE
    direction_index = promotion_plane // 3
    piece_index = promotion_plane % 3

    relative_file_delta = _RELATIVE_PROMOTION_FILES[direction_index]
    file_delta = relative_file_delta if piece.color == chess.WHITE else -relative_file_delta
    rank_delta = 1 if piece.color == chess.WHITE else -1

    from_file = chess.square_file(from_square)
    from_rank = chess.square_rank(from_square)
    to_file = from_file + file_delta
    to_rank = from_rank + rank_delta

    if not _contains_square(to_file, to_rank):
        raise IllegalMoveError("underpromotion action points outside the board")

    return chess.Move(
        from_square,
        chess.square(to_file, to_rank),
        promotion=_UNDERPROMOTION_INDEX_TO_PIECE[piece_index],
    )


def _queen_promotion_for(board: chess.Board, from_square: int, to_square: int) -> int | None:
    piece = board.piece_at(from_square)
    if piece is None or piece.piece_type != chess.PAWN:
        return None

    promotion_rank = 7 if piece.color == chess.WHITE else 0
    if chess.square_rank(to_square) != promotion_rank:
        return None

    return chess.QUEEN


def _canonical_square(*, file: int, rank: int, turn: bool) -> int:
    if turn == chess.WHITE:
        return chess.square(file, rank)
    return chess.square(7 - file, 7 - rank)


def _contains_square(file: int, rank: int) -> bool:
    return 0 <= file < 8 and 0 <= rank < 8


def _unit(value: int) -> int:
    if value < 0:
        return -1
    if value > 0:
        return 1
    return 0
