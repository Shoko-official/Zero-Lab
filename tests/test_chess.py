from __future__ import annotations

import chess
import pytest

from zero_lab.games.base import FIRST_PLAYER, SECOND_PLAYER, IllegalMoveError
from zero_lab.games.chess import (
    ACTION_SIZE,
    ChessGame,
    ChessState,
    action_from_move,
    move_from_action,
)


def test_chess_reset_exposes_standard_legal_moves() -> None:
    state = ChessGame().reset(seed=0)

    assert state.current_player == FIRST_PLAYER
    assert state.action_size == ACTION_SIZE
    assert len(state.legal_actions()) == 20
    assert sum(state.legal_action_mask()) == 20
    assert not state.is_terminal


def test_chess_applies_encoded_legal_move() -> None:
    state = ChessState()
    action = action_from_move(state.to_board(), chess.Move.from_uci("e2e4"))

    next_state = state.apply(action)

    assert next_state.current_player == SECOND_PLAYER
    assert next_state.to_board().piece_at(chess.E4) == chess.Piece(chess.PAWN, chess.WHITE)


def test_chess_round_trips_all_starting_legal_moves() -> None:
    board = chess.Board()

    decoded_moves = {
        move_from_action(board, action_from_move(board, move))
        for move in board.legal_moves
    }

    assert decoded_moves == set(board.legal_moves)


def test_chess_rejects_illegal_action() -> None:
    state = ChessState()

    with pytest.raises(IllegalMoveError, match="legal"):
        state.apply(0)


def test_chess_canonical_observation_flips_for_black() -> None:
    state = ChessState()
    state = state.apply(action_from_move(state.to_board(), chess.Move.from_uci("e2e4")))
    observation = state.canonical_observation()

    assert len(observation) == 64
    assert observation[3] == 6
    assert observation[59] == -6


def test_chess_serializes_round_trip() -> None:
    game = ChessGame()
    state = ChessState()
    state = state.apply(action_from_move(state.to_board(), chess.Move.from_uci("e2e4")))
    state = state.apply(action_from_move(state.to_board(), chess.Move.from_uci("c7c5")))

    restored = game.deserialize(state.serialize())

    assert restored == state


def test_chess_detects_checkmate_outcome() -> None:
    state = ChessState()
    for uci in ("f2f3", "e7e5", "g2g4", "d8h4"):
        state = state.apply(action_from_move(state.to_board(), chess.Move.from_uci(uci)))

    assert state.is_terminal
    assert state.outcome_for(FIRST_PLAYER) == -1
    assert state.outcome_for(SECOND_PLAYER) == 1
    assert state.legal_actions() == ()


def test_chess_encodes_underpromotion_as_distinct_action() -> None:
    state = ChessState("8/P7/8/8/8/8/8/k6K w - - 0 1")
    board = state.to_board()
    queen_action = action_from_move(board, chess.Move.from_uci("a7a8q"))
    knight_action = action_from_move(board, chess.Move.from_uci("a7a8n"))

    assert queen_action in state.legal_actions()
    assert knight_action in state.legal_actions()
    assert queen_action != knight_action
