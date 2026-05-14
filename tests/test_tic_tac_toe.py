from __future__ import annotations

import pytest

from zero_lab.games.base import FIRST_PLAYER, SECOND_PLAYER, IllegalMoveError
from zero_lab.games.toy.tic_tac_toe import TicTacToeGame, TicTacToeState


def test_tic_tac_toe_reset_has_all_cells_legal() -> None:
    state = TicTacToeGame().reset(seed=0)

    assert state.current_player == FIRST_PLAYER
    assert state.legal_action_mask() == (True,) * 9
    assert state.legal_actions() == tuple(range(9))
    assert not state.is_terminal


def test_tic_tac_toe_applies_move_and_flips_perspective() -> None:
    state = TicTacToeState().apply(4)

    assert state.current_player == SECOND_PLAYER
    assert state.board[4] == FIRST_PLAYER
    assert state.canonical_observation()[4] == SECOND_PLAYER


def test_tic_tac_toe_rejects_occupied_cell() -> None:
    state = TicTacToeState().apply(4)

    with pytest.raises(IllegalMoveError, match="occupied"):
        state.apply(4)


def test_tic_tac_toe_detects_win_and_blocks_further_moves() -> None:
    state = TicTacToeState()
    for action in (0, 3, 1, 4, 2):
        state = state.apply(action)

    assert state.is_terminal
    assert state.outcome_for(FIRST_PLAYER) == 1
    assert state.outcome_for(SECOND_PLAYER) == -1
    assert state.legal_actions() == ()

    with pytest.raises(IllegalMoveError, match="terminal"):
        state.apply(8)


def test_tic_tac_toe_serializes_round_trip() -> None:
    game = TicTacToeGame()
    state = TicTacToeState().apply(0).apply(4).apply(8)

    restored = game.deserialize(state.serialize())

    assert restored == state
