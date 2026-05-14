from __future__ import annotations

import pytest

from zero_lab.games.base import FIRST_PLAYER, SECOND_PLAYER, IllegalMoveError
from zero_lab.games.toy.connect_four import COLUMNS, ROWS, ConnectFourGame, ConnectFourState


def test_connect_four_reset_has_all_columns_legal() -> None:
    state = ConnectFourGame().reset(seed=0)

    assert state.current_player == FIRST_PLAYER
    assert state.legal_action_mask() == (True,) * COLUMNS
    assert state.legal_actions() == tuple(range(COLUMNS))
    assert not state.is_terminal


def test_connect_four_applies_moves_from_bottom_of_column() -> None:
    state = ConnectFourState().apply(3)

    assert state.current_player == SECOND_PLAYER
    assert state.board[-COLUMNS + 3] == FIRST_PLAYER
    assert state.canonical_observation()[-COLUMNS + 3] == SECOND_PLAYER


def test_connect_four_rejects_full_column() -> None:
    state = ConnectFourState()
    for _ in range(ROWS):
        state = state.apply(0)

    assert state.legal_action_mask()[0] is False

    with pytest.raises(IllegalMoveError, match="full column"):
        state.apply(0)


def test_connect_four_detects_vertical_win() -> None:
    state = ConnectFourState()
    for action in (0, 1, 0, 1, 0, 1, 0):
        state = state.apply(action)

    assert state.is_terminal
    assert state.outcome_for(FIRST_PLAYER) == 1
    assert state.outcome_for(SECOND_PLAYER) == -1
    assert state.legal_actions() == ()


def test_connect_four_serializes_round_trip() -> None:
    game = ConnectFourGame()
    state = ConnectFourState().apply(2).apply(2).apply(3)

    restored = game.deserialize(state.serialize())

    assert restored == state
