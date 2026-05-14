"""Small deterministic games used for correctness tests."""

from zero_lab.games.toy.connect_four import ConnectFourGame, ConnectFourState
from zero_lab.games.toy.tic_tac_toe import TicTacToeGame, TicTacToeState

__all__ = [
    "ConnectFourGame",
    "ConnectFourState",
    "TicTacToeGame",
    "TicTacToeState",
]
