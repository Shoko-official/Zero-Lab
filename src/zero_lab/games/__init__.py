"""Game adapters and game-state contracts."""

from zero_lab.games.base import GameRules, GameState, IllegalMoveError
from zero_lab.games.chess import ChessGame, ChessState
from zero_lab.games.toy import ConnectFourGame, ConnectFourState, TicTacToeGame, TicTacToeState

__all__ = [
    "ChessGame",
    "ChessState",
    "ConnectFourGame",
    "ConnectFourState",
    "GameRules",
    "GameState",
    "IllegalMoveError",
    "TicTacToeGame",
    "TicTacToeState",
]
