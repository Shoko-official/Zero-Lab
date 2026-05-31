"""Chess showcase match runner."""

from __future__ import annotations

import random
from dataclasses import dataclass

from zero_lab.evaluation.agents import EvaluationAgent
from zero_lab.games.base import FIRST_PLAYER, SECOND_PLAYER
from zero_lab.games.chess import ChessGame, ChessState, move_from_action


@dataclass(frozen=True, slots=True)
class ChessMatchConfig:
    seed: int = 0
    games_per_side: int = 1
    max_plies: int = 80

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if isinstance(self.games_per_side, bool) or self.games_per_side <= 0:
            raise ValueError("games_per_side must be a positive integer")
        if isinstance(self.max_plies, bool) or self.max_plies <= 0:
            raise ValueError("max_plies must be a positive integer")

    def to_dict(self) -> dict[str, int]:
        return {
            "games_per_side": self.games_per_side,
            "max_plies": self.max_plies,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class ChessMoveRecord:
    ply: int
    player: int
    action: int
    uci: str
    fen_after: str

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "fen_after": self.fen_after,
            "player": self.player,
            "ply": self.ply,
            "uci": self.uci,
        }


@dataclass(frozen=True, slots=True)
class ChessGameRecord:
    seed: int
    white_agent: str
    black_agent: str
    moves: tuple[ChessMoveRecord, ...]
    final_fen: str
    terminal: bool
    outcome_for_white: int | None
    winner: str | None
    termination: str

    @property
    def plies(self) -> int:
        return len(self.moves)

    def to_dict(self) -> dict[str, object]:
        return {
            "black_agent": self.black_agent,
            "final_fen": self.final_fen,
            "game": ChessGame().name,
            "moves": [move.to_dict() for move in self.moves],
            "outcome_for_white": self.outcome_for_white,
            "plies": self.plies,
            "seed": self.seed,
            "terminal": self.terminal,
            "termination": self.termination,
            "white_agent": self.white_agent,
            "winner": self.winner,
        }


def run_chess_matches(
    *,
    white_agent: EvaluationAgent,
    black_agent: EvaluationAgent,
    config: ChessMatchConfig | None = None,
) -> tuple[ChessGameRecord, ...]:
    active_config = ChessMatchConfig() if config is None else config
    records: list[ChessGameRecord] = []
    match_index = 0
    for _ in range(active_config.games_per_side):
        records.append(
            run_chess_match(
                white_agent=white_agent,
                black_agent=black_agent,
                seed=active_config.seed + match_index,
                max_plies=active_config.max_plies,
            )
        )
        match_index += 1
        records.append(
            run_chess_match(
                white_agent=black_agent,
                black_agent=white_agent,
                seed=active_config.seed + match_index,
                max_plies=active_config.max_plies,
            )
        )
        match_index += 1
    return tuple(records)


def run_chess_match(
    *,
    white_agent: EvaluationAgent,
    black_agent: EvaluationAgent,
    seed: int,
    max_plies: int,
) -> ChessGameRecord:
    if isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(max_plies, bool) or max_plies <= 0:
        raise ValueError("max_plies must be a positive integer")

    rng = random.Random(seed)
    state: ChessState = ChessGame().reset(seed=seed)
    agents_by_player = {
        FIRST_PLAYER: white_agent,
        SECOND_PLAYER: black_agent,
    }
    moves: list[ChessMoveRecord] = []

    while not state.is_terminal and len(moves) < max_plies:
        board = state.to_board()
        agent = agents_by_player[state.current_player]
        action = agent.select_action(state, rng=rng)
        if action not in state.legal_actions():
            raise ValueError(f"{agent.name} selected illegal action {action}")
        uci = move_from_action(board, action).uci()
        current_player = state.current_player
        state = state.apply(action)
        moves.append(
            ChessMoveRecord(
                ply=len(moves) + 1,
                player=current_player,
                action=action,
                uci=uci,
                fen_after=state.fen,
            )
        )

    outcome = state.outcome_for(FIRST_PLAYER) if state.is_terminal else None
    return ChessGameRecord(
        seed=seed,
        white_agent=white_agent.name,
        black_agent=black_agent.name,
        moves=tuple(moves),
        final_fen=state.fen,
        terminal=state.is_terminal,
        outcome_for_white=outcome,
        winner=_winner_name(
            outcome_for_white=outcome,
            white_agent=white_agent,
            black_agent=black_agent,
        ),
        termination="terminal" if state.is_terminal else "max_plies",
    )


def _winner_name(
    *,
    outcome_for_white: int | None,
    white_agent: EvaluationAgent,
    black_agent: EvaluationAgent,
) -> str | None:
    if outcome_for_white == 1:
        return white_agent.name
    if outcome_for_white == -1:
        return black_agent.name
    return None
