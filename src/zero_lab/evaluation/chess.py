"""Chess showcase match runner."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

from zero_lab.evaluation.agents import EvaluationAgent, RandomLegalMoveAgent, UniformSearchAgent
from zero_lab.evaluation.reports import MatchScore
from zero_lab.games.base import FIRST_PLAYER, SECOND_PLAYER
from zero_lab.games.chess import ChessGame, ChessState, move_from_action

DEFAULT_CHESS_SHOWCASE_LIMITATIONS = (
    "No chess training is performed by this showcase.",
    "The report covers fixed-seed baseline games only.",
    "Games can end by max_plies before a chess terminal result.",
)


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
class ChessBaselineConfig:
    seed: int = 0
    games_per_side: int = 1
    max_plies: int = 24
    simulations: int = 4

    def __post_init__(self) -> None:
        ChessMatchConfig(
            seed=self.seed,
            games_per_side=self.games_per_side,
            max_plies=self.max_plies,
        )
        if isinstance(self.simulations, bool) or self.simulations <= 0:
            raise ValueError("simulations must be a positive integer")

    def to_match_config(self) -> ChessMatchConfig:
        return ChessMatchConfig(
            seed=self.seed,
            games_per_side=self.games_per_side,
            max_plies=self.max_plies,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "agents": [
                {"name": "random_legal", "role": "agent_one"},
                {
                    "name": "uniform_search",
                    "role": "agent_two",
                    "simulations": self.simulations,
                    "temperature": 0.0,
                },
            ],
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


@dataclass(frozen=True, slots=True)
class ChessShowcaseReport:
    config: ChessBaselineConfig
    games: tuple[ChessGameRecord, ...]
    scores: dict[str, MatchScore]
    limitations: tuple[str, ...] = DEFAULT_CHESS_SHOWCASE_LIMITATIONS

    def to_dict(self) -> dict[str, object]:
        return {
            "config": self.config.to_dict(),
            "game": ChessGame().name,
            "games": [game.to_dict() for game in self.games],
            "limitations": list(self.limitations),
            "scores": {
                agent: score.to_dict()
                for agent, score in sorted(self.scores.items())
            },
            "seeds": [game.seed for game in self.games],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def run_chess_baseline_evaluation(
    config: ChessBaselineConfig | None = None,
) -> ChessShowcaseReport:
    active_config = ChessBaselineConfig() if config is None else config
    games = run_chess_matches(
        white_agent=RandomLegalMoveAgent(),
        black_agent=UniformSearchAgent(simulations=active_config.simulations),
        config=active_config.to_match_config(),
    )
    return ChessShowcaseReport(
        config=active_config,
        games=games,
        scores=_score_chess_games(games),
    )


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


def _score_chess_games(games: tuple[ChessGameRecord, ...]) -> dict[str, MatchScore]:
    score_values = {
        agent: {"draws": 0, "losses": 0, "unfinished": 0, "wins": 0}
        for agent in _agent_names(games)
    }
    for game in games:
        white_score = score_values[game.white_agent]
        black_score = score_values[game.black_agent]
        if not game.terminal:
            white_score["unfinished"] += 1
            black_score["unfinished"] += 1
        elif game.outcome_for_white == 0:
            white_score["draws"] += 1
            black_score["draws"] += 1
        elif game.outcome_for_white == 1:
            white_score["wins"] += 1
            black_score["losses"] += 1
        elif game.outcome_for_white == -1:
            white_score["losses"] += 1
            black_score["wins"] += 1
        else:
            raise ValueError("terminal chess game must have an outcome")

    return {
        agent: MatchScore(
            draws=values["draws"],
            losses=values["losses"],
            unfinished=values["unfinished"],
            wins=values["wins"],
        )
        for agent, values in score_values.items()
    }


def _agent_names(games: tuple[ChessGameRecord, ...]) -> tuple[str, ...]:
    names: list[str] = []
    for game in games:
        if game.white_agent not in names:
            names.append(game.white_agent)
        if game.black_agent not in names:
            names.append(game.black_agent)
    return tuple(names)
