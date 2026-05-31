from __future__ import annotations

import json

import chess

from zero_lab.evaluation import (
    ChessBaselineConfig,
    ChessMatchConfig,
    RandomLegalMoveAgent,
    run_chess_baseline_evaluation,
    run_chess_match,
    run_chess_matches,
)


def test_chess_match_runner_records_legal_uci_moves() -> None:
    record = run_chess_match(
        white_agent=RandomLegalMoveAgent(),
        black_agent=RandomLegalMoveAgent(),
        seed=19,
        max_plies=8,
    )

    board = chess.Board()
    for move_record in record.moves:
        move = chess.Move.from_uci(move_record.uci)
        assert move in board.legal_moves
        board.push(move)

    assert record.seed == 19
    assert record.plies == 8
    assert record.final_fen == board.fen()
    assert record.termination == "max_plies"


def test_chess_match_runner_is_reproducible_with_seed() -> None:
    first = run_chess_match(
        white_agent=RandomLegalMoveAgent(),
        black_agent=RandomLegalMoveAgent(),
        seed=23,
        max_plies=6,
    )
    second = run_chess_match(
        white_agent=RandomLegalMoveAgent(),
        black_agent=RandomLegalMoveAgent(),
        seed=23,
        max_plies=6,
    )

    assert first == second


def test_chess_matches_alternate_colors_and_seeds() -> None:
    records = run_chess_matches(
        white_agent=RandomLegalMoveAgent(label="agent_a"),
        black_agent=RandomLegalMoveAgent(label="agent_b"),
        config=ChessMatchConfig(seed=31, games_per_side=1, max_plies=4),
    )

    assert [record.seed for record in records] == [31, 32]
    assert [(record.white_agent, record.black_agent) for record in records] == [
        ("agent_a", "agent_b"),
        ("agent_b", "agent_a"),
    ]


def test_chess_baseline_evaluation_builds_public_report() -> None:
    report = run_chess_baseline_evaluation(
        ChessBaselineConfig(seed=41, games_per_side=1, max_plies=4, simulations=1)
    )
    payload = report.to_dict()

    assert payload["game"] == "chess"
    assert payload["seeds"] == [41, 42]
    scores = payload["scores"]
    games = payload["games"]

    assert isinstance(scores, dict)
    assert isinstance(games, list)
    assert set(scores) == {"random_legal", "uniform_search"}
    assert len(games) == 2
    assert "No chess training" in str(payload["limitations"])


def test_chess_baseline_report_json_is_stable() -> None:
    report = run_chess_baseline_evaluation(
        ChessBaselineConfig(seed=43, games_per_side=1, max_plies=2, simulations=1)
    )

    payload = json.loads(report.to_json())

    assert payload["config"]["seed"] == 43
    assert payload["config"]["agents"][1]["simulations"] == 1
    assert payload["games"][0]["moves"][0]["uci"]
