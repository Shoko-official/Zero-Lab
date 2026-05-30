from __future__ import annotations

import json

import pytest

from zero_lab.evaluation import (
    MatchConfig,
    RandomLegalMoveAgent,
    UniformSearchAgent,
    run_head_to_head,
    summarize_match_results,
)
from zero_lab.games.toy import ConnectFourGame, TicTacToeGame


def test_match_report_contains_json_ready_summary() -> None:
    config = MatchConfig(seed=31, games_per_side=1)
    results = run_head_to_head(
        games=(TicTacToeGame(), ConnectFourGame()),
        first_agent=RandomLegalMoveAgent(),
        second_agent=UniformSearchAgent(simulations=4),
        config=config,
    )

    report = summarize_match_results(results, config=config)
    payload = report.to_dict()

    assert payload["config"] == {"games_per_side": 1, "max_moves": 512, "seed": 31}
    assert payload["games"] == ["tic_tac_toe", "connect_four"]
    assert payload["seeds"] == [31, 32, 33, 34]
    scores = payload["scores"]
    matches = payload["matches"]

    assert isinstance(scores, dict)
    assert isinstance(matches, list)
    assert set(scores) == {"random_legal", "uniform_search"}
    assert len(matches) == 4
    assert "Elo ratings" in str(payload["limitations"])


def test_match_report_json_is_stable() -> None:
    config = MatchConfig(seed=5, games_per_side=1)
    results = run_head_to_head(
        games=(TicTacToeGame(),),
        first_agent=RandomLegalMoveAgent(),
        second_agent=UniformSearchAgent(simulations=4),
        config=config,
    )

    payload = json.loads(summarize_match_results(results, config=config).to_json())

    assert payload["config"]["seed"] == 5
    assert payload["matches"][0]["seed"] == 5


def test_match_report_rejects_empty_results() -> None:
    with pytest.raises(ValueError, match="results"):
        summarize_match_results((), config=MatchConfig())
