from __future__ import annotations

import json

from zero_lab.evaluation import (
    AlphaZeroCheckpoint,
    MatchConfig,
    PromotionConfig,
    RandomLegalMoveAgent,
    UniformSearchAgent,
    build_alpha_zero_promotion_report,
    compare_alpha_zero_checkpoints,
)
from zero_lab.games.toy import TicTacToeGame


def test_promotion_report_preserves_checkpoint_and_seed_metadata() -> None:
    match_config = MatchConfig(seed=71, games_per_side=1)
    promotion_config = PromotionConfig(match_config=match_config)
    comparison = compare_alpha_zero_checkpoints(
        champion=AlphaZeroCheckpoint(
            name="champion",
            uri="checkpoints/champion.pt",
            commit_hash="aaa1111",
        ),
        candidate=AlphaZeroCheckpoint(
            name="candidate",
            uri="checkpoints/candidate.pt",
            commit_hash="bbb2222",
        ),
        champion_agent=RandomLegalMoveAgent(),
        candidate_agent=UniformSearchAgent(simulations=4),
        games=(TicTacToeGame(),),
        config=match_config,
    )

    report = build_alpha_zero_promotion_report(comparison, config=promotion_config)
    payload = report.to_dict()
    champion = payload["champion"]
    candidate = payload["candidate"]
    seed_policy = payload["seed_policy"]
    config = payload["config"]

    assert payload["schema_version"] == 1
    assert isinstance(champion, dict)
    assert isinstance(candidate, dict)
    assert isinstance(seed_policy, dict)
    assert isinstance(config, dict)
    assert champion["commit_hash"] == "aaa1111"
    assert candidate["commit_hash"] == "bbb2222"
    assert seed_policy["base_seed"] == 71
    assert config["match"] == {
        "games_per_side": 1,
        "max_moves": 512,
        "seed": 71,
    }
    assert "candidate_elo_confidence_interval" in payload
    assert "results" in payload


def test_promotion_report_json_is_stable() -> None:
    match_config = MatchConfig(seed=3, games_per_side=1)
    comparison = compare_alpha_zero_checkpoints(
        champion=AlphaZeroCheckpoint(
            name="champion",
            uri="checkpoints/champion.pt",
            commit_hash="aaa1111",
        ),
        candidate=AlphaZeroCheckpoint(
            name="candidate",
            uri="checkpoints/candidate.pt",
            commit_hash="bbb2222",
        ),
        champion_agent=RandomLegalMoveAgent(),
        candidate_agent=UniformSearchAgent(simulations=4),
        games=(TicTacToeGame(),),
        config=match_config,
    )

    payload = json.loads(
        build_alpha_zero_promotion_report(
            comparison,
            config=PromotionConfig(match_config=match_config),
        ).to_json()
    )

    assert payload["candidate"]["name"] == "candidate"
    assert payload["promotion"]["decision"] in {"hold", "promote"}
