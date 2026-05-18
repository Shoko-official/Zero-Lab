from __future__ import annotations

from pathlib import Path

import pytest

from zero_lab.replay import (
    AlphaZeroSample,
    EpisodeRecord,
    EpisodeStep,
    append_episode,
    iter_alpha_zero_samples,
    read_episodes,
    summarize_replay,
)


def test_episode_step_rejects_policy_that_does_not_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        EpisodeStep(
            action=0,
            current_player=1,
            policy=(0.2, 0.2),
            state="state",
            value_target=0.0,
        )


def test_episode_record_round_trips_through_dict() -> None:
    step = EpisodeStep(
        action=0,
        current_player=1,
        policy=(1.0,),
        state="state",
        value_target=1.0,
    )
    record = EpisodeRecord(game="toy", outcome=1, steps=(step,), terminal_state="terminal")

    restored = EpisodeRecord.from_dict(record.to_dict())

    assert restored == record


def test_jsonl_storage_appends_and_reads_episodes(tmp_path: Path) -> None:
    path = tmp_path / "episodes.jsonl"
    record = EpisodeRecord(
        game="toy",
        outcome=0,
        steps=(
            EpisodeStep(
                action=0,
                current_player=1,
                policy=(1.0,),
                state="state",
                value_target=0.0,
            ),
        ),
        terminal_state="terminal",
    )

    append_episode(path, record)
    append_episode(path, record)

    assert list(read_episodes(path)) == [record, record]


def test_replay_summary_counts_games_outcomes_and_steps(tmp_path: Path) -> None:
    path = tmp_path / "episodes.jsonl"
    record = EpisodeRecord(
        game="toy",
        outcome=1,
        steps=(
            EpisodeStep(
                action=0,
                current_player=1,
                policy=(1.0,),
                state="state",
                value_target=1.0,
            ),
        ),
        terminal_state="terminal",
    )
    append_episode(path, record)

    summary = summarize_replay(path)

    assert summary.to_dict() == {
        "episodes": 1,
        "games": {"toy": 1},
        "outcomes": {"1": 1},
        "steps": 1,
    }


def test_iter_alpha_zero_samples_streams_replay_steps(tmp_path: Path) -> None:
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    first_record = EpisodeRecord(
        game="toy",
        outcome=1,
        steps=(
            EpisodeStep(
                action=0,
                current_player=1,
                policy=(0.25, 0.75),
                state="state-a",
                value_target=1.0,
            ),
        ),
        terminal_state="terminal-a",
    )
    second_record = EpisodeRecord(
        game="toy",
        outcome=-1,
        steps=(
            EpisodeStep(
                action=1,
                current_player=-1,
                policy=(0.6, 0.4),
                state="state-b",
                value_target=1.0,
            ),
        ),
        terminal_state="terminal-b",
    )
    append_episode(first_path, first_record)
    append_episode(second_path, second_record)

    assert list(iter_alpha_zero_samples((first_path, second_path))) == [
        AlphaZeroSample(
            action=0,
            current_player=1,
            game="toy",
            policy=(0.25, 0.75),
            state="state-a",
            value_target=1.0,
        ),
        AlphaZeroSample(
            action=1,
            current_player=-1,
            game="toy",
            policy=(0.6, 0.4),
            state="state-b",
            value_target=1.0,
        ),
    ]
