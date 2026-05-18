from __future__ import annotations

import pytest

from zero_lab.replay import EpisodeRecord, EpisodeStep, append_episode, read_episodes


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


def test_jsonl_storage_appends_and_reads_episodes(tmp_path) -> None:
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
