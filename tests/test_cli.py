from __future__ import annotations

import json
from pathlib import Path

import pytest

from zero_lab.cli.main import main


def test_smoke_test_writes_runtime_payload_and_log(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["smoke-test", "--run-dir", str(tmp_path), "--seed", "123"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["project"] == "zero-lab"
    assert payload["seed"] == 123
    assert payload["status"] == "ok"
    assert payload["seeded_modules"] == ["python.random"]
    assert (tmp_path / "zero_lab.log").exists()


def test_show_config_prints_effective_config(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["show-config", "--run-dir", str(tmp_path), "--log-level", "debug"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["log_level"] == "DEBUG"
    assert payload["run_dir"] == str(tmp_path)


def test_list_games_prints_builtin_adapters(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["list-games"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload == [
        {"action_size": 9, "name": "tic_tac_toe"},
        {"action_size": 7, "name": "connect_four"},
        {"action_size": 4672, "name": "chess"},
    ]


def test_search_demo_runs_alpha_zero_search(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["search-demo", "--simulations", "16"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["game"] == "tic_tac_toe"
    assert payload["best_action"] == 2
    assert payload["simulations"] == 16
    assert sum(payload["visit_counts"].values()) == 16


def test_self_play_demo_writes_replay(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "self-play.jsonl"
    exit_code = main(
        [
            "self-play-demo",
            "--simulations",
            "4",
            "--seed",
            "3",
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["game"] == "tic_tac_toe"
    assert payload["length"] > 0
    assert payload["output"] == str(output)
    assert output.exists()


def test_replay_summary_prints_replay_counts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "self-play.jsonl"
    main(["self-play-demo", "--simulations", "4", "--output", str(output)])
    capsys.readouterr()

    exit_code = main(["replay-summary", str(output)])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["episodes"] == 1
    assert payload["games"] == {"tic_tac_toe": 1}
    assert payload["steps"] > 0


def test_replay_batch_summary_prints_training_batch_counts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "self-play.jsonl"
    main(["self-play-demo", "--simulations", "4", "--output", str(output)])
    capsys.readouterr()

    exit_code = main(
        [
            "replay-batch-summary",
            str(output),
            "--batch-size",
            "2",
            "--drop-remainder",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["batch_size"] == 2
    assert payload["drop_remainder"] is True
    assert payload["source_samples"] > 0
    assert payload["emitted_samples"] % 2 == 0
    assert payload["action_sizes"] == [9]
    assert payload["observation_sizes"] == [9]


def test_train_alpha_zero_trains_replay_and_writes_checkpoint(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    replay_path = tmp_path / "self-play.jsonl"
    run_dir = tmp_path / "run"
    main(
        [
            "self-play-demo",
            "--simulations",
            "4",
            "--seed",
            "3",
            "--output",
            str(replay_path),
        ]
    )
    capsys.readouterr()

    exit_code = main(
        [
            "train-alpha-zero",
            str(replay_path),
            "--run-dir",
            str(run_dir),
            "--seed",
            "3",
            "--batch-size",
            "1",
            "--steps",
            "1",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    checkpoint_path = run_dir / "checkpoints" / "tic_tac_toe-alpha-zero.pt"

    assert exit_code == 0
    assert payload["game"] == "tic_tac_toe"
    assert payload["batch_size"] == 1
    assert payload["steps"] == 1
    assert payload["samples"] == 1
    assert payload["checkpoint_path"] == str(checkpoint_path)
    assert checkpoint_path.exists()


def test_evaluate_prints_baseline_report(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "evaluate",
            "--games",
            "tic_tac_toe",
            "--seed",
            "7",
            "--simulations",
            "4",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["config"]["seed"] == 7
    assert payload["config"]["agents"][1]["name"] == "uniform_search"
    assert payload["config"]["agents"][1]["simulations"] == 4
    assert payload["games"] == ["tic_tac_toe"]
    assert set(payload["scores"]) == {"random_legal", "uniform_search"}
    assert len(payload["matches"]) == 2


def test_evaluate_writes_report_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "reports" / "evaluation.json"

    exit_code = main(
        [
            "evaluate",
            "--games",
            "tic_tac_toe",
            "--simulations",
            "4",
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8")) == json.loads(captured.out)


def test_chess_evaluate_prints_showcase_report(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "chess-evaluate",
            "--seed",
            "5",
            "--games-per-side",
            "1",
            "--max-plies",
            "4",
            "--simulations",
            "1",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["game"] == "chess"
    assert payload["config"]["seed"] == 5
    assert payload["config"]["agents"][1]["simulations"] == 1
    assert payload["seeds"] == [5, 6]
    assert set(payload["scores"]) == {"random_legal", "uniform_search"}
    assert len(payload["games"]) == 2
