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
