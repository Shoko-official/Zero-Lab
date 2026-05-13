from __future__ import annotations

import json
from pathlib import Path

import pytest

from zero_lab.core.config import RuntimeConfig, load_runtime_config


def test_default_runtime_config_is_valid() -> None:
    config = RuntimeConfig()

    assert config.project_name == "zero-lab"
    assert config.seed == 0
    assert config.log_level == "INFO"


def test_load_runtime_config_from_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "runtime.toml"
    config_path.write_text(
        "\n".join(
            [
                "[runtime]",
                'project_name = "zero-lab-test"',
                'run_dir = "runs/test"',
                "seed = 7",
                'log_level = "warning"',
            ]
        ),
        encoding="utf-8",
    )

    config = load_runtime_config(config_path)

    assert config.project_name == "zero-lab-test"
    assert config.run_dir.parts == ("runs", "test")
    assert config.seed == 7
    assert config.log_level == "WARNING"


def test_load_runtime_config_from_json(tmp_path: Path) -> None:
    config_path = tmp_path / "runtime.json"
    config_path.write_text(json.dumps({"runtime": {"seed": 11}}), encoding="utf-8")

    config = load_runtime_config(config_path)

    assert config.seed == 11


def test_runtime_config_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="seed"):
        RuntimeConfig(seed=-1)
