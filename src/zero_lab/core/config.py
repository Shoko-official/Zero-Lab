"""Runtime configuration primitives."""

from __future__ import annotations

import json
import logging
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

_VALID_LOG_LEVELS = frozenset(logging.getLevelNamesMapping())


@dataclass(frozen=True)
class RuntimeConfig:
    project_name: str = "zero-lab"
    run_dir: Path = Path("runs/default")
    seed: int = 0
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if not self.project_name:
            raise ValueError("project_name must not be empty")
        if isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")

        object.__setattr__(self, "run_dir", Path(self.run_dir))

        normalized_log_level = self.log_level.upper()
        if normalized_log_level not in _VALID_LOG_LEVELS:
            raise ValueError(f"unsupported log level: {self.log_level}")
        object.__setattr__(self, "log_level", normalized_log_level)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> RuntimeConfig:
        runtime_values = values.get("runtime", values)
        if not isinstance(runtime_values, Mapping):
            raise ValueError("runtime config must be a mapping")

        return cls(
            project_name=_read_string(runtime_values, "project_name", cls.project_name),
            run_dir=Path(_read_string(runtime_values, "run_dir", str(cls.run_dir))),
            seed=_read_int(runtime_values, "seed", cls.seed),
            log_level=_read_string(runtime_values, "log_level", cls.log_level),
        )

    def with_overrides(
        self,
        *,
        run_dir: Path | None = None,
        seed: int | None = None,
        log_level: str | None = None,
    ) -> RuntimeConfig:
        return replace(
            self,
            run_dir=self.run_dir if run_dir is None else run_dir,
            seed=self.seed if seed is None else seed,
            log_level=self.log_level if log_level is None else log_level,
        )

    def to_dict(self) -> dict[str, str | int]:
        return {
            "log_level": self.log_level,
            "project_name": self.project_name,
            "run_dir": str(self.run_dir),
            "seed": self.seed,
        }


def load_runtime_config(path: Path | str | None = None) -> RuntimeConfig:
    if path is None:
        return RuntimeConfig()

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    suffix = config_path.suffix.lower()
    if suffix == ".json":
        with config_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    elif suffix == ".toml":
        with config_path.open("rb") as handle:
            loaded = tomllib.load(handle)
    else:
        raise ValueError(f"unsupported config file extension: {config_path.suffix}")

    if not isinstance(loaded, Mapping):
        raise ValueError("runtime config file must contain a mapping")

    return RuntimeConfig.from_mapping(loaded)


def _read_string(values: Mapping[str, Any], key: str, default: str) -> str:
    raw_value = values.get(key, default)
    if not isinstance(raw_value, str):
        raise ValueError(f"{key} must be a string")
    return raw_value


def _read_int(values: Mapping[str, Any], key: str, default: int) -> int:
    raw_value = values.get(key, default)
    if isinstance(raw_value, bool) or not isinstance(raw_value, int):
        raise ValueError(f"{key} must be an integer")
    return raw_value
