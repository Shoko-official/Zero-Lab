"""Replay-backed AlphaZero training loops."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch

from zero_lab.games import GameRules
from zero_lab.training.alpha_zero.steps import (
    AlphaZeroTrainStepConfig,
    AlphaZeroTrainStepResult,
    run_alpha_zero_training_step,
)
from zero_lab.training.alpha_zero.torch_batches import iter_torch_alpha_zero_training_batches

ALPHA_ZERO_CHECKPOINT_VERSION = 1


@dataclass(frozen=True, slots=True)
class AlphaZeroTrainerConfig:
    batch_size: int = 32
    max_steps: int = 1
    drop_remainder: bool = False
    device: torch.device | str | None = None
    dtype: torch.dtype = torch.float32
    step: AlphaZeroTrainStepConfig = AlphaZeroTrainStepConfig()
    checkpoint_path: Path | None = None
    resume_from: Path | None = None

    def __post_init__(self) -> None:
        _require_positive_integer(self.batch_size, "batch_size")
        _require_positive_integer(self.max_steps, "max_steps")
        if self.checkpoint_path is not None:
            object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))
        if self.resume_from is not None:
            object.__setattr__(self, "resume_from", Path(self.resume_from))


@dataclass(frozen=True, slots=True)
class AlphaZeroCheckpointMetadata:
    steps: int
    samples: int
    last_total_loss: float
    last_policy_loss: float
    last_value_loss: float

    def __post_init__(self) -> None:
        _require_non_negative_integer(self.steps, "steps")
        _require_non_negative_integer(self.samples, "samples")
        _require_finite(self.last_total_loss, "last_total_loss")
        _require_finite(self.last_policy_loss, "last_policy_loss")
        _require_finite(self.last_value_loss, "last_value_loss")

    def to_dict(self) -> dict[str, float | int]:
        return {
            "last_policy_loss": self.last_policy_loss,
            "last_total_loss": self.last_total_loss,
            "last_value_loss": self.last_value_loss,
            "samples": self.samples,
            "steps": self.steps,
        }


@dataclass(frozen=True, slots=True)
class AlphaZeroTrainingLoopResult:
    steps: int
    samples: int
    last_total_loss: float
    last_policy_loss: float
    last_value_loss: float
    checkpoint_path: Path | None

    def __post_init__(self) -> None:
        _require_non_negative_integer(self.steps, "steps")
        _require_non_negative_integer(self.samples, "samples")
        _require_finite(self.last_total_loss, "last_total_loss")
        _require_finite(self.last_policy_loss, "last_policy_loss")
        _require_finite(self.last_value_loss, "last_value_loss")
        if self.checkpoint_path is not None:
            object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))

    def to_dict(self) -> dict[str, float | int | str | None]:
        return {
            "checkpoint_path": None if self.checkpoint_path is None else str(self.checkpoint_path),
            "last_policy_loss": self.last_policy_loss,
            "last_total_loss": self.last_total_loss,
            "last_value_loss": self.last_value_loss,
            "samples": self.samples,
            "steps": self.steps,
        }


def train_alpha_zero_from_replay(
    paths: Path | Sequence[Path],
    *,
    games: Mapping[str, GameRules],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: AlphaZeroTrainerConfig | None = None,
) -> AlphaZeroTrainingLoopResult:
    resolved_config = AlphaZeroTrainerConfig() if config is None else config
    steps = 0
    samples = 0
    last_step: AlphaZeroTrainStepResult | None = None

    if resolved_config.resume_from is not None:
        metadata = load_alpha_zero_checkpoint(
            resolved_config.resume_from,
            model=model,
            optimizer=optimizer,
        )
        steps = metadata.steps
        samples = metadata.samples

    initial_steps = steps
    skipped_steps = 0
    for batch in iter_torch_alpha_zero_training_batches(
        paths,
        games=games,
        batch_size=resolved_config.batch_size,
        drop_remainder=resolved_config.drop_remainder,
        device=resolved_config.device,
        dtype=resolved_config.dtype,
    ):
        if skipped_steps < initial_steps:
            skipped_steps += 1
            continue

        last_step = run_alpha_zero_training_step(
            model,
            optimizer,
            batch,
            config=resolved_config.step,
        )
        steps += 1
        samples += batch.shape.batch_size
        if steps - initial_steps >= resolved_config.max_steps:
            break

    if last_step is None:
        raise ValueError("no training batches were produced")

    result = AlphaZeroTrainingLoopResult(
        steps=steps,
        samples=samples,
        last_total_loss=last_step.total_loss,
        last_policy_loss=last_step.policy_loss,
        last_value_loss=last_step.value_loss,
        checkpoint_path=resolved_config.checkpoint_path,
    )
    if resolved_config.checkpoint_path is not None:
        save_alpha_zero_checkpoint(
            resolved_config.checkpoint_path,
            model=model,
            optimizer=optimizer,
            result=result,
        )
    return result


def save_alpha_zero_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    result: AlphaZeroTrainingLoopResult,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "last_policy_loss": result.last_policy_loss,
            "last_total_loss": result.last_total_loss,
            "last_value_loss": result.last_value_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "samples": result.samples,
            "schema_version": ALPHA_ZERO_CHECKPOINT_VERSION,
            "steps": result.steps,
        },
        checkpoint_path,
    )


def load_alpha_zero_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> AlphaZeroCheckpointMetadata:
    payload = _read_mapping(cast(object, torch.load(Path(path), map_location="cpu")))
    schema_version = _read_int(payload, "schema_version")
    if schema_version != ALPHA_ZERO_CHECKPOINT_VERSION:
        raise ValueError("unsupported AlphaZero checkpoint schema version")

    model_state = _read_mapping(payload.get("model_state_dict"))
    optimizer_state = _read_mapping(payload.get("optimizer_state_dict"))
    model.load_state_dict(model_state)
    optimizer.load_state_dict(cast(dict[str, object], optimizer_state))

    return AlphaZeroCheckpointMetadata(
        steps=_read_int(payload, "steps"),
        samples=_read_int(payload, "samples"),
        last_total_loss=_read_float(payload, "last_total_loss"),
        last_policy_loss=_read_float(payload, "last_policy_loss"),
        last_value_loss=_read_float(payload, "last_value_loss"),
    )


def _read_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    return cast(Mapping[str, object], value)


def _read_int(values: Mapping[str, object], key: str) -> int:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _read_float(values: Mapping[str, object], key: str) -> float:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number")
    normalized = float(value)
    _require_finite(normalized, key)
    return normalized


def _require_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_non_negative_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
