from __future__ import annotations

import pytest
import torch

from zero_lab.training.alpha_zero import (
    TorchAlphaZeroTrainingBatch,
    as_torch_alpha_zero_training_batch,
)
from zero_lab.training.alpha_zero.batches import AlphaZeroTrainingBatch


def make_training_batch() -> AlphaZeroTrainingBatch:
    return AlphaZeroTrainingBatch.from_sequences(
        observations=((1, 0, -1), (0, 1, 0)),
        legal_action_masks=((True, False), (True, True)),
        target_policies=((1.0, 0.0), (0.25, 0.75)),
        target_values=(1, -1),
        selected_actions=(0, 1),
        current_players=(1, -1),
        action_size=2,
    )


def test_as_torch_alpha_zero_training_batch_converts_cpu_tensors() -> None:
    batch = as_torch_alpha_zero_training_batch(make_training_batch())

    assert batch.observations.shape == torch.Size((2, 3))
    assert batch.legal_action_masks.shape == torch.Size((2, 2))
    assert batch.target_policies.shape == torch.Size((2, 2))
    assert batch.target_values.shape == torch.Size((2,))
    assert batch.selected_actions.shape == torch.Size((2,))
    assert batch.current_players.shape == torch.Size((2,))
    assert batch.observations.dtype == torch.float32
    assert batch.target_policies.dtype == torch.float32
    assert batch.target_values.dtype == torch.float32
    assert batch.legal_action_masks.dtype == torch.bool
    assert batch.selected_actions.dtype == torch.long
    assert batch.current_players.dtype == torch.long
    assert batch.device == torch.device("cpu")
    assert batch.dtype == torch.float32


def test_as_torch_alpha_zero_training_batch_honors_dtype_and_device() -> None:
    batch = as_torch_alpha_zero_training_batch(
        make_training_batch(),
        device="cpu",
        dtype=torch.float64,
    )

    assert batch.observations.device == torch.device("cpu")
    assert batch.observations.dtype == torch.float64
    assert batch.target_policies.dtype == torch.float64
    assert batch.target_values.dtype == torch.float64
    assert batch.legal_action_masks.dtype == torch.bool


def test_torch_alpha_zero_training_batch_rejects_wrong_shape() -> None:
    source = as_torch_alpha_zero_training_batch(make_training_batch())

    with pytest.raises(ValueError, match="tensor shape"):
        TorchAlphaZeroTrainingBatch(
            observations=source.observations[:1],
            legal_action_masks=source.legal_action_masks,
            target_policies=source.target_policies,
            target_values=source.target_values,
            selected_actions=source.selected_actions,
            current_players=source.current_players,
            shape=source.shape,
        )


def test_torch_alpha_zero_training_batch_rejects_wrong_mask_dtype() -> None:
    source = as_torch_alpha_zero_training_batch(make_training_batch())

    with pytest.raises(TypeError, match="legal_action_masks"):
        TorchAlphaZeroTrainingBatch(
            observations=source.observations,
            legal_action_masks=source.legal_action_masks.to(torch.float32),
            target_policies=source.target_policies,
            target_values=source.target_values,
            selected_actions=source.selected_actions,
            current_players=source.current_players,
            shape=source.shape,
        )
