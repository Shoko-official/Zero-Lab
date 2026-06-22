from __future__ import annotations

import pytest
import torch

from zero_lab.training.alpha_zero import LinearAlphaZeroModel


def test_linear_alpha_zero_model_returns_policy_logits_and_values() -> None:
    model = LinearAlphaZeroModel(observation_size=9, action_size=9)
    observations = torch.zeros((2, 9), dtype=torch.float32)

    policy_logits, values = model(observations)

    assert policy_logits.shape == torch.Size((2, 9))
    assert values.shape == torch.Size((2,))


def test_linear_alpha_zero_model_validates_sizes() -> None:
    with pytest.raises(ValueError, match="observation_size"):
        LinearAlphaZeroModel(observation_size=0, action_size=9)
    with pytest.raises(ValueError, match="action_size"):
        LinearAlphaZeroModel(observation_size=9, action_size=0)
