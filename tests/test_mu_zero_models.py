from __future__ import annotations

import pytest

from zero_lab.models.mu_zero import (
    MuZeroInitialOutput,
    MuZeroRecurrentOutput,
    RecurrentBatch,
    validate_initial_observations,
)


def test_mu_zero_initial_observations_validate_masks() -> None:
    observations = validate_initial_observations(
        observations=((1, 0, -1),),
        legal_action_masks=((True, False),),
        action_size=2,
    )

    assert observations == ((1.0, 0.0, -1.0),)


def test_mu_zero_initial_output_validates_latent_and_policy_shapes() -> None:
    output = MuZeroInitialOutput.from_sequences(
        latent_states=((0.1, 0.2, 0.3),),
        policy_logits=((1.0, -1.0),),
        values=(0.25,),
        batch_size=1,
        action_size=2,
        latent_size=3,
    )

    assert output.latent_states == ((0.1, 0.2, 0.3),)
    assert output.policy_value.values == (0.25,)


def test_recurrent_batch_validates_actions() -> None:
    batch = RecurrentBatch.from_sequences(
        latent_states=((0.1, 0.2), (0.3, 0.4)),
        actions=(0, 1),
        legal_action_masks=((True, False), (False, True)),
        action_size=2,
    )

    assert batch.actions == (0, 1)
    assert batch.batch_size == 2
    assert batch.latent_size == 2


def test_recurrent_batch_rejects_out_of_range_action() -> None:
    with pytest.raises(ValueError, match="inside the action space"):
        RecurrentBatch.from_sequences(
            latent_states=((0.1, 0.2),),
            actions=(2,),
            legal_action_masks=((True, False),),
            action_size=2,
        )


def test_mu_zero_recurrent_output_validates_rewards() -> None:
    output = MuZeroRecurrentOutput.from_sequences(
        latent_states=((0.1, 0.2),),
        policy_logits=((1.0, -1.0),),
        values=(0.25,),
        rewards=(-1,),
        batch_size=1,
        action_size=2,
        latent_size=2,
    )

    assert output.rewards == (-1.0,)


def test_mu_zero_recurrent_output_rejects_reward_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="rewards batch dimension"):
        MuZeroRecurrentOutput.from_sequences(
            latent_states=((0.1, 0.2),),
            policy_logits=((1.0, -1.0),),
            values=(0.25,),
            rewards=(-1, 1),
            batch_size=1,
            action_size=2,
            latent_size=2,
        )
