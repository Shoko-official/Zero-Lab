"""PyTorch module adapters for AlphaZero search evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import torch

from zero_lab.games import GameState
from zero_lab.search.alpha_zero import Evaluation
from zero_lab.search.alpha_zero.model_evaluator import softmax_policy, terminal_evaluation


@dataclass(frozen=True, slots=True)
class TorchAlphaZeroEvaluator:
    model: torch.nn.Module
    device: torch.device | str | None = None
    dtype: torch.dtype = torch.float32

    def evaluate(self, state: GameState) -> Evaluation:
        return self.evaluate_batch((state,))[0]

    def evaluate_batch(self, states: Sequence[GameState]) -> tuple[Evaluation, ...]:
        states_tuple = tuple(states)
        if not states_tuple:
            return ()

        evaluations: list[Evaluation | None] = [None] * len(states_tuple)
        pending_positions: list[int] = []
        pending_states: list[GameState] = []
        for position, state in enumerate(states_tuple):
            if state.is_terminal:
                evaluations[position] = terminal_evaluation(state)
            else:
                pending_positions.append(position)
                pending_states.append(state)

        if pending_states:
            self._evaluate_pending(pending_states, pending_positions, evaluations)

        return tuple(_require_evaluation(evaluation) for evaluation in evaluations)

    def _evaluate_pending(
        self,
        states: Sequence[GameState],
        positions: Sequence[int],
        evaluations: list[Evaluation | None],
    ) -> None:
        action_size = states[0].action_size
        observation_size = len(states[0].canonical_observation())
        for state in states:
            if state.action_size != action_size:
                raise ValueError("batched torch evaluation requires matching action_size")
            if len(state.canonical_observation()) != observation_size:
                raise ValueError("batched torch evaluation requires matching observation size")

        target_device = None if self.device is None else torch.device(self.device)
        observations = torch.as_tensor(
            tuple(state.canonical_observation() for state in states),
            dtype=self.dtype,
            device=target_device,
        )
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                output = cast(object, self.model(observations))
        finally:
            if was_training:
                self.model.train()

        policy_logits, values = _unpack_torch_output(output, len(states), action_size)
        policy_rows = policy_logits.detach().cpu().tolist()
        value_rows = values.detach().cpu().tolist()
        for batch_index, position in enumerate(positions):
            state = states[batch_index]
            evaluations[position] = Evaluation(
                policy=softmax_policy(policy_rows[batch_index], state.legal_actions()),
                value=float(value_rows[batch_index]),
            )


def _unpack_torch_output(
    output: object,
    batch_size: int,
    action_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(output, tuple) or len(output) != 2:
        raise TypeError("model output must be a tuple of policy logits and predicted values")

    policy_logits, predicted_values = output
    if not isinstance(policy_logits, torch.Tensor):
        raise TypeError("policy logits must be a tensor")
    if not isinstance(predicted_values, torch.Tensor):
        raise TypeError("predicted values must be a tensor")

    if tuple(policy_logits.shape) != (batch_size, action_size):
        raise ValueError("policy logits shape must match batch_size and action_size")

    values = predicted_values
    if values.ndim == 2 and tuple(values.shape) == (batch_size, 1):
        values = values.squeeze(dim=1)
    if tuple(values.shape) != (batch_size,):
        raise ValueError("predicted values shape must match batch_size")
    return policy_logits, values


def _require_evaluation(evaluation: Evaluation | None) -> Evaluation:
    if evaluation is None:
        raise RuntimeError("missing evaluation")
    return evaluation
