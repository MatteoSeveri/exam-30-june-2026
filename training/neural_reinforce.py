"""PyTorch REINFORCE update for neural Briscola policies."""

from __future__ import annotations

from statistics import mean

import torch

from policy import NeuralSoftmaxPolicy

from .episode import EpisodeResult
from .reinforce import (
    ReinforceConfig,
    TrainStats,
    _all_steps,
    _baseline_for_step,
    _baseline_values,
    _score_margin,
)


def neural_reinforce_update(
    policy: NeuralSoftmaxPolicy,
    episodes: list[EpisodeResult],
    config: ReinforceConfig = ReinforceConfig(),
    optimizer: torch.optim.Optimizer | None = None,
) -> TrainStats:
    """Apply one PyTorch REINFORCE update from collected episode batches."""

    if not episodes:
        raise ValueError("Serve almeno un episodio per fare un update")

    steps = _all_steps(episodes)
    if not steps:
        raise ValueError("Serve almeno una decisione del learner per fare un update")

    if optimizer is None:
        optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = config.learning_rate

    baseline_values = _baseline_values(episodes, config.baseline)
    optimizer.zero_grad()

    losses: list[torch.Tensor] = []
    entropies: list[float] = []
    for episode in episodes:
        for decision_index, step in enumerate(episode.steps):
            advantage = step.reward_to_go - _baseline_for_step(
                decision_index,
                baseline=config.baseline,
                baseline_values=baseline_values,
            )
            cards, logits = policy.action_logits_tensor(step.osservazione)
            if step.azione not in cards:
                raise ValueError("Action is not legal")
            action_index = cards.index(step.azione)
            log_probabilities = torch.log_softmax(logits, dim=0)
            probabilities = torch.softmax(logits, dim=0)
            entropy = -(probabilities * log_probabilities).sum()
            entropies.append(float(entropy.detach().item()))
            # Average over episodes: the learning rate stays tied to games.
            losses.append(
                -log_probabilities[action_index] * (advantage / len(episodes))
                - config.entropy_coef * entropy / len(episodes)
            )

    loss = torch.stack(losses).sum()
    loss.backward()

    gradient_norm = _gradient_norm(policy)
    optimizer.step()

    return TrainStats(
        episodes=len(episodes),
        learner_decisions=len(steps),
        mean_return=float(mean(episode.episode_return for episode in episodes)),
        mean_score_margin=float(mean(_score_margin(episode) for episode in episodes)),
        gradient_norm=gradient_norm,
        baseline=config.baseline,
        baseline_values=baseline_values,
        mean_entropy=float(mean(entropies)),
    )


def _gradient_norm(policy: NeuralSoftmaxPolicy) -> float:
    gradients = [
        parameter.grad.detach().norm(2)
        for parameter in policy.parameters()
        if parameter.grad is not None
    ]
    if not gradients:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(gradients), ord=2).item())
