"""Exporter-side bridge for CleanRL's pinned CartPole DQN QNetwork."""

from __future__ import annotations

import torch.nn as nn


class CleanRLQNetwork(nn.Module):
    def __init__(self, observation_dim: int = 4, action_n: int = 2) -> None:
        super().__init__()
        # Exact module names and dimensions from CleanRL cleanrl/dqn.py at
        # fe8d8a03c41a7ef5b523e2e354bd01c363e786bb.
        self.network = nn.Sequential(
            nn.Linear(observation_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, action_n),
        )

    def forward(self, observation):
        return self.network(observation)


def build_actor(
    observation_dim: int = 4,
    action_n: int = 2,
) -> CleanRLQNetwork:
    return CleanRLQNetwork(observation_dim=observation_dim, action_n=action_n)
