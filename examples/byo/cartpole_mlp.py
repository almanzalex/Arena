"""Self-contained CartPole MLP actor for BYO TorchScript export.

Mirrors the layer layout of CleanRL's pinned CartPole DQN QNetwork
(``cleanrl/dqn.py`` at ``fe8d8a03c41a7ef5b523e2e354bd01c363e786bb``) so labs can
rehearse the producer path without cloning CleanRL. Weights are deterministic
by default; pass a real checkpoint through the export script when you have one.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CartPoleMLP(nn.Module):
    def __init__(self, observation_dim: int = 4, action_n: int = 2) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, action_n),
        )
        # Fixed fills (not RNG) so export→verify→inspect digests are stable
        # across processes without a real checkpoint.
        with torch.no_grad():
            for idx, module in enumerate(self.network):
                if isinstance(module, nn.Linear):
                    module.weight.fill_(0.01 * (idx + 1))
                    if module.bias is not None:
                        module.bias.fill_(0.001 * (idx + 1))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)


def build_actor(
    observation_dim: int = 4,
    action_n: int = 2,
) -> CartPoleMLP:
    return CartPoleMLP(observation_dim=observation_dim, action_n=action_n)


# CartPole-v1 space descriptors used by the export script and tests.
CARTPOLE_OBSERVATION = {
    "type": "Box",
    "shape": [4],
    "dtype": "float32",
    "low": [-4.8, -3.4028235e38, -0.41887903, -3.4028235e38],
    "high": [4.8, 3.4028235e38, 0.41887903, 3.4028235e38],
}
CARTPOLE_ACTION = {
    "type": "Discrete",
    "n": 2,
    "dtype": "int64",
    "masks": "none",
}
REFERENCE_CASES = [
    {"observation": [0.0, 0.0, 0.0, 0.0], "mode": "deterministic"},
    {"observation": [0.05, 0.2, -0.03, -0.4], "mode": "deterministic"},
    {"observation": [-0.08, -0.6, 0.12, 0.8], "mode": "deterministic"},
    {"observation": [0.2, 1.5, -0.2, -1.0], "mode": "deterministic"},
]
