"""Single-agent PettingZoo Parallel wrapper around Gymnasium CartPole-v1.

Used by the mini-train example so a trained Arena policy can be matched under the
same task packaging path as other Parallel envs (via ``entrypoint_bundle``).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete
from pettingzoo.utils.env import ParallelEnv

AGENT = "agent"
CARTPOLE_OBS = {
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


class CartPoleParallel(ParallelEnv):
    """One-agent Parallel API over ``CartPole-v1``."""

    metadata = {"name": "arena_cartpole_v0", "render_modes": []}

    def __init__(self) -> None:
        self.possible_agents = [AGENT]
        self.agents: list[str] = []
        self._env = gym.make("CartPole-v1")
        self._obs_space = Box(
            low=np.asarray(CARTPOLE_OBS["low"], dtype=np.float32),
            high=np.asarray(CARTPOLE_OBS["high"], dtype=np.float32),
            dtype=np.float32,
        )
        self._act_space = Discrete(int(CARTPOLE_ACTION["n"]))

    def observation_space(self, agent: str) -> Box:
        del agent
        return self._obs_space

    def action_space(self, agent: str) -> Discrete:
        del agent
        return self._act_space

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ):
        obs, info = self._env.reset(seed=seed, options=options)
        self.agents = list(self.possible_agents)
        return {AGENT: np.asarray(obs, dtype=np.float32)}, {AGENT: dict(info)}

    def step(self, actions: dict[str, int]):
        if not self.agents:
            return {}, {}, {}, {}, {}
        action = int(actions[AGENT])
        obs, reward, terminated, truncated, info = self._env.step(action)
        obs_out = {AGENT: np.asarray(obs, dtype=np.float32)}
        rewards = {AGENT: float(reward)}
        terminations = {AGENT: bool(terminated)}
        truncations = {AGENT: bool(truncated)}
        infos = {AGENT: dict(info)}
        if terminated or truncated:
            self.agents = []
        return obs_out, rewards, terminations, truncations, infos

    def close(self) -> None:
        self._env.close()


def parallel_env(**kwargs: Any) -> CartPoleParallel:
    """Factory name expected by ``entrypoint_bundle`` packaging."""
    del kwargs
    return CartPoleParallel()
