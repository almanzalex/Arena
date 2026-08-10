"""PettingZoo Parallel wrapper around Gymnasium CartPole-v1 (free classic control).

Used by ``run_real_env_match.py`` via ``entrypoint_bundle`` packaging so Arena core
does not claim a first-party CartPole task. Requires ``gymnasium`` (pulled in by
``arena[pettingzoo]``).
"""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np
from gymnasium.spaces import Box, Discrete
from pettingzoo.utils.env import ParallelEnv

AGENT = "agent"


def parallel_env(*, max_cycles: int = 200, **kwargs: Any) -> ParallelEnv:
    """Factory consumed by ``entrypoint_bundle`` (``packaging.factory``)."""
    del kwargs  # reserved for future CartPole kwargs
    return GymnasiumCartPoleParallel(max_cycles=max_cycles)


class GymnasiumCartPoleParallel(ParallelEnv):
    """Single-agent Parallel env that owns a real ``CartPole-v1`` instance."""

    metadata = {"name": "gymnasium_cartpole_parallel_v0", "render_modes": []}

    def __init__(self, max_cycles: int = 200) -> None:
        super().__init__()
        self.max_cycles = int(max_cycles)
        self.possible_agents = [AGENT]
        self.agents: list[str] = []
        self._env = gymnasium.make("CartPole-v1")
        self._cycle = 0
        self._obs_space = Box(
            low=np.asarray(self._env.observation_space.low, dtype=np.float32),
            high=np.asarray(self._env.observation_space.high, dtype=np.float32),
            dtype=np.float32,
        )
        self._act_space = Discrete(int(self._env.action_space.n))

    def observation_space(self, agent: str) -> Box:
        del agent
        return self._obs_space

    def action_space(self, agent: str) -> Discrete:
        del agent
        return self._act_space

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.agents = list(self.possible_agents)
        self._cycle = 0
        observation, info = self._env.reset(seed=seed, options=options)
        obs = {AGENT: np.asarray(observation, dtype=np.float32)}
        infos = {AGENT: dict(info)}
        return obs, infos

    def step(self, actions: dict[str, int]):
        if not self.agents:
            return {}, {}, {}, {}, {}
        action = int(actions[AGENT])
        observation, reward, terminated, truncated, info = self._env.step(action)
        self._cycle += 1
        if self._cycle >= self.max_cycles:
            truncated = True
        obs = {AGENT: np.asarray(observation, dtype=np.float32)}
        rewards = {AGENT: float(reward)}
        terminations = {AGENT: bool(terminated)}
        truncations = {AGENT: bool(truncated)}
        infos = {AGENT: dict(info)}
        if terminated or truncated:
            self.agents = []
        return obs, rewards, terminations, truncations, infos

    def render(self) -> None:
        return None

    def close(self) -> None:
        self.agents = []
        self._env.close()
