"""Minimal edge-case PettingZoo Parallel envs + helpers for adversarial tests.

Intentionally unusual-but-in-scope tasks (single-agent Parallel, immediate
termination, zero-length, large discrete spaces, mid-episode crash) used to stress
the trajectory / failure-accounting / compatibility promises beyond pilot RPS.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium
import numpy as np
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv

from arena.adapters.policy_custom_torch import build_module, export_policy


class _BaseParallel(ParallelEnv):
    metadata = {"name": "arena_adversarial_v0", "render_modes": []}
    _agents: tuple[str, ...] = ("player_0",)
    _obs_n = 4
    _act_n = 3

    def __init__(self, max_cycles: int = 4, render_mode: str | None = None) -> None:
        self.max_cycles = int(max_cycles)
        self.render_mode = render_mode
        self.possible_agents = list(self._agents)
        self.agents: list[str] = []
        self._cycle = 0

    def observation_space(self, agent: str) -> gymnasium.Space:
        return Discrete(self._obs_n)

    def action_space(self, agent: str) -> gymnasium.Space:
        return Discrete(self._act_n)

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.agents = list(self.possible_agents)
        self._cycle = 0
        obs = {a: 0 for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions: dict[str, int]):
        self._cycle += 1
        rewards = {a: 0.0 for a in self.agents}
        terminations = {a: False for a in self.agents}
        truncations = {a: self._cycle >= self.max_cycles for a in self.agents}
        obs = {a: (int(actions[a]) % self._obs_n) for a in self.agents}
        infos = {a: {} for a in self.agents}
        if all(truncations.values()):
            self.agents = []
        return obs, rewards, terminations, truncations, infos

    def render(self) -> None:
        return None

    def close(self) -> None:
        self.agents = []


class SingleAgentParallel(_BaseParallel):
    """A one-agent Parallel task (single-agent RL exercised through the MARL runner)."""

    _agents = ("solo",)


class ImmediateTermParallel(_BaseParallel):
    """Terminates (not truncates) all agents on the very first step."""

    _agents = ("player_0", "player_1")

    def step(self, actions: dict[str, int]):
        rewards = {a: 1.0 for a in self.agents}
        terminations = {a: True for a in self.agents}
        truncations = {a: False for a in self.agents}
        obs = {a: 1 for a in self.agents}
        infos = {a: {} for a in self.agents}
        self.agents = []
        return obs, rewards, terminations, truncations, infos


class ZeroLengthParallel(_BaseParallel):
    """describe_task sees agents, but every *episode* ends with zero steps.

    ``make_env`` builds a fresh instance per episode, so a class-level counter is used:
    the first reset (compatibility introspection) exposes agents; later resets (the
    actual episodes) expose none, yielding zero-length episodes that must still be
    recorded and accounted for.
    """

    _agents = ("player_0",)
    _reset_calls = 0

    @classmethod
    def reset_counter(cls) -> None:
        cls._reset_calls = 0

    def reset(self, seed: int | None = None, options: dict | None = None):
        type(self)._reset_calls += 1
        if type(self)._reset_calls == 1:
            self.agents = list(self.possible_agents)
        else:
            self.agents = []
        self._cycle = 0
        obs = {a: 0 for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, infos


class LargeDiscreteParallel(_BaseParallel):
    """Large discrete action + observation spaces."""

    _agents = ("player_0", "player_1")
    _obs_n = 512
    _act_n = 257


class BoxObsParallel(ParallelEnv):
    """Single-agent Parallel task with a Box observation (float32, shape [4]).

    Observations form a seeded stream so (a) different seeds diverge and (b) the same
    seed reproduces the same stream, letting us detect recurrent-state resets at
    episode boundaries through the match runner itself. Matches the F3 recurrent
    policy's declared observation contract exactly.
    """

    metadata = {"name": "arena_boxobs_v0", "render_modes": []}

    def __init__(self, max_cycles: int = 5, render_mode: str | None = None) -> None:
        self.max_cycles = int(max_cycles)
        self.render_mode = render_mode
        self.possible_agents = ["agent"]
        self.agents: list[str] = []
        self._cycle = 0
        self._rng: np.random.Generator | None = None

    def observation_space(self, agent: str) -> gymnasium.Space:
        return Box(low=-10.0, high=10.0, shape=(4,), dtype=np.float32)

    def action_space(self, agent: str) -> gymnasium.Space:
        return Discrete(3)

    def reset(self, seed: int | None = None, options: dict | None = None):
        self._rng = np.random.default_rng(0 if seed is None else seed)
        self.agents = list(self.possible_agents)
        self._cycle = 0
        obs = {a: self._rng.normal(size=4).astype(np.float32) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions: dict[str, int]):
        self._cycle += 1
        rewards = {a: 0.0 for a in self.agents}
        terminations = {a: False for a in self.agents}
        truncations = {a: self._cycle >= self.max_cycles for a in self.agents}
        obs = {a: self._rng.normal(size=4).astype(np.float32) for a in self.agents}
        infos = {a: {} for a in self.agents}
        if all(truncations.values()):
            self.agents = []
        return obs, rewards, terminations, truncations, infos

    def render(self) -> None:
        return None

    def close(self) -> None:
        self.agents = []


class MidEpisodeCrashParallel(_BaseParallel):
    """Raises inside step() partway through an episode (unexpected env failure)."""

    _agents = ("player_0", "player_1")

    def step(self, actions: dict[str, int]):
        if self._cycle >= 1:
            raise ValueError("simulated mid-episode environment failure")
        return super().step(actions)


def make_discrete_policy(
    out_dir: Path,
    *,
    role: str,
    obs_n: int = 4,
    action_n: int = 3,
    masks: str = "none",
    seed: int = 0,
) -> Path:
    """Export a minimal Discrete-obs/Discrete-action policy bundle."""
    import torch

    arch = {
        "type": "mlp_categorical",
        "observation_dim": obs_n,
        "hidden_dims": [16],
        "action_n": action_n,
    }
    torch.manual_seed(seed)
    return export_policy(
        out_dir=out_dir,
        name=f"adv-{role}-o{obs_n}-a{action_n}-{masks}",
        roles=[role],
        observation={"type": "Discrete", "n": obs_n, "dtype": "int64"},
        action={"type": "Discrete", "n": action_n, "dtype": "int64", "masks": masks},
        architecture=arch,
        state_dict=build_module(arch).state_dict(),
        preprocessing={"id": "normalize_v0", "mean": 0.0, "std": 1.0},
        modes=["deterministic", "stochastic"],
    )
