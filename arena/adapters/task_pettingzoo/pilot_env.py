"""Self-contained PettingZoo Parallel competitive pilot env (RPS, no pygame)."""

from __future__ import annotations

import functools
from typing import Any

import gymnasium
import numpy as np
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv

# Observation: 0=start/none, 1=rock, 2=paper, 3=scissors (opponent's last move + 1)
# Action: 0=rock, 1=paper, 2=scissors
AGENTS = ("player_0", "player_1")
AEC_PILOT_ENV = "arena/competitive_rps_aec_v0"
DYNAMIC_PILOT_ENV = "arena/dynamic_lineup_aec_v0"
DYNAMIC_REENTRY_ENV = "arena/dynamic_reentry_aec_v0"
VECTOR_PILOT_ENV = "arena/vector_coordination_v0"


def env(**kwargs: Any) -> ParallelEnv:
    """Factory matching PettingZoo naming conventions."""
    return CompetitiveRPSParallel(**kwargs)


parallel_env = env


def vector_parallel_env(**kwargs: Any):
    """Box-observation pilot used to qualify generic OpenEnv contracts."""
    return VectorCoordinationParallel(**kwargs)


def aec_env(**kwargs: Any):
    """AEC twin of the Parallel RPS pilot (same spaces/payoffs)."""
    return CompetitiveRPSAEC(**kwargs)


def dynamic_aec_env(**kwargs: Any):
    """Forced join/leave fixture used to qualify the dynamic lifecycle contract."""
    return DynamicLineupAEC(**kwargs)


def dynamic_reentry_aec_env(**kwargs: Any):
    """Forced rejoin + multi-event fixture for lifecycle-v2 qualification."""
    return DynamicReentryAEC(**kwargs)


class CompetitiveRPSParallel(ParallelEnv):
    """Two-player zero-sum Rock–Paper–Scissors (Parallel API)."""

    metadata = {"name": "arena_competitive_rps_v0", "render_modes": []}

    def __init__(self, max_cycles: int = 1, render_mode: str | None = None) -> None:
        self.max_cycles = int(max_cycles)
        self.render_mode = render_mode
        self.possible_agents = list(AGENTS)
        self.agents: list[str] = []
        self._cycle = 0
        self._last_actions: dict[str, int] = {}
        self._np_random: np.random.Generator | None = None

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> gymnasium.Space:
        return Discrete(4)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> gymnasium.Space:
        return Discrete(3)

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        self.agents = list(self.possible_agents)
        self._cycle = 0
        self._last_actions = {a: -1 for a in self.agents}
        obs = {a: 0 for a in self.agents}  # start token
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions: dict[str, int]):
        if not self.agents:
            return {}, {}, {}, {}, {}

        a0, a1 = self.agents[0], self.agents[1]
        act0 = int(actions[a0])
        act1 = int(actions[a1])
        self._last_actions = {a0: act0, a1: act1}

        # Standard RPS payoff for player_0
        if act0 == act1:
            r0, r1 = 0.0, 0.0
        elif (act0 - act1) % 3 == 1:
            r0, r1 = 1.0, -1.0
        else:
            r0, r1 = -1.0, 1.0

        self._cycle += 1
        terminations = {a: False for a in self.agents}
        truncations = {a: self._cycle >= self.max_cycles for a in self.agents}
        rewards = {a0: r0, a1: r1}
        # Observation is opponent's last action + 1 (0 reserved for start)
        observations = {
            a0: act1 + 1,
            a1: act0 + 1,
        }
        infos = {a: {} for a in self.agents}

        if all(truncations.values()) or all(terminations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self) -> None:
        return None

    def close(self) -> None:
        self.agents = []


class VectorCoordinationParallel(ParallelEnv):
    """Seeded one-step coordination with a non-RPS observation contract."""

    metadata = {"name": "arena_vector_coordination_v0", "render_modes": []}

    def __init__(self, render_mode: str | None = None) -> None:
        self.render_mode = render_mode
        self.possible_agents = list(AGENTS)
        self.agents: list[str] = []
        self._signal = 0

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        del agent
        return Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str):
        del agent
        return Discrete(2)

    def reset(self, seed: int | None = None, options: dict | None = None):
        del options
        rng = np.random.default_rng(0 if seed is None else int(seed))
        self._signal = int(rng.integers(0, 2))
        self.agents = list(self.possible_agents)
        observation = np.asarray([float(self._signal), 0.0], dtype=np.float32)
        return (
            {agent: observation.copy() for agent in self.agents},
            {agent: {"signal": self._signal} for agent in self.agents},
        )

    def step(self, actions: dict[str, int]):
        if set(actions) != set(self.agents):
            raise ValueError(
                f"vector coordination requires actions for {self.agents}, got {sorted(actions)}"
            )
        rewards = {
            agent: 1.0
            if int(actions[agent]) == self._signal
            and len({int(value) for value in actions.values()}) == 1
            else 0.0
            for agent in self.agents
        }
        observation = np.asarray(
            [float(self._signal), float(next(iter(actions.values())))],
            dtype=np.float32,
        )
        observations = {agent: observation.copy() for agent in self.agents}
        terminations = {agent: True for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {"signal": self._signal} for agent in self.agents}
        self.agents = []
        return observations, rewards, terminations, truncations, infos

    def close(self) -> None:
        self.agents = []


class CompetitiveRPSAEC:
    """Two-player zero-sum Rock–Paper–Scissors (AEC API).

    Agents alternate turns; rewards accumulate so each full round matches Parallel payoffs.
    """

    metadata = {"name": "arena_competitive_rps_aec_v0", "is_parallelizable": False, "render_modes": []}

    def __init__(self, max_cycles: int = 1, render_mode: str | None = None) -> None:
        self.max_cycles = int(max_cycles)
        self.render_mode = render_mode
        self.possible_agents = list(AGENTS)
        self.agents: list[str] = []
        self.agent_selection: str | None = None
        self._cycle = 0
        self._pending: dict[str, int] = {}
        self._last_actions: dict[str, int] = {}
        self.rewards: dict[str, float] = {}
        self._cumulative_rewards: dict[str, float] = {}
        self.terminations: dict[str, bool] = {}
        self.truncations: dict[str, bool] = {}
        self.infos: dict[str, dict] = {}
        self._np_random: np.random.Generator | None = None

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        return Discrete(4)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str):
        return Discrete(3)

    def observe(self, agent: str) -> int:
        if not self._last_actions:
            return 0
        opp = self.possible_agents[0] if agent == self.possible_agents[1] else self.possible_agents[1]
        last = self._last_actions.get(opp, -1)
        return 0 if last < 0 else last + 1

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        self.agents = list(self.possible_agents)
        self.agent_selection = self.agents[0]
        self._cycle = 0
        self._pending = {}
        self._last_actions = {a: -1 for a in self.agents}
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        return None

    def step(self, action: int):
        if not self.agents:
            return
        agent = self.agent_selection
        assert agent is not None
        self._pending[agent] = int(action)
        self.rewards = {a: 0.0 for a in self.possible_agents}

        if len(self._pending) == len(self.possible_agents):
            a0, a1 = self.possible_agents[0], self.possible_agents[1]
            act0, act1 = self._pending[a0], self._pending[a1]
            self._last_actions = {a0: act0, a1: act1}
            if act0 == act1:
                r0, r1 = 0.0, 0.0
            elif (act0 - act1) % 3 == 1:
                r0, r1 = 1.0, -1.0
            else:
                r0, r1 = -1.0, 1.0
            self.rewards = {a0: r0, a1: r1}
            for a, r in self.rewards.items():
                self._cumulative_rewards[a] = self._cumulative_rewards.get(a, 0.0) + r
            self._cycle += 1
            self._pending = {}
            trunc = self._cycle >= self.max_cycles
            self.truncations = {a: trunc for a in self.possible_agents}
            self.terminations = {a: False for a in self.possible_agents}
            if trunc:
                self.agents = []
                self.agent_selection = None
                return

        # Advance to next living agent.
        if self.agents:
            idx = self.agents.index(agent) if agent in self.agents else -1
            nxt = self.agents[(idx + 1) % len(self.agents)]
            self.agent_selection = nxt

    def close(self) -> None:
        self.agents = []


class DynamicLineupAEC:
    """Small deterministic AEC environment with one birth and two removal events.

    ``agent_0`` and ``agent_1`` start alive. After both have acted, ``agent_0``
    leaves and ``agent_2`` joins. The second lineup then acts once each before
    the episode ends. All possible agents share the RPS-shaped contract so the
    fixture focuses on lifecycle correctness rather than space conversion.
    """

    metadata = {
        "name": "arena_dynamic_lineup_aec_v0",
        "is_parallelizable": False,
        "render_modes": [],
    }

    def __init__(self, render_mode: str | None = None) -> None:
        self.render_mode = render_mode
        self.possible_agents = ["agent_0", "agent_1", "agent_2"]
        self.agents: list[str] = []
        self.agent_selection: str | None = None
        self.rewards: dict[str, float] = {}
        self.terminations: dict[str, bool] = {}
        self.truncations: dict[str, bool] = {}
        self.infos: dict[str, dict[str, Any]] = {}
        self._turn = 0
        self._last_action = -1

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        if agent not in self.possible_agents:
            raise ValueError(f"unknown agent {agent!r}")
        return Discrete(4)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str):
        if agent not in self.possible_agents:
            raise ValueError(f"unknown agent {agent!r}")
        return Discrete(3)

    def observe(self, agent: str) -> int:
        if agent not in self.possible_agents:
            raise ValueError(f"unknown agent {agent!r}")
        return 0 if self._last_action < 0 else self._last_action + 1

    def reset(self, seed: int | None = None, options: dict | None = None):
        del seed, options
        self.agents = ["agent_0", "agent_1"]
        self.agent_selection = "agent_0"
        self._turn = 0
        self._last_action = -1
        self.rewards = {agent: 0.0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}
        return None

    def step(self, action: int) -> None:
        if not self.agents or self.agent_selection is None:
            raise ValueError("cannot step a completed dynamic-lineup episode")
        acting = self.agent_selection
        value = int(action)
        if value not in (0, 1, 2):
            raise ValueError(f"invalid action {value}")
        self._last_action = value
        self.rewards = {agent: 0.0 for agent in self.possible_agents}
        self.rewards[acting] = 1.0 if value == self.possible_agents.index(acting) else 0.0
        self._turn += 1

        if self._turn == 1:
            self.agent_selection = "agent_1"
        elif self._turn == 2:
            # Forced lifecycle boundary: one removal and one birth.
            self.terminations["agent_0"] = True
            self.agents = ["agent_1", "agent_2"]
            self.agent_selection = "agent_1"
        elif self._turn == 3:
            self.agent_selection = "agent_2"
        else:
            for agent in list(self.agents):
                self.truncations[agent] = True
            self.agents = []
            self.agent_selection = None

    def close(self) -> None:
        self.agents = []
        self.agent_selection = None


class DynamicReentryAEC:
    """Deterministic lifecycle fixture with removal, rejoin, and joint birth.

    ``agent_0`` leaves after acting, then rejoins on the same boundary where
    ``agent_1`` leaves and ``agent_2`` is born. This makes agent-ID reuse and
    multiple lifecycle events observable without introducing task complexity.
    """

    metadata = {
        "name": "arena_dynamic_reentry_aec_v0",
        "is_parallelizable": False,
        "render_modes": [],
    }

    def __init__(self, render_mode: str | None = None) -> None:
        self.render_mode = render_mode
        self.possible_agents = ["agent_0", "agent_1", "agent_2"]
        self.agents: list[str] = []
        self.agent_selection: str | None = None
        self.rewards: dict[str, float] = {}
        self.terminations: dict[str, bool] = {}
        self.truncations: dict[str, bool] = {}
        self.infos: dict[str, dict[str, Any]] = {}
        self._turn = 0
        self._last_action = -1

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        if agent not in self.possible_agents:
            raise ValueError(f"unknown agent {agent!r}")
        return Discrete(4)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str):
        if agent not in self.possible_agents:
            raise ValueError(f"unknown agent {agent!r}")
        return Discrete(3)

    def observe(self, agent: str) -> int:
        if agent not in self.possible_agents:
            raise ValueError(f"unknown agent {agent!r}")
        return 0 if self._last_action < 0 else self._last_action + 1

    def reset(self, seed: int | None = None, options: dict | None = None):
        del seed, options
        self.agents = ["agent_0", "agent_1"]
        self.agent_selection = "agent_0"
        self._turn = 0
        self._last_action = -1
        self.rewards = {agent: 0.0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}
        return None

    def step(self, action: int) -> None:
        if not self.agents or self.agent_selection is None:
            raise ValueError("cannot step a completed dynamic-reentry episode")
        acting = self.agent_selection
        value = int(action)
        if value not in (0, 1, 2):
            raise ValueError(f"invalid action {value}")
        self._last_action = value
        self.rewards = {agent: 0.0 for agent in self.possible_agents}
        self.rewards[acting] = 1.0 if value == self.possible_agents.index(acting) else 0.0
        self._turn += 1

        if self._turn == 1:
            self.terminations["agent_0"] = True
            self.agents = ["agent_1"]
            self.agent_selection = "agent_1"
        elif self._turn == 2:
            self.terminations["agent_1"] = True
            self.terminations["agent_0"] = False
            self.agents = ["agent_0", "agent_2"]
            self.agent_selection = "agent_0"
        elif self._turn == 3:
            self.agent_selection = "agent_2"
        else:
            for agent in list(self.agents):
                self.truncations[agent] = True
            self.agents = []
            self.agent_selection = None

    def close(self) -> None:
        self.agents = []
        self.agent_selection = None
