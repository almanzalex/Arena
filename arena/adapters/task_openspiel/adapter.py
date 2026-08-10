"""Qualified OpenSpiel semantic-family adapters.

The catalog stays evidence-gated, but it is no longer tied to one state machine:
sequential deterministic, sequential chance/imperfect-information, and
simultaneous games share a contract-checked packager.
"""

from __future__ import annotations

import functools
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np

from arena.core.errors import missing_extra,  ArenaError, SchemaError

FROZEN_GAME = "tic_tac_toe"
FROZEN_GAMES: dict[str, dict[str, Any]] = {
    "tic_tac_toe": {
        "interaction": "aec",
        "dynamics": "SEQUENTIAL",
        "chance_mode": "DETERMINISTIC",
        "information": "PERFECT_INFORMATION",
        "observation_kind": "observation",
        "observation_size": 27,
        "actions": 9,
    },
    "connect_four": {
        "interaction": "aec",
        "dynamics": "SEQUENTIAL",
        "chance_mode": "DETERMINISTIC",
        "information": "PERFECT_INFORMATION",
        "observation_kind": "observation",
        "observation_size": 126,
        "actions": 7,
    },
    "breakthrough": {
        "interaction": "aec",
        "dynamics": "SEQUENTIAL",
        "chance_mode": "DETERMINISTIC",
        "information": "PERFECT_INFORMATION",
        "observation_kind": "observation",
        "observation_size": 192,
        "actions": 768,
    },
    # Representative chance + imperfect-information family.
    "kuhn_poker": {
        "interaction": "aec",
        "dynamics": "SEQUENTIAL",
        "chance_mode": "EXPLICIT_STOCHASTIC",
        "information": "IMPERFECT_INFORMATION",
        "observation_kind": "information_state",
        "observation_size": 11,
        "actions": 2,
    },
    # Representative deterministic simultaneous-action family.
    "matrix_rps": {
        "interaction": "parallel",
        "dynamics": "SIMULTANEOUS",
        "chance_mode": "DETERMINISTIC",
        "information": "ONE_SHOT",
        "observation_kind": "observation",
        "observation_size": 1,
        "actions": 3,
    },
}


def normalize_game_id(value: str) -> str:
    return value.removeprefix("openspiel://")


def frozen_game_spec(value: str) -> dict[str, Any]:
    game = normalize_game_id(str(value))
    if game not in FROZEN_GAMES:
        raise SchemaError(
            f"OpenSpiel qualified catalog is {sorted(FROZEN_GAMES)!r}; game {game!r} "
            "is unqualified. Add a semantic-family fixture, reference trace, and "
            "qualification evidence before expanding the catalog."
        )
    return dict(FROZEN_GAMES[game])


def interaction_for_game(value: str) -> str:
    return str(frozen_game_spec(value)["interaction"])


def _require_pyspiel() -> Any:
    try:
        import pyspiel
    except ImportError as e:
        raise missing_extra(
            "openspiel",
            feature="OpenSpiel task adapter",
            capability="openspiel",
        ) from e
    return pyspiel


def _enum_has(value: Any, expected: str) -> bool:
    return expected in str(value).upper()


class _OpenSpielBase:
    """Shared contract checks and observation/reward/chance mechanics."""

    metadata: dict[str, Any]

    def __init__(self, game: str) -> None:
        self.game_id = normalize_game_id(game)
        self.contract = frozen_game_spec(self.game_id)
        pyspiel = _require_pyspiel()
        self.game = pyspiel.load_game(self.game_id)
        game_type = self.game.get_type()
        if self.game.num_players() != 2:
            raise SchemaError(
                f"OpenSpiel {self.game_id!r} requires two players for this qualified adapter"
            )
        expected_enums = {
            "dynamics": (game_type.dynamics, self.contract["dynamics"]),
            "chance_mode": (game_type.chance_mode, self.contract["chance_mode"]),
            "information": (game_type.information, self.contract["information"]),
        }
        drift = {
            key: {"expected": expected, "actual": str(actual)}
            for key, (actual, expected) in expected_enums.items()
            if not _enum_has(actual, str(expected))
        }
        observation_size = self._declared_observation_size()
        actual_shape = {
            "observation_size": observation_size,
            "actions": int(self.game.num_distinct_actions()),
        }
        expected_shape = {
            "observation_size": int(self.contract["observation_size"]),
            "actions": int(self.contract["actions"]),
        }
        if drift or actual_shape != expected_shape:
            raise SchemaError(
                f"OpenSpiel contract drift for {self.game_id!r}: "
                f"enum_drift={drift}, expected={expected_shape}, actual={actual_shape}"
            )
        self.metadata = {
            "name": f"arena_openspiel_{self.game_id}_v0",
            "render_modes": [],
        }
        self.possible_agents = ["player_0", "player_1"]
        self.agents: list[str] = []
        self.rewards: dict[str, float] = {}
        self.terminations: dict[str, bool] = {}
        self.truncations: dict[str, bool] = {}
        self.infos: dict[str, dict[str, Any]] = {}
        self._state: Any = None
        self._returns = np.zeros(2, dtype=np.float64)
        self._rng = np.random.default_rng(0)

    def _declared_observation_size(self) -> int:
        if self.contract["observation_kind"] == "information_state":
            return int(self.game.information_state_tensor_size())
        return int(self.game.observation_tensor_size())

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> Any:
        del agent
        try:
            from gymnasium import spaces
        except ImportError as e:  # pragma: no cover
            raise ArenaError("Gymnasium is required; install 'arena[openspiel]'") from e
        return spaces.Dict(
            {
                "observation": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self._declared_observation_size(),),
                    dtype=np.float32,
                ),
                "action_mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(int(self.game.num_distinct_actions()),),
                    dtype=np.int8,
                ),
            }
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> Any:
        del agent
        try:
            from gymnasium.spaces import Discrete
        except ImportError as e:  # pragma: no cover
            raise ArenaError("Gymnasium is required; install 'arena[openspiel]'") from e
        return Discrete(int(self.game.num_distinct_actions()))

    def _player(self, agent: str) -> int:
        if agent not in self.possible_agents:
            raise SchemaError(f"unknown OpenSpiel agent {agent!r}")
        return self.possible_agents.index(agent)

    def _tensor(self, player: int) -> np.ndarray:
        if self.contract["observation_kind"] == "information_state":
            value = self._state.information_state_tensor(player)
        else:
            value = self._state.observation_tensor(player)
        return np.asarray(value, dtype=np.float32)

    def observe(self, agent: str) -> dict[str, Any]:
        player = self._player(agent)
        tensor = self._tensor(player)
        mask = np.zeros(int(self.game.num_distinct_actions()), dtype=np.int8)
        if not self._state.is_terminal():
            try:
                legal = self._state.legal_actions(player)
            except Exception:  # noqa: BLE001
                legal = []
            if legal:
                mask[np.asarray(legal, dtype=np.int64)] = 1
        return {"observation": tensor, "action_mask": mask}

    def _advance_chance(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        while not self._state.is_terminal() and self._state.is_chance_node():
            outcomes = list(self._state.chance_outcomes())
            if not outcomes:
                raise SchemaError(
                    f"OpenSpiel {self.game_id!r} reported a chance node without outcomes"
                )
            actions = np.asarray([int(action) for action, _prob in outcomes], dtype=np.int64)
            probabilities = np.asarray(
                [float(probability) for _action, probability in outcomes], dtype=np.float64
            )
            if not np.isclose(probabilities.sum(), 1.0):
                raise SchemaError(
                    f"OpenSpiel {self.game_id!r} chance probabilities do not sum to one"
                )
            index = int(self._rng.choice(len(actions), p=probabilities))
            action = int(actions[index])
            probability = float(probabilities[index])
            events.append({"action": action, "probability": probability})
            self._state.apply_action(action)
        return events

    def _set_infos(self, chance_events: list[dict[str, Any]]) -> None:
        self.infos = {
            agent: {
                "openspiel": {
                    "game": self.game_id,
                    "observation_kind": self.contract["observation_kind"],
                    "chance_events": list(chance_events),
                }
            }
            for agent in self.possible_agents
        }

    def _set_rewards(self) -> None:
        now = np.asarray(self._state.returns(), dtype=np.float64)
        delta = now - self._returns
        self._returns = now
        self.rewards = {
            agent: float(delta[index])
            for index, agent in enumerate(self.possible_agents)
        }

    def render(self) -> str:
        return str(self._state)

    def close(self) -> None:
        self.agents = []


class OpenSpielSequentialAEC(_OpenSpielBase):
    """AEC facade for deterministic or explicit-chance sequential games."""

    def __init__(self, game: str = FROZEN_GAME) -> None:
        super().__init__(game)
        if self.contract["interaction"] != "aec":
            raise SchemaError(
                f"OpenSpiel game {self.game_id!r} requires "
                f"interaction={self.contract['interaction']}"
            )
        self.agent_selection: str | None = None

    def _agent_for_player(self, player: int) -> str:
        if player not in (0, 1):
            raise SchemaError(
                f"OpenSpiel {self.game_id} returned unsupported player id {player}; "
                "the chance adapter must consume chance nodes before policy selection"
            )
        return f"player_{player}"

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        del options
        self._rng = np.random.default_rng(0 if seed is None else int(seed))
        self._state = self.game.new_initial_state()
        self._returns = np.zeros(2, dtype=np.float64)
        chance_events = self._advance_chance()
        self.agents = list(self.possible_agents)
        self.rewards = {agent: 0.0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self._set_infos(chance_events)
        if self._state.is_terminal():
            self.agents = []
            self.agent_selection = None
        else:
            self.agent_selection = self._agent_for_player(self._state.current_player())

    def step(self, action: int) -> None:
        if not self.agents or self.agent_selection is None:
            raise SchemaError("cannot step a completed OpenSpiel episode")
        player = self._player(self.agent_selection)
        legal = list(self._state.legal_actions(player))
        if int(action) not in legal:
            raise SchemaError(
                f"illegal OpenSpiel action {action} for {self.agent_selection}; legal={legal}"
            )
        self._state.apply_action(int(action))
        chance_events = self._advance_chance()
        self._set_rewards()
        self._set_infos(chance_events)
        terminal = bool(self._state.is_terminal())
        self.terminations = {agent: terminal for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        if terminal:
            self.agents = []
            self.agent_selection = None
        else:
            self.agent_selection = self._agent_for_player(self._state.current_player())


class OpenSpielSimultaneousParallel(_OpenSpielBase):
    """Parallel facade for qualified simultaneous-action games."""

    def __init__(self, game: str = "matrix_rps") -> None:
        super().__init__(game)
        if self.contract["interaction"] != "parallel":
            raise SchemaError(
                f"OpenSpiel game {self.game_id!r} requires "
                f"interaction={self.contract['interaction']}"
            )

    def _require_simultaneous_or_terminal(self) -> None:
        if not self._state.is_terminal() and not self._state.is_simultaneous_node():
            raise SchemaError(
                f"OpenSpiel {self.game_id!r} left the qualified simultaneous state machine"
            )

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del options
        self._rng = np.random.default_rng(0 if seed is None else int(seed))
        self._state = self.game.new_initial_state()
        self._returns = np.zeros(2, dtype=np.float64)
        chance_events = self._advance_chance()
        self._require_simultaneous_or_terminal()
        self.agents = [] if self._state.is_terminal() else list(self.possible_agents)
        self.rewards = {agent: 0.0 for agent in self.possible_agents}
        terminal = bool(self._state.is_terminal())
        self.terminations = {agent: terminal for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self._set_infos(chance_events)
        return (
            {agent: self.observe(agent) for agent in self.agents},
            {agent: dict(self.infos[agent]) for agent in self.agents},
        )

    def step(
        self, actions: dict[str, int]
    ) -> tuple[
        dict[str, Any],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, Any],
    ]:
        if not self.agents:
            raise SchemaError("cannot step a completed OpenSpiel episode")
        if set(actions) != set(self.agents):
            raise SchemaError(
                f"simultaneous OpenSpiel actions must cover agents exactly; "
                f"expected={sorted(self.agents)}, got={sorted(actions)}"
            )
        ordered: list[int] = []
        for agent in self.possible_agents:
            player = self._player(agent)
            action = int(actions[agent])
            legal = list(self._state.legal_actions(player))
            if action not in legal:
                raise SchemaError(
                    f"illegal OpenSpiel action {action} for {agent}; legal={legal}"
                )
            ordered.append(action)
        self._state.apply_actions(ordered)
        chance_events = self._advance_chance()
        self._require_simultaneous_or_terminal()
        self._set_rewards()
        self._set_infos(chance_events)
        terminal = bool(self._state.is_terminal())
        self.terminations = {agent: terminal for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.agents = [] if terminal else list(self.possible_agents)
        return (
            {agent: self.observe(agent) for agent in self.agents},
            dict(self.rewards),
            dict(self.terminations),
            dict(self.truncations),
            {agent: dict(self.infos[agent]) for agent in self.agents},
        )


class OpenSpielPackager:
    kind = "openspiel"

    @staticmethod
    def interaction_for_game(game: str) -> str:
        return interaction_for_game(game)

    def make_env(self, spec: dict[str, Any], *, trust_task_code: bool = False) -> Any:
        del trust_task_code
        game = normalize_game_id(str(spec.get("game") or spec.get("env") or FROZEN_GAME))
        contract = frozen_game_spec(game)
        declared = str(spec.get("interaction", contract["interaction"]))
        if declared != contract["interaction"]:
            raise SchemaError(
                f"OpenSpiel {game!r} requires interaction={contract['interaction']}, "
                f"got {declared!r}"
            )
        if declared == "parallel":
            return OpenSpielSimultaneousParallel(game=game)
        return OpenSpielSequentialAEC(game=game)

    def describe_task(self, spec: dict[str, Any]) -> dict[str, Any]:
        from arena.adapters.task_pettingzoo.adapter import describe_env_contract

        game = normalize_game_id(str(spec.get("game") or spec.get("env") or FROZEN_GAME))
        contract = frozen_game_spec(game)
        interaction = str(spec.get("interaction", contract["interaction"]))
        if interaction != contract["interaction"]:
            raise SchemaError(
                f"OpenSpiel {game!r} requires interaction={contract['interaction']}, "
                f"got {interaction!r}"
            )
        try:
            package_version = version("open_spiel")
        except PackageNotFoundError:
            package_version = "uninstalled"
        result = describe_env_contract(
            {**spec, "interaction": interaction},
            self.make_env({**spec, "interaction": interaction}),
            adapter_name="openspiel",
            version=f"open_spiel-{package_version}:{game}",
        )
        result["game_semantics"] = {
            key: contract[key]
            for key in (
                "dynamics",
                "chance_mode",
                "information",
                "observation_kind",
            )
        }
        if contract["chance_mode"] != "DETERMINISTIC":
            result["chance_rng"] = "numpy_generator"
        return result


# Compatibility name retained for 0.3 importers.
OpenSpielTicTacToeAEC = OpenSpielSequentialAEC
