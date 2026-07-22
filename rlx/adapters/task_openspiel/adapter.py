"""Narrow OpenSpiel adapter: exactly ``tic_tac_toe`` in RLX 0.3."""

from __future__ import annotations

import functools
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np

from rlx.core.errors import RlxError, SchemaError

FROZEN_GAME = "tic_tac_toe"


def _require_pyspiel() -> Any:
    try:
        import pyspiel
    except ImportError as e:
        raise RlxError(
            "OpenSpiel adapter is optional. Install with: pip install 'rlx[openspiel]'"
        ) from e
    return pyspiel


class OpenSpielTicTacToeAEC:
    """PettingZoo-shaped AEC facade over OpenSpiel's authoritative game state."""

    metadata = {"name": "rlx_openspiel_tic_tac_toe_v0", "render_modes": []}

    def __init__(self, game: str = FROZEN_GAME) -> None:
        if game != FROZEN_GAME:
            raise SchemaError(
                f"OpenSpiel 0.3 freezes exactly {FROZEN_GAME!r}; game {game!r} is unknown. "
                "Add a task-packager case, reference trace, and qualification evidence "
                "before expanding the claim."
            )
        pyspiel = _require_pyspiel()
        self.game_id = game
        self.game = pyspiel.load_game(game)
        self.possible_agents = ["player_0", "player_1"]
        self.agents: list[str] = []
        self.agent_selection: str | None = None
        self.rewards: dict[str, float] = {}
        self.terminations: dict[str, bool] = {}
        self.truncations: dict[str, bool] = {}
        self.infos: dict[str, dict[str, Any]] = {}
        self._state: Any = None
        self._returns = np.zeros(2, dtype=np.float64)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> Any:
        del agent
        try:
            from gymnasium import spaces
        except ImportError as e:  # pragma: no cover
            raise RlxError("Gymnasium is required; install 'rlx[openspiel]'") from e
        return spaces.Dict(
            {
                "observation": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(int(self.game.observation_tensor_size()),),
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
            raise RlxError("Gymnasium is required; install 'rlx[openspiel]'") from e
        return Discrete(int(self.game.num_distinct_actions()))

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        # Tic-tac-toe has no chance nodes; seed is still accepted for contract symmetry.
        del seed, options
        self._state = self.game.new_initial_state()
        self.agents = list(self.possible_agents)
        self._returns = np.zeros(2, dtype=np.float64)
        self.rewards = {agent: 0.0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}
        self.agent_selection = self._agent_for_player(self._state.current_player())

    def _agent_for_player(self, player: int) -> str:
        if player not in (0, 1):
            raise SchemaError(
                f"OpenSpiel tic_tac_toe returned unsupported player id {player}; "
                "chance/simultaneous nodes are outside the frozen 0.3 claim"
            )
        return f"player_{player}"

    def observe(self, agent: str) -> dict[str, Any]:
        player = self.possible_agents.index(agent)
        tensor = np.asarray(self._state.observation_tensor(player), dtype=np.float32)
        mask = np.zeros(int(self.game.num_distinct_actions()), dtype=np.int8)
        if not self._state.is_terminal() and player == self._state.current_player():
            mask[self._state.legal_actions(player)] = 1
        return {"observation": tensor, "action_mask": mask}

    def step(self, action: int) -> None:
        if not self.agents or self.agent_selection is None:
            raise SchemaError("cannot step a completed OpenSpiel episode")
        player = self.possible_agents.index(self.agent_selection)
        legal = self._state.legal_actions(player)
        if int(action) not in legal:
            raise SchemaError(
                f"illegal OpenSpiel action {action} for {self.agent_selection}; legal={legal}"
            )
        self._state.apply_action(int(action))
        now = np.asarray(self._state.returns(), dtype=np.float64)
        delta = now - self._returns
        self._returns = now
        self.rewards = {
            agent: float(delta[i]) for i, agent in enumerate(self.possible_agents)
        }
        terminal = bool(self._state.is_terminal())
        self.terminations = {agent: terminal for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        if terminal:
            self.agents = []
            self.agent_selection = None
        else:
            self.agent_selection = self._agent_for_player(self._state.current_player())

    def render(self) -> str:
        return str(self._state)

    def close(self) -> None:
        self.agents = []
        self.agent_selection = None


class OpenSpielPackager:
    kind = "openspiel"

    def make_env(self, spec: dict[str, Any], *, trust_task_code: bool = False) -> Any:
        del trust_task_code
        game = str(spec.get("game") or spec.get("env") or FROZEN_GAME)
        if game.startswith("openspiel://"):
            game = game.removeprefix("openspiel://")
        return OpenSpielTicTacToeAEC(game=game)

    def describe_task(self, spec: dict[str, Any]) -> dict[str, Any]:
        from rlx.adapters.task_pettingzoo.adapter import describe_env_contract

        if str(spec.get("interaction", "aec")) != "aec":
            raise SchemaError("OpenSpiel tic_tac_toe requires interaction=aec")
        try:
            package_version = version("open_spiel")
        except PackageNotFoundError:
            # make_env emits the actionable optional-extra message.
            package_version = "uninstalled"
        return describe_env_contract(
            {**spec, "interaction": "aec"},
            self.make_env(spec),
            adapter_name="openspiel",
            version=f"open_spiel-{package_version}:{FROZEN_GAME}",
        )
