"""Frozen OpenEnv RPS pilot server used for T-01/T-03 / R-05 qualification.

Run as a **separate** process or container — the Arena client must connect via
``packaging.base_url`` / ``ARENA_OPENENV_BASE_URL``, not embed this server:

* Process: ``python -m arena.adapters.task_openenv.server --port 8000``
* Docker: ``docker compose -f docker/openenv/docker-compose.yml up --build -d``
* Recipe helper: ``arena.adapters.task_openenv.service_recipe``

The server uses OpenEnv's own FastAPI/WebSocket transport and wraps Arena's
frozen PettingZoo RPS (or vector) environment; it does not implement a competing
container layer.
"""

from __future__ import annotations

from typing import Any

from arena.core.errors import missing_extra


def create_app(env_kind: str = "rps") -> Any:
    try:
        from openenv.core.env_server.http_server import create_app as openenv_create_app
        from openenv.core.env_server.interfaces import Environment
        from openenv.core.env_server.types import Action, Observation, State
        from pydantic import Field
    except ImportError as e:  # pragma: no cover - optional dependency error path
        raise missing_extra("openenv", feature="OpenEnv pilot server", capability="openenv") from e

    if env_kind == "vector":
        from arena.adapters.task_pettingzoo.pilot_env import VectorCoordinationParallel

        class VectorAction(Action):
            actions: dict[str, int]

        class VectorObservation(Observation):
            observations: dict[str, list[float]]
            rewards: dict[str, float] = Field(default_factory=dict)
            terminations: dict[str, bool] = Field(default_factory=dict)
            truncations: dict[str, bool] = Field(default_factory=dict)
            infos: dict[str, dict[str, Any]] = Field(default_factory=dict)

        class VectorEnvironment(
            Environment[VectorAction, VectorObservation, State]
        ):
            SUPPORTS_CONCURRENT_SESSIONS = True

            def __init__(self) -> None:
                super().__init__()
                self._env = VectorCoordinationParallel()
                self._state = State(step_count=0)

            def reset(
                self,
                seed: int | None = None,
                episode_id: str | None = None,
                **kwargs: Any,
            ) -> VectorObservation:
                del kwargs
                observations, infos = self._env.reset(seed=seed)
                self._state = State(episode_id=episode_id, step_count=0)
                return VectorObservation(
                    observations=observations,
                    rewards={agent: 0.0 for agent in observations},
                    terminations={agent: False for agent in observations},
                    truncations={agent: False for agent in observations},
                    infos=infos,
                    done=False,
                    reward=0.0,
                )

            def step(
                self,
                action: VectorAction,
                timeout_s: float | None = None,
                **kwargs: Any,
            ) -> VectorObservation:
                del timeout_s, kwargs
                observations, rewards, terminations, truncations, infos = self._env.step(
                    action.actions
                )
                self._state.step_count += 1
                return VectorObservation(
                    observations=observations,
                    rewards=rewards,
                    terminations=terminations,
                    truncations=truncations,
                    infos=infos,
                    done=True,
                    reward=float(sum(rewards.values())),
                )

            @property
            def state(self) -> State:
                return self._state

            def close(self) -> None:
                self._env.close()

        return openenv_create_app(
            VectorEnvironment,
            VectorAction,
            VectorObservation,
            env_name="arena-vector-coordination-v0",
            max_concurrent_envs=16,
        )

    if env_kind != "rps":
        raise ValueError(f"unknown OpenEnv fixture {env_kind!r}; use rps|vector")

    from arena.adapters.task_pettingzoo.pilot_env import CompetitiveRPSParallel

    class RPSAction(Action):
        actions: dict[str, int]

    class RPSObservation(Observation):
        observations: dict[str, int]
        rewards: dict[str, float] = Field(default_factory=dict)
        terminations: dict[str, bool] = Field(default_factory=dict)
        truncations: dict[str, bool] = Field(default_factory=dict)
        infos: dict[str, dict[str, Any]] = Field(default_factory=dict)

    class RPSEnvironment(Environment[RPSAction, RPSObservation, State]):
        SUPPORTS_CONCURRENT_SESSIONS = True

        def __init__(self) -> None:
            super().__init__()
            self._env = CompetitiveRPSParallel(max_cycles=1)
            self._state = State(step_count=0)

        def reset(
            self,
            seed: int | None = None,
            episode_id: str | None = None,
            **kwargs: Any,
        ) -> RPSObservation:
            del kwargs
            observations, infos = self._env.reset(seed=seed)
            self._state = State(episode_id=episode_id, step_count=0)
            return RPSObservation(
                observations=observations,
                rewards={agent: 0.0 for agent in observations},
                terminations={agent: False for agent in observations},
                truncations={agent: False for agent in observations},
                infos=infos,
                done=False,
                reward=0.0,
            )

        def step(
            self,
            action: RPSAction,
            timeout_s: float | None = None,
            **kwargs: Any,
        ) -> RPSObservation:
            del timeout_s, kwargs
            observations, rewards, terminations, truncations, infos = self._env.step(
                action.actions
            )
            self._state.step_count += 1
            done = all(terminations.values()) or all(truncations.values())
            return RPSObservation(
                observations=observations,
                rewards=rewards,
                terminations=terminations,
                truncations=truncations,
                infos=infos,
                done=done,
                reward=float(sum(rewards.values())),
            )

        @property
        def state(self) -> State:
            return self._state

        def close(self) -> None:
            self._env.close()

    return openenv_create_app(
        RPSEnvironment,
        RPSAction,
        RPSObservation,
        env_name="arena-competitive-rps-v0",
        max_concurrent_envs=16,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--env", choices=["rps", "vector"], default="rps")
    args = parser.parse_args()
    try:
        import uvicorn
    except ImportError as e:  # pragma: no cover
        raise missing_extra("openenv", feature="OpenEnv pilot uvicorn", capability="openenv") from e
    uvicorn.run(create_app(args.env), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
