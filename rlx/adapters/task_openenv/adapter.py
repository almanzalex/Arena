"""OpenEnv client adapter exposing a PettingZoo-shaped Parallel task.

OpenEnv transports one joint multi-agent action per ``step``. The remote
environment remains owned and hosted by OpenEnv; RLX only maps its serialized
result onto the existing Parallel match contract.
"""

from __future__ import annotations

import asyncio
import functools
import json
from typing import Any
from urllib.request import urlopen

from rlx.core.errors import RlxError, SchemaError, TaskRuntimeError
from rlx.core.identity import canonical_json, digest_uri, sha256_bytes

PILOT_ENV = "openenv://rlx/competitive_rps_v0"
PILOT_AGENTS = ("player_0", "player_1")
PILOT_CONTRACT = {
    "agents": list(PILOT_AGENTS),
    "roles": {
        agent: {
            "agents": [agent],
            "observation": {"type": "Discrete", "n": 4, "dtype": "int64"},
            "action": {"type": "Discrete", "n": 3, "dtype": "int64"},
        }
        for agent in PILOT_AGENTS
    },
    "provides_masks": False,
    "dynamic_agents": False,
}


def _space_from_contract(data: dict[str, Any]) -> Any:
    try:
        import gymnasium.spaces as spaces
        import numpy as np
    except ImportError as e:  # pragma: no cover - covered by optional-extra gate
        raise RlxError(
            "Gymnasium is required by the OpenEnv task bridge. "
            "Install with: pip install 'rlx[openenv]'"
        ) from e
    kind = data.get("type")
    if kind == "Discrete":
        return spaces.Discrete(int(data["n"]))
    if kind == "Box":
        return spaces.Box(
            low=data.get("low", -np.inf),
            high=data.get("high", np.inf),
            shape=tuple(data["shape"]),
            dtype=np.dtype(data.get("dtype", "float32")),
        )
    raise SchemaError(
        f"OpenEnv bridge does not support RLX space {kind!r}; add a registered task "
        "packager case and qualification fixture before claiming it"
    )


def _payload(value: Any) -> dict[str, Any]:
    observation = getattr(value, "observation", value)
    if hasattr(observation, "model_dump"):
        observation = observation.model_dump()
    if not isinstance(observation, dict):
        raise TaskRuntimeError(
            "OpenEnv response observation must be a JSON object",
            kind="protocol_error",
            details={"type": type(observation).__name__},
        )
    return observation


def _transport_error(exc: Exception, *, operation: str) -> TaskRuntimeError:
    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)) or "timeout" in lowered:
        kind = "timeout"
    elif any(word in lowered for word in ("connection", "disconnect", "closed", "refused")):
        kind = "disconnect"
    elif any(word in lowered for word in ("server error", "execution_error", "container")):
        kind = "container_crash"
    else:
        kind = "protocol_error"
    return TaskRuntimeError(
        f"OpenEnv {operation} failed: {text}",
        kind=kind,
        details={"operation": operation, "upstream_type": type(exc).__name__},
    )


@functools.lru_cache(maxsize=32)
def _verify_schema_pin(base_url: str, expected: str, timeout_seconds: float) -> None:
    """Refuse an imported endpoint whose advertised protocol schema drifted."""
    try:
        with urlopen(f"{base_url.rstrip('/')}/schema", timeout=timeout_seconds) as response:  # noqa: S310
            schema = json.loads(response.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        raise _transport_error(e, operation="schema verification") from e
    actual = digest_uri(sha256_bytes(canonical_json(schema)))
    if actual != expected:
        raise TaskRuntimeError(
            "OpenEnv endpoint schema changed after task import",
            kind="protocol_error",
            details={"expected_schema_digest": expected, "actual_schema_digest": actual},
        )


class OpenEnvParallelEnv:
    """Synchronous RLX view of an OpenEnv ``GenericEnvClient`` session."""

    metadata = {"name": "rlx_openenv_bridge_v0", "render_modes": []}

    def __init__(self, spec: dict[str, Any]) -> None:
        if str(spec.get("interaction", "parallel")) != "parallel":
            raise SchemaError("OpenEnv 0.3 pilot supports interaction=parallel only")
        packaging = spec.get("packaging") if isinstance(spec.get("packaging"), dict) else {}
        contract = spec.get("contract") or PILOT_CONTRACT
        if not isinstance(contract, dict) or not contract.get("roles"):
            raise SchemaError("OpenEnv task requires a pinned contract.roles mapping")
        self.spec = spec
        self.contract = contract
        self.possible_agents = list(contract.get("agents") or contract["roles"].keys())
        self.agents: list[str] = []
        self._client_context: Any | None = None
        client_factory = packaging.get("_client_factory")
        try:
            if client_factory is not None:
                client = client_factory(spec)
                self._client_context = client
            else:
                try:
                    from openenv.core import GenericEnvClient
                except ImportError as e:
                    raise RlxError(
                        "OpenEnv adapter is optional. Install with: pip install 'rlx[openenv]'"
                    ) from e
                base_url = packaging.get("base_url") or spec.get("base_url")
                if not base_url:
                    raise SchemaError(
                        "OpenEnv task requires packaging.base_url pinned by `rlx task import`"
                    )
                if packaging.get("schema_digest"):
                    _verify_schema_pin(
                        str(base_url),
                        str(packaging["schema_digest"]),
                        float(packaging.get("connect_timeout_seconds", 10)),
                    )
                async_client = GenericEnvClient(
                    base_url=str(base_url),
                    connect_timeout_s=float(packaging.get("connect_timeout_seconds", 10)),
                    message_timeout_s=float(packaging.get("message_timeout_seconds", 60)),
                )
                self._client_context = async_client.sync()
            if hasattr(self._client_context, "__enter__"):
                self._client = self._client_context.__enter__()
            else:
                self._client = self._client_context
        except (RlxError, SchemaError):
            raise
        except Exception as e:  # noqa: BLE001
            raise _transport_error(e, operation="connect") from e

    def observation_space(self, agent: str) -> Any:
        return _space_from_contract(self.contract["roles"][agent]["observation"])

    def action_space(self, agent: str) -> Any:
        return _space_from_contract(self.contract["roles"][agent]["action"])

    def reset(self, seed: int | None = None, options: dict | None = None):
        del options
        try:
            result = self._client.reset(seed=seed)
            data = _payload(result)
        except TaskRuntimeError:
            raise
        except Exception as e:  # noqa: BLE001
            raise _transport_error(e, operation="reset") from e
        observations = data.get("observations")
        if not isinstance(observations, dict):
            raise TaskRuntimeError(
                "OpenEnv reset response missing observations mapping",
                kind="protocol_error",
            )
        self.agents = list(self.possible_agents)
        infos = data.get("infos") or {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions: dict[str, Any]):
        try:
            result = self._client.step({"actions": actions})
            data = _payload(result)
        except TaskRuntimeError:
            raise
        except Exception as e:  # noqa: BLE001
            raise _transport_error(e, operation="step") from e
        required = ("observations", "rewards", "terminations", "truncations")
        missing = [key for key in required if not isinstance(data.get(key), dict)]
        if missing:
            raise TaskRuntimeError(
                f"OpenEnv step response missing mappings: {', '.join(missing)}",
                kind="protocol_error",
                details={"missing": missing},
            )
        terminations = data["terminations"]
        truncations = data["truncations"]
        if self.agents and (
            all(bool(terminations.get(a, False)) for a in self.agents)
            or all(bool(truncations.get(a, False)) for a in self.agents)
        ):
            self.agents = []
        return (
            data["observations"],
            data["rewards"],
            terminations,
            truncations,
            data.get("infos") or {},
        )

    def render(self) -> None:
        return None

    def close(self) -> None:
        self.agents = []
        context = self._client_context
        self._client_context = None
        if context is None:
            return
        try:
            if hasattr(context, "__exit__"):
                context.__exit__(None, None, None)
            elif hasattr(context, "close"):
                context.close()
        except Exception:
            # A close-time disconnect must not overwrite already-recorded episode evidence.
            pass


class OpenEnvPackager:
    kind = "openenv"

    def make_env(self, spec: dict[str, Any], *, trust_task_code: bool = False) -> Any:
        del trust_task_code
        return OpenEnvParallelEnv(spec)

    def describe_task(self, spec: dict[str, Any]) -> dict[str, Any]:
        contract = spec.get("contract") or (PILOT_CONTRACT if spec.get("env") == PILOT_ENV else None)
        if not isinstance(contract, dict) or not contract.get("roles"):
            raise SchemaError(
                "OpenEnv task has no pinned RLX contract. Re-import with --contract; "
                "server JSON Schema alone does not define per-role Gym spaces."
            )
        roles = contract["roles"]
        agents = list(contract.get("agents") or roles.keys())
        if set(agents) != set(roles):
            raise SchemaError("OpenEnv contract agents must exactly match contract.roles")
        for agent, role in roles.items():
            if not isinstance(role, dict) or not isinstance(role.get("observation"), dict):
                raise SchemaError(f"OpenEnv contract role {agent!r} requires observation")
            if not isinstance(role.get("action"), dict):
                raise SchemaError(f"OpenEnv contract role {agent!r} requires action")
            _space_from_contract(role["observation"])
            _space_from_contract(role["action"])
        packaging = spec.get("packaging") if isinstance(spec.get("packaging"), dict) else {}
        schema_digest = packaging.get("schema_digest")
        if schema_digest is None:
            schema_digest = digest_uri(sha256_bytes(canonical_json(contract)))
        return {
            "adapter": "openenv",
            "env": spec.get("env") or PILOT_ENV,
            "version": str(packaging.get("source_revision") or spec.get("version") or "unpinned"),
            "agents": agents,
            "roles": roles,
            "provides_masks": bool(contract.get("provides_masks", False)),
            "interaction": str(spec.get("interaction", "parallel")),
            "dynamic_agents": bool(contract.get("dynamic_agents", False)),
            "transport": {
                "kind": "openenv",
                "base_url": packaging.get("base_url"),
                "schema_digest": schema_digest,
            },
            "config": dict(spec.get("config") or {}),
        }
