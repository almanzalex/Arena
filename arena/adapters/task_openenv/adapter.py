"""OpenEnv client adapter exposing a PettingZoo-shaped Parallel task.

OpenEnv transports one joint multi-agent action per ``step``. The remote
environment remains owned and hosted by OpenEnv; Arena only maps its serialized
result onto the existing Parallel match contract.
"""

from __future__ import annotations

import asyncio
import json
import math
import warnings
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

from arena.core.errors import missing_extra,  ArenaError, SchemaError, TaskRuntimeError
from arena.core.identity import canonical_json, digest_uri, sha256_bytes
from arena.core.spaces import decode_bound_value

PILOT_ENV = "openenv://arena/competitive_rps_v0"
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
        raise missing_extra(
            "openenv",
            feature="OpenEnv task bridge (Gymnasium)",
            capability="openenv",
        ) from e
    kind = data.get("type")
    if kind == "Discrete":
        return spaces.Discrete(int(data["n"]))
    if kind == "Box":
        return spaces.Box(
            low=decode_bound_value(data.get("low", -np.inf)),
            high=decode_bound_value(data.get("high", np.inf)),
            shape=tuple(data["shape"]),
            dtype=np.dtype(data.get("dtype", "float32")),
        )
    if kind == "MultiDiscrete":
        return spaces.MultiDiscrete(
            np.asarray(data["nvec"], dtype=np.int64),
            dtype=np.dtype(data.get("dtype", "int64")),
        )
    if kind == "Dict":
        fields = data.get("spaces")
        if not isinstance(fields, dict) or not fields:
            raise SchemaError("OpenEnv Dict space requires a non-empty spaces mapping")
        return spaces.Dict(
            {
                str(key): _space_from_contract(dict(value))
                for key, value in fields.items()
            }
        )
    raise SchemaError(
        f"OpenEnv bridge does not support Arena space {kind!r}; add a registered task "
        "packager case and qualification fixture before claiming it"
    )


def _wire_value(value: Any) -> Any:
    """Convert registered runtime values into JSON-safe OpenEnv payloads."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _wire_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_wire_value(item) for item in value]
    return value


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


def _verify_schema_pin(base_url: str, expected: str, timeout_seconds: float) -> None:
    """Refuse an imported endpoint whose advertised protocol schema drifted."""
    parsed = urlparse(base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise SchemaError(
            "OpenEnv packaging.base_url must be an http(s) URL without "
            "embedded credentials, query, or fragment"
        )
    try:
        # The URL scheme and credential boundary are validated immediately above.
        with urlopen(  # nosec B310
            f"{base_url.rstrip('/')}/schema", timeout=timeout_seconds
        ) as response:
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


def _require_exact_agents(
    value: Any,
    *,
    field: str,
    expected: list[str],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TaskRuntimeError(
            f"OpenEnv response {field} must be a mapping",
            kind="protocol_error",
            details={"field": field, "type": type(value).__name__},
        )
    if set(value) != set(expected):
        raise TaskRuntimeError(
            f"OpenEnv response {field} does not match active agents",
            kind="protocol_error",
            details={
                "field": field,
                "expected": sorted(expected),
                "actual": sorted(map(str, value)),
            },
        )
    return value


def _validate_observation(
    env: OpenEnvParallelEnv,
    *,
    agent: str,
    value: Any,
) -> None:
    space = env.observation_space(agent)
    try:
        valid = bool(space.contains(value))
    except Exception:
        valid = False
    if not valid:
        raise TaskRuntimeError(
            f"OpenEnv observation for {agent!r} violates the pinned space",
            kind="protocol_error",
            details={
                "field": f"observations.{agent}",
                "space": repr(space),
                "value_type": type(value).__name__,
            },
        )


def _validate_step_payload(
    env: OpenEnvParallelEnv,
    data: dict[str, Any],
    *,
    expected: list[str],
) -> tuple[
    dict[str, Any],
    dict[str, float],
    dict[str, bool],
    dict[str, bool],
    dict[str, Any],
]:
    observations = _require_exact_agents(
        data.get("observations"), field="observations", expected=expected
    )
    rewards_raw = _require_exact_agents(
        data.get("rewards"), field="rewards", expected=expected
    )
    terminations_raw = _require_exact_agents(
        data.get("terminations"), field="terminations", expected=expected
    )
    truncations_raw = _require_exact_agents(
        data.get("truncations"), field="truncations", expected=expected
    )
    infos = _require_exact_agents(
        data.get("infos") or {agent: {} for agent in expected},
        field="infos",
        expected=expected,
    )
    rewards: dict[str, float] = {}
    terminations: dict[str, bool] = {}
    truncations: dict[str, bool] = {}
    for agent in expected:
        _validate_observation(env, agent=agent, value=observations[agent])
        reward = rewards_raw[agent]
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            raise TaskRuntimeError(
                f"OpenEnv reward for {agent!r} must be a finite number",
                kind="protocol_error",
                details={"field": f"rewards.{agent}", "value": repr(reward)},
            )
        numeric_reward = float(reward)
        if not math.isfinite(numeric_reward):
            raise TaskRuntimeError(
                f"OpenEnv reward for {agent!r} must be finite",
                kind="protocol_error",
                details={"field": f"rewards.{agent}", "value": repr(reward)},
            )
        rewards[agent] = numeric_reward
        for label, raw, target in (
            ("terminations", terminations_raw, terminations),
            ("truncations", truncations_raw, truncations),
        ):
            if not isinstance(raw[agent], bool):
                raise TaskRuntimeError(
                    f"OpenEnv {label} for {agent!r} must be boolean",
                    kind="protocol_error",
                    details={"field": f"{label}.{agent}", "value": repr(raw[agent])},
                )
            target[agent] = raw[agent]
        if not isinstance(infos[agent], dict):
            raise TaskRuntimeError(
                f"OpenEnv infos for {agent!r} must be a mapping",
                kind="protocol_error",
                details={"field": f"infos.{agent}"},
            )
    return observations, rewards, terminations, truncations, infos


class OpenEnvParallelEnv:
    """Synchronous Arena view of an OpenEnv ``GenericEnvClient`` session."""

    metadata = {"name": "arena_openenv_bridge_v0", "render_modes": []}

    def __init__(self, spec: dict[str, Any]) -> None:
        if str(spec.get("interaction", "parallel")) != "parallel":
            raise SchemaError("OpenEnv generic bridge currently requires interaction=parallel")
        packaging = spec.get("packaging") if isinstance(spec.get("packaging"), dict) else {}
        contract = spec.get("contract") or PILOT_CONTRACT
        if not isinstance(contract, dict) or not contract.get("roles"):
            raise SchemaError("OpenEnv task requires a pinned contract.roles mapping")
        self.spec = spec
        self.contract = contract
        self.possible_agents = list(contract.get("agents") or contract["roles"].keys())
        self.agents: list[str] = []
        self.cleanup_diagnostics: list[dict[str, Any]] = []
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
                    raise missing_extra(
                        "openenv",
                        feature="OpenEnv task adapter",
                        capability="openenv",
                    ) from e
                base_url = packaging.get("base_url") or spec.get("base_url")
                if not base_url:
                    raise SchemaError(
                        "OpenEnv task requires packaging.base_url pinned by `arena task import`"
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
        except (ArenaError, SchemaError):
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
        observations = _require_exact_agents(
            data.get("observations"),
            field="observations",
            expected=self.possible_agents,
        )
        for agent in self.possible_agents:
            _validate_observation(self, agent=agent, value=observations[agent])
        self.agents = list(self.possible_agents)
        infos = _require_exact_agents(
            data.get("infos") or {agent: {} for agent in self.agents},
            field="infos",
            expected=self.agents,
        )
        if any(not isinstance(infos[agent], dict) for agent in self.agents):
            raise TaskRuntimeError(
                "OpenEnv reset infos values must be mappings",
                kind="protocol_error",
                details={"field": "infos"},
            )
        return observations, infos

    def step(self, actions: dict[str, Any]):
        expected = list(self.agents)
        _require_exact_agents(actions, field="actions", expected=expected)
        for agent in expected:
            space = self.action_space(agent)
            if not bool(space.contains(actions[agent])):
                raise TaskRuntimeError(
                    f"OpenEnv action for {agent!r} violates the pinned action space",
                    kind="protocol_error",
                    details={"field": f"actions.{agent}", "space": repr(space)},
                )
        try:
            result = self._client.step({"actions": _wire_value(actions)})
            data = _payload(result)
        except TaskRuntimeError:
            raise
        except Exception as e:  # noqa: BLE001
            raise _transport_error(e, operation="step") from e
        observations, rewards, terminations, truncations, infos = _validate_step_payload(
            self,
            data,
            expected=expected,
        )
        if self.agents and (
            all(bool(terminations.get(a, False)) for a in self.agents)
            or all(bool(truncations.get(a, False)) for a in self.agents)
        ):
            self.agents = []
        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
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
        except Exception as exc:
            # A close-time disconnect must not overwrite already-recorded episode evidence.
            diagnostic = {
                "schema": "arena.cleanup-diagnostic/v1",
                "code": "OPENENV_CLOSE_FAILED",
                "message": str(exc),
                "operation": "close",
            }
            self.cleanup_diagnostics.append(diagnostic)
            warnings.warn(
                f"OpenEnv close failed after episode evidence was retained: {exc}",
                ResourceWarning,
                stacklevel=2,
            )


class OpenEnvPackager:
    kind = "openenv"

    def make_env(self, spec: dict[str, Any], *, trust_task_code: bool = False) -> Any:
        del trust_task_code
        return OpenEnvParallelEnv(spec)

    def describe_task(self, spec: dict[str, Any]) -> dict[str, Any]:
        contract = spec.get("contract") or (PILOT_CONTRACT if spec.get("env") == PILOT_ENV else None)
        if not isinstance(contract, dict) or not contract.get("roles"):
            raise SchemaError(
                "OpenEnv task has no pinned Arena contract. Re-import with --contract; "
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
        contract_digest = digest_uri(sha256_bytes(canonical_json(contract)))
        protocol = dict(packaging.get("protocol") or {})
        if protocol.get("contract_digest") not in {None, contract_digest}:
            raise SchemaError(
                "OpenEnv packaging.protocol.contract_digest does not match task contract"
            )
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
                "protocol": {
                    "schema": str(
                        protocol.get("schema", "arena.openenv-capabilities/v1")
                    ),
                    "interaction": "parallel",
                    "features": list(
                        protocol.get("features")
                        or [
                            "seeded_reset",
                            "joint_action",
                            "typed_contract",
                            "failure_taxonomy",
                        ]
                    ),
                    "contract_digest": contract_digest,
                },
            },
            "config": dict(spec.get("config") or {}),
        }
