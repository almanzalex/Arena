"""Task import and semantic trace-equivalence operations (Arena 0.3)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse
from urllib.request import urlopen

import numpy as np

from arena.adapters.task_openenv.adapter import PILOT_CONTRACT, PILOT_ENV
from arena.adapters.task_pettingzoo.adapter import make_env
from arena.core.errors import ConformanceError, SchemaError
from arena.core.identity import canonical_json, digest_uri, sha256_bytes
from arena.core.manifests import (
    TASK_SCHEMA,
    dump_yaml,
    load_manifest,
    task_content_digest,
    validate_task_manifest,
    validate_trace_suite,
)


def load_task_spec(ref: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(ref, dict):
        return dict(ref)
    path = Path(ref)
    if path.exists():
        task = load_manifest(path)
        if task.get("schema") == TASK_SCHEMA:
            validate_task_manifest(task)
        return task
    text = str(ref)
    if text.startswith("pettingzoo://"):
        return {"adapter": "pettingzoo-parallel", "env": text.removeprefix("pettingzoo://")}
    if text.startswith("openspiel://"):
        from arena.adapters.task_openspiel import interaction_for_game

        return {
            "adapter": "openspiel",
            "env": text,
            "interaction": interaction_for_game(text),
        }
    raise SchemaError(f"cannot load task reference {text!r}; pass a task YAML or registered URI")


def import_openenv_task(
    source: str,
    *,
    name: str,
    out: Path | str,
    contract_path: Path | str | None = None,
    source_revision: str | None = None,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Pin an OpenEnv endpoint plus the Arena role-space contract it cannot infer."""
    parsed = urlparse(source)
    if parsed.scheme != "openenv":
        raise SchemaError("task import currently supports openenv:// sources")
    query = parse_qs(parsed.query)
    transport = str(query.get("transport", ["http"])[0])
    if transport not in {"http", "https"}:
        raise SchemaError("OpenEnv URI transport must be http|https")
    if not parsed.netloc:
        raise SchemaError(
            "OpenEnv URI must include a server, e.g. "
            "openenv://127.0.0.1:8000/arena/competitive_rps_v0"
        )
    if parsed.username is not None or parsed.password is not None:
        raise SchemaError(
            "OpenEnv URI must not embed credentials; configure transport "
            "authentication outside the task identity"
        )
    if parsed.fragment:
        raise SchemaError("OpenEnv URI must not contain a fragment")
    base_url = f"{transport}://{parsed.netloc}"
    env_path = unquote(parsed.path.lstrip("/"))
    env_uri = f"openenv://{env_path}" if env_path else source.split("?", 1)[0]
    if contract_path is not None:
        contract = load_manifest(contract_path)
    elif env_uri == PILOT_ENV:
        contract = PILOT_CONTRACT
    else:
        raise SchemaError(
            "OpenEnv JSON Schema does not define Arena per-role Gym spaces; "
            "pass --contract for non-pilot environments"
        )
    try:
        # ``transport`` is restricted to http/https and ``netloc`` is required above.
        with urlopen(  # nosec B310
            f"{base_url}/schema", timeout=timeout_seconds
        ) as response:
            upstream_schema = json.loads(response.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        raise SchemaError(f"cannot pin OpenEnv /schema at {base_url}: {e}") from e
    schema_digest = digest_uri(sha256_bytes(canonical_json(upstream_schema)))
    contract_digest = digest_uri(sha256_bytes(canonical_json(contract)))
    manifest = {
        "schema": TASK_SCHEMA,
        "name": name,
        "adapter": "openenv",
        "env": env_uri,
        "interaction": "parallel",
        "version": source_revision or "openenv-server",
        "packaging": {
            "kind": "openenv",
            "base_url": base_url,
            "source_revision": source_revision or "unreported",
            "schema_digest": schema_digest,
            "connect_timeout_seconds": timeout_seconds,
            "message_timeout_seconds": 60,
            "protocol": {
                "schema": "arena.openenv-capabilities/v1",
                "interaction": "parallel",
                "features": [
                    "seeded_reset",
                    "joint_action",
                    "typed_contract",
                    "failure_taxonomy",
                ],
                "contract_digest": contract_digest,
            },
        },
        "contract": contract,
    }
    validate_task_manifest(manifest)
    manifest["digest"] = task_content_digest(manifest)
    dump_yaml(manifest, out)
    return manifest


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def capture_task_trace(task: dict[str, Any], suite: dict[str, Any]) -> list[dict[str, Any]]:
    suite = validate_trace_suite(dict(suite))
    interaction = str(task.get("interaction", suite.get("interaction", "parallel")))
    traces: list[dict[str, Any]] = []
    for episode_index, episode in enumerate(suite["episodes"]):
        env = make_env(task)
        try:
            seed = int(episode.get("seed", 0))
            events: list[dict[str, Any]] = []
            if interaction in {"aec", "dynamic_aec"}:
                env.reset(seed=seed)
                events.append(
                    {
                        "kind": "reset",
                        "agent_selection": env.agent_selection,
                        "observations": {
                            agent: _jsonable(env.observe(agent)) for agent in env.agents
                        },
                    }
                )
                for action_spec in episode["actions"]:
                    if not env.agents:
                        break
                    if not isinstance(action_spec, dict) or "action" not in action_spec:
                        raise SchemaError("AEC trace actions require {agent, action}")
                    expected_agent = action_spec.get("agent", env.agent_selection)
                    if expected_agent != env.agent_selection:
                        raise ConformanceError(
                            f"trace expected {expected_agent}, task selected {env.agent_selection}"
                        )
                    acting = env.agent_selection
                    before = _jsonable(env.observe(acting))
                    env.step(action_spec["action"])
                    events.append(
                        {
                            "kind": "step",
                            "agent": acting,
                            "observation": before,
                            "action": _jsonable(action_spec["action"]),
                            "rewards": _jsonable(env.rewards),
                            "terminations": _jsonable(env.terminations),
                            "truncations": _jsonable(env.truncations),
                            "next_agent": env.agent_selection,
                        }
                    )
            else:
                observations, infos = env.reset(seed=seed)
                events.append(
                    {"kind": "reset", "observations": _jsonable(observations), "infos": _jsonable(infos)}
                )
                for actions in episode["actions"]:
                    if not env.agents:
                        break
                    observations, rewards, terms, truncs, infos = env.step(actions)
                    events.append(
                        {
                            "kind": "step",
                            "actions": _jsonable(actions),
                            "observations": _jsonable(observations),
                            "rewards": _jsonable(rewards),
                            "terminations": _jsonable(terms),
                            "truncations": _jsonable(truncs),
                            "infos": _jsonable(infos),
                        }
                    )
            traces.append({"episode_index": episode_index, "seed": seed, "events": events})
        finally:
            env.close()
    return traces


def _compare(left: Any, right: Any, *, path: str, tolerance: float, diffs: list[dict[str, Any]]) -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            diffs.append({"path": path, "left_keys": sorted(left), "right_keys": sorted(right)})
            return
        for key in sorted(left):
            _compare(left[key], right[key], path=f"{path}.{key}", tolerance=tolerance, diffs=diffs)
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            diffs.append({"path": path, "left_len": len(left), "right_len": len(right)})
            return
        for i, (a, b) in enumerate(zip(left, right, strict=True)):
            _compare(a, b, path=f"{path}[{i}]", tolerance=tolerance, diffs=diffs)
        return
    if isinstance(left, bool) or isinstance(right, bool):
        if type(left) is not type(right) or left != right:
            diffs.append({"path": path, "left": left, "right": right})
        return
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance):
            diffs.append({"path": path, "left": left, "right": right, "tolerance": tolerance})
        return
    if left != right:
        diffs.append({"path": path, "left": left, "right": right})


def verify_task_equivalence(
    left_task: dict[str, Any],
    right_task: dict[str, Any] | None,
    suite: dict[str, Any],
) -> dict[str, Any]:
    left = capture_task_trace(left_task, suite)
    left_trace_digest = digest_uri(sha256_bytes(canonical_json(left)))
    right = capture_task_trace(right_task, suite) if right_task is not None else suite.get("reference")
    reference_digest = suite.get("reference_digest")
    if right is None and reference_digest is None:
        raise SchemaError(
            "equivalence requires a second task, trace suite reference, or reference_digest"
        )
    tolerance = float((suite.get("tolerances") or {}).get("default", 0.0))
    diffs: list[dict[str, Any]] = []
    if right is not None:
        _compare(left, right, path="$", tolerance=tolerance, diffs=diffs)
    elif left_trace_digest != reference_digest:
        diffs.append(
            {
                "path": "$",
                "captured_trace_digest": left_trace_digest,
                "reference_digest": reference_digest,
            }
        )
    result = {
        "schema": "arena.equivalence-report/v1",
        "ok": not diffs,
        "left_task_digest": task_content_digest(left_task),
        "right_task_digest": task_content_digest(right_task) if right_task is not None else None,
        "trace_suite_digest": digest_uri(sha256_bytes(canonical_json(suite))),
        "tolerance": tolerance,
        "diffs": diffs,
        "episodes": len(left),
        "captured_trace_digest": left_trace_digest,
        "shared_task_intent_digest": digest_uri(
            sha256_bytes(
                canonical_json(
                    {
                        "schema": "arena.verified-task-intent/v1",
                        "interaction": suite.get("interaction", "parallel"),
                        "trace_suite_digest": digest_uri(
                            sha256_bytes(canonical_json(suite))
                        ),
                        "trace_result_digest": left_trace_digest,
                        "tolerance": tolerance,
                    }
                )
            )
        ),
    }
    if diffs:
        raise ConformanceError(json.dumps(result, sort_keys=True))
    return result
