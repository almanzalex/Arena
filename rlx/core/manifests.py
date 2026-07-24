"""Schema-versioned manifests: load, validate, canonicalize."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml
from yaml.events import AliasEvent

from rlx.core.errors import SchemaError
from rlx.core.identity import canonical_json, digest_uri, parse_digest, sha256_bytes
from rlx.core.io import DEFAULT_MAX_BYTES, atomic_write_bytes, read_text_bounded

POLICY_SCHEMA = "rlx.policy/v0alpha1"
MATCH_SCHEMA = "rlx.match/v0alpha1"
TRAJECTORY_SCHEMA = "rlx.trajectory/v0alpha1"
RUN_SCHEMA = "rlx.run/v0alpha1"
POPULATION_SCHEMA = "rlx.population/v0alpha1"
EVALUATION_SCHEMA = "rlx.evaluation/v0alpha1"
EVAL_RUN_SCHEMA = "rlx.eval-run/v0alpha1"
EVAL_REPORT_SCHEMA = "rlx.eval-report/v0alpha1"
EVALUATION_INTENT_SCHEMA = "rlx.evaluation-intent/v1"
EVALUATION_BINDING_SCHEMA = "rlx.evaluation-binding/v1"
EVAL_RUN_V1_SCHEMA = "rlx.eval-run/v1"
EVAL_REPORT_V1_SCHEMA = "rlx.eval-report/v1"
DATASET_SCHEMA = "rlx.dataset/v0alpha1"
EVAL_BUNDLE_SCHEMA = "rlx.eval-bundle/v0alpha1"
TASK_SCHEMA = "rlx.task/v0alpha1"
TRACE_SUITE_SCHEMA = "rlx.trace-suite/v1"
MAX_MANIFEST_DEPTH = 128


class _StrictSafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects aliases, merge keys, and duplicate keys."""

    def compose_node(self, parent: Any, index: Any) -> Any:
        if self.check_event(AliasEvent):
            event = self.peek_event()
            raise SchemaError(
                f"YAML aliases are not allowed in identity-bearing manifests: {event.anchor!r}"
            )
        return super().compose_node(parent, index)

    def construct_mapping(self, node: Any, deep: bool = False) -> dict[Any, Any]:
        if not isinstance(node, yaml.MappingNode):
            raise SchemaError("expected a YAML mapping node")
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            if (
                isinstance(key_node, yaml.ScalarNode)
                and key_node.tag == "tag:yaml.org,2002:merge"
            ):
                raise SchemaError("YAML merge keys are not allowed in manifests")
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in mapping
            except TypeError as exc:
                raise SchemaError("manifest mapping keys must be scalar/hashable") from exc
            if duplicate:
                raise SchemaError(f"duplicate manifest key: {key!r}")
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _StrictSafeLoader.construct_mapping,
)


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SchemaError(f"duplicate manifest key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise SchemaError(f"non-finite JSON number is not allowed: {value}")


def _validate_depth(value: Any, *, depth: int = 0) -> None:
    if depth > MAX_MANIFEST_DEPTH:
        raise SchemaError(f"manifest exceeds maximum nesting depth {MAX_MANIFEST_DEPTH}")
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise SchemaError(
                    f"manifest mapping keys must be strings, got {type(key).__name__}"
                )
            _validate_depth(item, depth=depth + 1)
    elif isinstance(value, list):
        for item in value:
            _validate_depth(item, depth=depth + 1)


def load_manifest(
    path: Path | str,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> dict[str, Any]:
    path = Path(path)
    text = read_text_bounded(path, max_bytes=max_bytes)
    try:
        if path.suffix.lower() in {".yaml", ".yml"}:
            # The custom loader subclasses SafeLoader and only adds rejection.
            data = yaml.load(text, Loader=_StrictSafeLoader)  # nosec B506
        else:
            data = json.loads(
                text,
                object_pairs_hook=_json_object,
                parse_constant=_reject_json_constant,
            )
    except SchemaError:
        raise
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise SchemaError(f"invalid manifest syntax in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SchemaError(f"manifest root must be a mapping: {path}")
    _validate_depth(data)
    canonical_json(data)
    return data


def dump_yaml(data: dict[str, Any], path: Path | str) -> None:
    path = Path(path)
    canonical_json(data)
    payload = yaml.safe_dump(data, sort_keys=False, allow_unicode=True).encode("utf-8")
    atomic_write_bytes(path, payload)


def dump_json(data: dict[str, Any], path: Path | str) -> None:
    atomic_write_bytes(path, canonical_json(data) + b"\n")


def validate_policy_manifest(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("schema") != POLICY_SCHEMA:
        raise SchemaError(f"expected schema {POLICY_SCHEMA}, got {data.get('schema')!r}")
    required = [
        "name",
        "roles",
        "runtime",
        "observation",
        "action",
        "state",
        "inference",
        "preprocessing",
        "payloads",
        "architecture",
    ]
    for key in required:
        if key not in data:
            raise SchemaError(f"policy manifest missing required field: {key}")
    roles = data["roles"]
    if "allowed" not in roles or not isinstance(roles["allowed"], list):
        raise SchemaError("roles.allowed must be a list")
    runtime = data["runtime"]
    if "adapter" not in runtime:
        raise SchemaError("runtime.adapter is required")
    action = data["action"]
    if "masks" not in action:
        raise SchemaError("action.masks is required (none|optional|required)")
    if action["masks"] not in {"none", "optional", "required"}:
        raise SchemaError("action.masks must be none|optional|required")
    state = data["state"]
    if "recurrent" not in state:
        raise SchemaError("state.recurrent is required")
    # Distinguish missing vs explicitly empty: reset_on=[] means "never reset".
    if state["recurrent"] and "reset_on" not in state:
        raise SchemaError("recurrent policies must declare state.reset_on")
    if state["recurrent"] and "reset_on" in state and not isinstance(state["reset_on"], list):
        raise SchemaError("state.reset_on must be a list")
    modes = data["inference"].get("modes")
    if not modes:
        raise SchemaError("inference.modes is required")
    tier = data.get("runtime", {}).get("tier", "template")
    from rlx.core.registry import PAYLOAD_LOADERS, ensure_plugins_loaded

    ensure_plugins_loaded()
    payload_case = PAYLOAD_LOADERS.get(str(tier))
    for required_payload in payload_case.required_payload_keys():
        if required_payload not in data["payloads"]:
            raise SchemaError(f"payloads.{required_payload} digest is required")
    if data.get("runtime", {}).get("adapter") == "custom-pytorch":
        from rlx.core.contracts import validate_architecture_spaces

        validate_architecture_spaces(
            observation=data["observation"],
            action=data["action"],
            architecture=data["architecture"],
            adapter="custom-pytorch",
        )
    return data


def validate_task_manifest(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("schema") != TASK_SCHEMA:
        raise SchemaError(f"expected schema {TASK_SCHEMA}, got {data.get('schema')!r}")
    for key in ("name", "adapter", "env", "interaction"):
        if not data.get(key):
            raise SchemaError(f"task manifest missing required field: {key}")
    if data["interaction"] not in {"parallel", "aec", "dynamic_aec"}:
        raise SchemaError("task interaction must be parallel|aec|dynamic_aec")
    declared = data.get("digest")
    if declared is not None:
        actual = task_content_digest(data)
        if declared != actual:
            raise SchemaError(
                f"task digest mismatch: declared {declared!r}, actual {actual!r}"
            )
    return data


def task_content_digest(data: dict[str, Any]) -> str:
    def identity_value(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: identity_value(item)
                for key, item in value.items()
                if not str(key).startswith("_")
            }
        if isinstance(value, list):
            return [identity_value(item) for item in value]
        return value

    identity = {
        key: identity_value(value)
        for key, value in data.items()
        if key not in {"name", "digest"} and not str(key).startswith("_")
    }
    return digest_uri(sha256_bytes(canonical_json(identity)))


def validate_trace_suite(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("schema") != TRACE_SUITE_SCHEMA:
        raise SchemaError(
            f"expected schema {TRACE_SUITE_SCHEMA}, got {data.get('schema')!r}"
        )
    episodes = data.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise SchemaError("trace suite requires a non-empty episodes list")
    for i, episode in enumerate(episodes):
        if not isinstance(episode, dict) or not isinstance(episode.get("actions"), list):
            raise SchemaError(f"trace suite episodes[{i}] requires actions list")
    return data


def validate_match_manifest(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("schema") != MATCH_SCHEMA:
        raise SchemaError(f"expected schema {MATCH_SCHEMA}, got {data.get('schema')!r}")
    for key in ("task", "assignments", "seeds", "action_mode"):
        if key not in data:
            raise SchemaError(f"match manifest missing required field: {key}")
    if data["action_mode"] not in {"deterministic", "stochastic"}:
        raise SchemaError("action_mode must be deterministic|stochastic")
    if not isinstance(data["assignments"], dict) or not data["assignments"]:
        raise SchemaError("assignments must be a non-empty mapping")
    return data


def expand_seeds(seeds_spec: dict[str, Any] | list[int]) -> list[int]:
    if isinstance(seeds_spec, list):
        return [int(s) for s in seeds_spec]
    if "list" in seeds_spec:
        return [int(s) for s in seeds_spec["list"]]
    start = int(seeds_spec.get("start", 0))
    count = int(seeds_spec.get("count", 1))
    return list(range(start, start + count))


def policy_content_digest(manifest: dict[str, Any]) -> str:
    """Digest over canonical policy identity fields.

    Identity is derived purely from executable content (weights, architecture,
    contracts, preprocessing, runtime). Descriptive metadata that varies between
    environments — ``lineage`` (e.g. local source-checkpoint paths) and mutable
    ``conformance`` notes — is intentionally excluded so that two bundles with
    identical behavior receive the same content-addressed identity across
    machines (see docs/EXPECTED.md cross-machine digest promise).
    """
    identity = {
        "schema": manifest.get("schema"),
        "name": manifest.get("name"),
        "roles": manifest.get("roles"),
        "runtime": manifest.get("runtime"),
        "observation": manifest.get("observation"),
        "action": manifest.get("action"),
        "state": manifest.get("state"),
        "inference": manifest.get("inference"),
        "preprocessing": manifest.get("preprocessing"),
        "architecture": manifest.get("architecture"),
        "payloads": manifest.get("payloads"),
    }
    return digest_uri(sha256_bytes(canonical_json(identity)))


def resolve_artifact_path(path: Path | str) -> Path:
    path = Path(path)
    if path.is_dir():
        for name in (
            "policy.yaml",
            "manifest.yaml",
            "match.yaml",
            "population.yaml",
            "evaluation.yaml",
            "dataset.yaml",
        ):
            candidate = path / name
            if candidate.exists():
                return candidate
        raise SchemaError(
            "no policy.yaml/manifest.yaml/match.yaml/population.yaml/"
            f"evaluation.yaml/dataset.yaml in directory: {path}"
        )
    return path


def _require_digest(value: str, *, field: str) -> str:
    text = str(value).strip()
    if not text.startswith("sha256:"):
        raise SchemaError(f"{field} must be a sha256: digest, got {value!r}")
    parse_digest(text)
    return text


def validate_population_manifest(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("schema") != POPULATION_SCHEMA:
        raise SchemaError(f"expected schema {POPULATION_SCHEMA}, got {data.get('schema')!r}")
    if "name" not in data:
        raise SchemaError("population manifest missing required field: name")
    members = data.get("members")
    if not isinstance(members, list) or not members:
        raise SchemaError("population.members must be a non-empty list")
    for i, member in enumerate(members):
        if not isinstance(member, dict):
            raise SchemaError(f"members[{i}] must be a mapping")
        if "policy" not in member:
            raise SchemaError(f"members[{i}].policy is required")
        _require_digest(member["policy"], field=f"members[{i}].policy")
        weight = member.get("weight", 1.0)
        try:
            w = float(weight)
        except (TypeError, ValueError) as e:
            raise SchemaError(f"members[{i}].weight must be a number >= 0") from e
        if w < 0:
            raise SchemaError(f"members[{i}].weight must be >= 0")
        if "tags" in member and not isinstance(member["tags"], list):
            raise SchemaError(f"members[{i}].tags must be a list")
        roles = member.get("roles")
        if roles is not None:
            if not isinstance(roles, dict) or not isinstance(roles.get("allowed", []), list):
                raise SchemaError(f"members[{i}].roles.allowed must be a list when roles is set")
    return data


def population_content_digest(data: dict[str, Any]) -> str:
    """Content identity over member digests/weights/constraints (not human name)."""
    members_identity = []
    for member in data["members"]:
        entry: dict[str, Any] = {
            "policy": member["policy"],
            "weight": float(member.get("weight", 1.0)),
        }
        if "generation" in member:
            entry["generation"] = member["generation"]
        if "tags" in member:
            entry["tags"] = list(member["tags"])
        if "roles" in member:
            entry["roles"] = {
                "allowed": list(member["roles"].get("allowed", [])),
            }
        members_identity.append(entry)
    members_identity.sort(key=lambda m: canonical_json(m))
    identity = {"schema": POPULATION_SCHEMA, "members": members_identity}
    return digest_uri(sha256_bytes(canonical_json(identity)))


def validate_evaluation_manifest(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("schema") != EVALUATION_SCHEMA:
        raise SchemaError(f"expected schema {EVALUATION_SCHEMA}, got {data.get('schema')!r}")
    for key in ("name", "task", "assignments", "seeds", "action_mode", "metrics"):
        if key not in data:
            raise SchemaError(f"evaluation manifest missing required field: {key}")
    if data["action_mode"] not in {"deterministic", "stochastic"}:
        raise SchemaError("action_mode must be deterministic|stochastic")
    provider = data.get("provider", "native")
    if not isinstance(provider, str) or not provider:
        raise SchemaError("evaluation provider must be a non-empty string")
    provider_config = data.get("provider_config", {})
    if not isinstance(provider_config, dict):
        raise SchemaError("evaluation provider_config must be a mapping")
    from rlx.core.registry import EVAL_PROVIDERS, ensure_plugins_loaded

    ensure_plugins_loaded()
    EVAL_PROVIDERS.get(provider)
    declared_task_intent = data.get("task_intent_digest")
    if declared_task_intent is not None:
        _require_digest(declared_task_intent, field="task_intent_digest")
    interaction = data.get("interaction", "parallel")
    if interaction not in {"parallel", "aec", "dynamic_aec"}:
        raise SchemaError("interaction must be parallel|aec|dynamic_aec")
    data = {
        **data,
        "interaction": interaction,
        "provider": provider,
        "provider_config": provider_config,
    }
    assignments = data["assignments"]
    if not isinstance(assignments, dict) or not assignments:
        raise SchemaError("assignments must be a non-empty mapping")
    for role, spec in assignments.items():
        if isinstance(spec, str):
            continue
        if not isinstance(spec, dict):
            raise SchemaError(f"assignments.{role} must be a digest path or mapping")
        kind = spec.get("kind", "policy")
        if kind not in {"policy", "population", "crossplay"}:
            raise SchemaError(
                f"assignments.{role}.kind must be policy|population|crossplay, got {kind!r}"
            )
        if kind == "policy" and "policy" not in spec and "ref" not in spec:
            raise SchemaError(f"assignments.{role} policy kind requires policy or ref")
        if kind == "population" and "population" not in spec:
            raise SchemaError(f"assignments.{role} population kind requires population")
        if kind == "crossplay" and "population" not in spec:
            raise SchemaError(f"assignments.{role} crossplay kind requires population")
    metrics = data["metrics"]
    if not isinstance(metrics, list) or not metrics:
        raise SchemaError("metrics must be a non-empty list")
    budgets = data.get("budgets") or {}
    if not isinstance(budgets, dict):
        raise SchemaError("budgets must be a mapping")
    if "executor" in budgets and budgets["executor"] not in {"process", "thread"}:
        raise SchemaError("budgets.executor must be process|thread")
    if "timeout_seconds" in budgets:
        timeout = budgets["timeout_seconds"]
        if isinstance(timeout, bool):
            raise SchemaError("budgets.timeout_seconds must be a finite number > 0")
        try:
            timeout_value = float(timeout)
        except (TypeError, ValueError) as exc:
            raise SchemaError(
                "budgets.timeout_seconds must be a finite number > 0"
            ) from exc
        if not math.isfinite(timeout_value) or timeout_value <= 0:
            raise SchemaError("budgets.timeout_seconds must be a finite number > 0")
    for field in ("max_stdout_bytes", "max_stderr_bytes"):
        if field in budgets:
            value = budgets[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise SchemaError(f"budgets.{field} must be an integer > 0")
    swaps = data.get("role_swaps", [])
    if swaps is not None and not isinstance(swaps, list):
        raise SchemaError("role_swaps must be a list")
    for i, swap in enumerate(swaps or []):
        if not isinstance(swap, dict) or "map" not in swap:
            raise SchemaError(f"role_swaps[{i}] must include map")
        if "transform" not in swap:
            raise SchemaError(
                f"role_swaps[{i}] requires declared transform "
                "(incompatible swaps must not silently rematch)"
            )
    failure_policy = data.get("failure_policy") or {}
    if not isinstance(failure_policy, dict):
        raise SchemaError("failure_policy must be a mapping")
    missingness = failure_policy.get("missingness", "fail")
    if missingness not in {"fail", "allow"}:
        raise SchemaError("failure_policy.missingness must be fail|allow")
    max_failed = failure_policy.get("max_failed_episodes", 0)
    if isinstance(max_failed, bool):
        raise SchemaError("failure_policy.max_failed_episodes must be an integer >= 0")
    try:
        max_failed_int = int(max_failed)
    except (TypeError, ValueError) as exc:
        raise SchemaError(
            "failure_policy.max_failed_episodes must be an integer >= 0"
        ) from exc
    if max_failed_int < 0 or max_failed_int != max_failed:
        raise SchemaError("failure_policy.max_failed_episodes must be an integer >= 0")
    return data


def evaluation_content_digest(data: dict[str, Any]) -> str:
    """Return the legacy-frozen v0alpha1 evaluation digest.

    This projection is intentionally unchanged from the 0.5 implementation.
    New code should additionally record :func:`evaluation_intent_digest` and
    :func:`evaluation_binding_digest`; changing this function would invalidate
    artifacts produced by earlier RLX releases.
    """
    identity: dict[str, Any] = {
        "schema": EVALUATION_SCHEMA,
        "task": data.get("task"),
        "interaction": data.get("interaction", "parallel"),
        "assignments": data.get("assignments"),
        "seeds": data.get("seeds"),
        "action_mode": data.get("action_mode"),
        "metrics": data.get("metrics"),
        "budgets": data.get("budgets"),
        "role_swaps": data.get("role_swaps", []),
        "failure_policy": data.get("failure_policy"),
        "sampling": data.get("sampling"),
        "recording": data.get("recording"),
    }
    provider = data.get("provider", "native")
    provider_config = data.get("provider_config", {})
    # Preserve 0.2 native-suite identities: the implicit/explicit native provider
    # with empty config is semantically the old execution path. Non-native
    # providers and configured providers remain identity-bearing.
    if provider != "native" or provider_config:
        identity["provider"] = provider
        identity["provider_config"] = provider_config
    return digest_uri(sha256_bytes(canonical_json(identity)))


_OPERATIONAL_KEYS = frozenset(
    {
        "allow_external",
        "auth",
        "base_url",
        "connect_timeout_seconds",
        "container",
        "credential",
        "credentials",
        "endpoint",
        "executor",
        "host",
        "isolation",
        "message_timeout_seconds",
        "process",
        "python",
        "retry",
        "timeout_seconds",
        "token",
        "workers",
    }
)


def _semantic_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _semantic_value(item)
            for key, item in value.items()
            if not str(key).startswith("_") and str(key) not in _OPERATIONAL_KEYS
        }
    if isinstance(value, list):
        return [_semantic_value(item) for item in value]
    return value


def _operational_value(value: Any) -> Any:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): item
        for key, item in value.items()
        if str(key).startswith("_") or str(key) in _OPERATIONAL_KEYS
    }


def task_intent_projection(task: dict[str, Any]) -> dict[str, Any]:
    """Project a task onto semantics shared across execution boundaries."""
    if not isinstance(task, dict):
        raise SchemaError("evaluation task must be a mapping")
    semantic = task.get("semantic")
    if semantic is not None:
        if not isinstance(semantic, dict):
            raise SchemaError("task.semantic must be a mapping")
        projected = _semantic_value(semantic)
    else:
        projected = _semantic_value(
            {
                key: value
                for key, value in task.items()
                if key not in {"adapter", "digest", "name", "packaging", "schema"}
            }
        )
        packaging = _semantic_value(task.get("packaging") or {})
        if packaging:
            projected["packaging"] = packaging
    return {
        "schema": "rlx.task-intent/v1",
        "canonicalization": "rlx-canonical-json/v1",
        "semantics": projected,
    }


def task_intent_digest(task: dict[str, Any]) -> str:
    return digest_uri(sha256_bytes(canonical_json(task_intent_projection(task))))


def evaluation_intent_projection(data: dict[str, Any]) -> dict[str, Any]:
    """Return stable semantic intent, excluding how and where it executes."""
    provider_config = data.get("provider_config") or {}
    if not isinstance(provider_config, dict):
        raise SchemaError("evaluation provider_config must be a mapping")
    semantic_provider = provider_config.get("semantic")
    if semantic_provider is not None:
        if not isinstance(semantic_provider, dict):
            raise SchemaError("provider_config.semantic must be a mapping")
        semantic_provider = _semantic_value(semantic_provider)
    else:
        semantic_provider = _semantic_value(provider_config)
    declared_task_intent = data.get("task_intent_digest")
    if declared_task_intent is not None:
        _require_digest(declared_task_intent, field="task_intent_digest")
    return {
        "schema": EVALUATION_INTENT_SCHEMA,
        "canonicalization": "rlx-canonical-json/v1",
        "task_intent_digest": declared_task_intent
        or task_intent_digest(data.get("task") or {}),
        "interaction": data.get("interaction", "parallel"),
        "assignments": data.get("assignments"),
        "seeds": data.get("seeds"),
        "action_mode": data.get("action_mode"),
        "metrics": data.get("metrics"),
        "budgets": _semantic_value(data.get("budgets") or {}),
        "role_swaps": data.get("role_swaps", []),
        "failure_policy": {
            **_semantic_value(data.get("failure_policy") or {}),
            "missingness": (data.get("failure_policy") or {}).get("missingness", "fail"),
            "max_failed_episodes": (data.get("failure_policy") or {}).get(
                "max_failed_episodes", 0
            ),
        },
        "sampling": data.get("sampling"),
        "recording": data.get("recording"),
        "provider_semantics": semantic_provider,
    }


def evaluation_intent_digest(data: dict[str, Any]) -> str:
    return digest_uri(sha256_bytes(canonical_json(evaluation_intent_projection(data))))


def evaluation_binding_projection(
    data: dict[str, Any],
    *,
    provider: str | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """Return operational execution choices that must be recorded, not conflated."""
    task = data.get("task") or {}
    provider_config = data.get("provider_config") or {}
    task_binding: dict[str, Any] = {
        "adapter": task.get("adapter") if isinstance(task, dict) else None,
    }
    if isinstance(task, dict):
        task_binding.update(_operational_value(task))
        packaging = _operational_value(task.get("packaging") or {})
        if packaging:
            task_binding["packaging"] = packaging
    if "semantic" in provider_config:
        provider_binding = {
            key: value
            for key, value in provider_config.items()
            if key != "semantic" and not str(key).startswith("_")
        }
    else:
        provider_binding = _operational_value(provider_config)
    return {
        "schema": EVALUATION_BINDING_SCHEMA,
        "canonicalization": "rlx-canonical-json/v1",
        "provider": provider or data.get("provider", "native"),
        "provider_config": provider_binding,
        "task": task_binding,
        "budgets": _operational_value(data.get("budgets") or {}),
        "workers": int(workers),
    }


def evaluation_binding_digest(
    data: dict[str, Any],
    *,
    provider: str | None = None,
    workers: int = 1,
) -> str:
    projection = evaluation_binding_projection(data, provider=provider, workers=workers)
    return digest_uri(sha256_bytes(canonical_json(projection)))


def validate_eval_run_manifest(data: dict[str, Any]) -> dict[str, Any]:
    schema = data.get("schema")
    if schema not in {EVAL_RUN_SCHEMA, EVAL_RUN_V1_SCHEMA}:
        raise SchemaError(
            f"expected schema {EVAL_RUN_SCHEMA} or {EVAL_RUN_V1_SCHEMA}, got {schema!r}"
        )
    for key in ("evaluation_digest", "sampling_ledger", "cells"):
        if key not in data:
            raise SchemaError(f"eval-run missing required field: {key}")
    if schema == EVAL_RUN_V1_SCHEMA:
        for key in (
            "evaluation_intent_digest",
            "execution_binding_digest",
            "state",
            "denominators",
            "semantic_result_digest",
        ):
            if key not in data:
                raise SchemaError(f"eval-run/v1 missing required field: {key}")
        if data["state"] not in {"complete", "incomplete", "failed", "cancelled"}:
            raise SchemaError("eval-run/v1 state is invalid")
    return data


def validate_eval_report_manifest(data: dict[str, Any]) -> dict[str, Any]:
    schema = data.get("schema")
    if schema not in {EVAL_REPORT_SCHEMA, EVAL_REPORT_V1_SCHEMA}:
        raise SchemaError(
            f"expected schema {EVAL_REPORT_SCHEMA} or {EVAL_REPORT_V1_SCHEMA}, "
            f"got {schema!r}"
        )
    for key in ("evaluation_digest", "eval_run_digest", "metrics"):
        if key not in data:
            raise SchemaError(f"eval-report missing required field: {key}")
    return data


def validate_dataset_manifest(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("schema") != DATASET_SCHEMA:
        raise SchemaError(f"expected schema {DATASET_SCHEMA}, got {data.get('schema')!r}")
    for key in ("name", "source_runs", "episodes", "query"):
        if key not in data:
            raise SchemaError(f"dataset missing required field: {key}")
    if not isinstance(data["episodes"], list):
        raise SchemaError("dataset.episodes must be a list of episode digests/paths")
    split_names: set[str] = set()
    for index, episode in enumerate(data["episodes"]):
        if not isinstance(episode, dict):
            raise SchemaError(f"dataset.episodes[{index}] must be a mapping")
        if episode.get("split") is not None:
            split = str(episode["split"])
            if not split:
                raise SchemaError(f"dataset.episodes[{index}].split must be non-empty")
            split_names.add(split)
    if data.get("splits") is not None:
        splits = data["splits"]
        if not isinstance(splits, dict) or splits.get("method") != "sha256_bucket/v1":
            raise SchemaError("dataset.splits.method must be sha256_bucket/v1")
        weights = splits.get("weights")
        counts = splits.get("counts")
        if not isinstance(weights, dict) or not weights:
            raise SchemaError("dataset.splits.weights must be a non-empty mapping")
        if not isinstance(counts, dict) or set(counts) != set(weights):
            raise SchemaError("dataset.splits.counts must cover split weights exactly")
        if split_names - set(weights):
            raise SchemaError(
                f"episode split names missing from dataset.splits: "
                f"{sorted(split_names - set(weights))}"
            )
    return data


def validate_eval_bundle_manifest(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("schema") != EVAL_BUNDLE_SCHEMA:
        raise SchemaError(f"expected schema {EVAL_BUNDLE_SCHEMA}, got {data.get('schema')!r}")
    for key in ("evaluation_digest", "artifacts"):
        if key not in data:
            raise SchemaError(f"eval-bundle missing required field: {key}")
    return data
