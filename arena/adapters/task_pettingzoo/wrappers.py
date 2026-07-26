"""Declarative PettingZoo / SuperSuit task wrapper contract.

Wrappers are applied on the *task* side so observation spaces match the training
stack (e.g. CleanRL Pistonball: color_reduction → resize → frame_stack). Missing
or unknown wrappers fail loud via the wrapper-op registry; Arena never silently
compares against unwrapped spaces.
"""

from __future__ import annotations

from typing import Any

from arena.core.errors import ArenaError, SchemaError
from arena.core.registry import WRAPPER_OPS, ensure_plugins_loaded
from arena.plugins.wrappers import require_supersuit, resolve_wrapper_kind


def normalize_wrappers(wrappers: Any) -> list[dict[str, Any]]:
    """Validate and normalize a declarative wrapper chain via the wrapper registry."""
    ensure_plugins_loaded()
    if wrappers is None:
        return []
    if not isinstance(wrappers, list):
        raise SchemaError(
            "task.wrappers must be a list of wrapper step dicts "
            f"(got {type(wrappers).__name__})"
        )
    normalized: list[dict[str, Any]] = []
    for i, step in enumerate(wrappers):
        if not isinstance(step, dict):
            raise SchemaError(f"task.wrappers[{i}] must be a dict, got {type(step).__name__}")
        raw_op = step.get("op") or step.get("name")
        if not raw_op:
            raise SchemaError(f"task.wrappers[{i}] requires 'op'")
        try:
            op_kind = resolve_wrapper_kind(str(raw_op))
        except SchemaError as e:
            # Preserve the historical "unknown task wrapper op" phrasing while
            # still including the extension recipe from the registry.
            raise SchemaError(
                f"unknown task wrapper op {raw_op!r} at wrappers[{i}]; {e}"
            ) from e
        op = WRAPPER_OPS.get(op_kind)
        entry = dict(step)
        entry["op"] = op_kind
        entry.pop("name", None)
        entry = op.normalize(entry, index=i)
        entry["op"] = op_kind
        normalized.append(entry)
    return normalized


def apply_wrappers(env: Any, wrappers: list[dict[str, Any]] | None) -> Any:
    """Apply a validated wrapper chain via SuperSuit. Empty chain is a no-op."""
    ensure_plugins_loaded()
    steps = normalize_wrappers(wrappers)
    if not steps:
        return env
    ss = require_supersuit()
    for i, step in enumerate(steps):
        op = WRAPPER_OPS.get(step["op"])
        try:
            env = op.apply(env, step, index=i, supersuit=ss)
        except SchemaError:
            raise
        except Exception as e:  # noqa: BLE001 — surface SuperSuit failures loudly
            raise ArenaError(
                f"failed applying task.wrappers[{i}] op={step['op']!r}: {e}. "
                "Check wrapper order and parameters against the training stack; "
                "Arena will not fall back to the unwrapped environment."
            ) from e
    return env


def wrappers_provenance(wrappers: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Stable provenance record for run/task manifests."""
    ensure_plugins_loaded()
    steps = normalize_wrappers(wrappers)
    record: dict[str, Any] = {
        "chain": [{"op": s["op"], **{k: v for k, v in s.items() if k != "op"}} for s in steps],
        "identity": _chain_identity(steps),
    }
    if steps:
        try:
            import supersuit

            record["backend"] = "supersuit"
            record["supersuit_version"] = getattr(supersuit, "__version__", "unknown")
        except ImportError:
            record["backend"] = "supersuit-required-but-missing"
    else:
        record["backend"] = "none"
    return record


def _chain_identity(steps: list[dict[str, Any]]) -> str:
    """Compact human-readable chain id, e.g. color_reduction>resize(64,64)>frame_stack(4)."""
    ensure_plugins_loaded()
    if not steps:
        return "unwrapped"
    parts: list[str] = []
    for s in steps:
        op = WRAPPER_OPS.get(s["op"])
        parts.append(op.identity_part(s))
    return ">".join(parts)


# Back-compat export for tests that imported the frozenset.
SUPPORTED_WRAPPER_OPS = frozenset({"color_reduction", "resize", "frame_stack"})
