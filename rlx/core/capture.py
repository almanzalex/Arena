"""Capture-from-source helpers: draft contracts from instrumented actors/envs.

Best-effort only. A human must confirm the draft before publish. Limits are
documented on each helper and in docs/policy-export.md.
"""

from __future__ import annotations

from typing import Any

from rlx.core.errors import SchemaError
from rlx.core.registry import ensure_plugins_loaded
from rlx.core.spaces import gymnasium_space_to_dict


def capture_action_case_from_space(space: Any, *, masks: str = "none") -> dict[str, Any]:
    """Draft an action-case dict from a Gymnasium/PettingZoo space.

    Limits:
    - Discrete / MultiDiscrete / Box / Dict only (registry-known types).
    - MultiDiscrete drafts ``logit_layout: {kind: concatenated}`` but does not
      invent non-contiguous slices.
    - Stochastic Box is never inferred; drafts ``distribution: deterministic``.
      Humans must upgrade to ``diagonal_gaussian`` with full RNG/transform fields.
    - Dict drafts canonical ``key_order`` from the space's iteration order and
      nested typed spaces; ``param_layout`` is not invented (must be confirmed).
    """
    ensure_plugins_loaded()
    draft = gymnasium_space_to_dict(space)
    atype = draft.get("type")
    if atype == "Discrete":
        draft["masks"] = masks
        return draft
    if atype == "MultiDiscrete":
        draft["masks"] = masks
        draft["logit_layout"] = {"kind": "concatenated"}
        draft["sampling_order"] = "sequential"
        return draft
    if atype == "Box":
        draft["masks"] = masks
        draft["distribution"] = "deterministic"
        return draft
    if atype == "Dict":
        spaces = draft.get("spaces") or {}
        draft["masks"] = masks
        draft["key_order"] = list(spaces.keys())
        # Nested fields get masks; param_layout left for human confirmation.
        nested = {}
        for key, child in spaces.items():
            child = dict(child)
            child.setdefault("masks", "none")
            if child.get("type") == "MultiDiscrete":
                child.setdefault("logit_layout", {"kind": "concatenated"})
                child.setdefault("sampling_order", "sequential")
            if child.get("type") == "Box":
                child.setdefault("distribution", "deterministic")
            nested[key] = child
        draft["spaces"] = nested
        draft["needs_human_confirm"] = ["param_layout"]
        return draft
    raise SchemaError(
        f"capture-from-source cannot draft action case for space type {atype!r}. "
        "Register a new action case and teach the capture helper, then re-run "
        "`rlx adapter qualify` before claiming support."
    )


def capture_wrapper_hints(wrappers: list[Any] | None = None) -> dict[str, Any]:
    """Best-effort wrapper hints from duck-typed env wrappers.

    Limits: only recognizes color_reduction / resize / frame_stack-like attributes.
    Unknown wrappers are listed under ``unknown`` and must not be silently skipped
    at export — the author must declare them or remove them.
    """
    ensure_plugins_loaded()
    from rlx.core.registry import WRAPPER_OPS

    known = sorted(WRAPPER_OPS.known())
    declared: list[dict[str, Any]] = []
    unknown: list[str] = []
    for w in wrappers or []:
        name = type(w).__name__.lower()
        hint: dict[str, Any] | None = None
        if "color" in name and "reduc" in name:
            hint = {"op": "color_reduction", "mode": getattr(w, "mode", "full")}
        elif "resize" in name:
            x = getattr(w, "x_size", getattr(w, "width", None))
            y = getattr(w, "y_size", getattr(w, "height", None))
            if x is not None and y is not None:
                hint = {"op": "resize", "x_size": int(x), "y_size": int(y)}
        elif "stack" in name or "framestack" in name.replace("_", ""):
            k = getattr(w, "stack_size", getattr(w, "k", getattr(w, "num_stack", None)))
            if k is not None:
                hint = {"op": "frame_stack", "stack_size": int(k)}
        if hint is None:
            unknown.append(type(w).__name__)
        else:
            declared.append(hint)
    return {
        "draft_wrappers": declared,
        "unknown": unknown,
        "registered_wrapper_ops": known,
        "needs_human_confirm": True,
        "limits": (
            "Best-effort duck-typing only. Unknown wrappers must be declared or "
            "removed before publish; RLX will not silently skip them."
        ),
    }


def capture_draft_from_env(env: Any, *, agent: str | None = None) -> dict[str, Any]:
    """Observe spaces from a Parallel env and emit a draft task/policy contract.

    Limits: does not invent stochastic distributions, Dict param_layout, or
    preprocess pipelines. Human confirmation required before publish.
    """
    ensure_plugins_loaded()
    agents = list(getattr(env, "agents", []) or getattr(env, "possible_agents", []))
    if not agents:
        raise SchemaError("capture_draft_from_env: env exposes no agents")
    target = agent or agents[0]
    if not hasattr(env, "observation_space") or not hasattr(env, "action_space"):
        raise SchemaError("capture_draft_from_env requires observation_space/action_space")
    obs_space = env.observation_space(target)
    act_space = env.action_space(target)
    obs = gymnasium_space_to_dict(obs_space)
    action = capture_action_case_from_space(act_space)
    return {
        "agent": target,
        "agents": agents,
        "observation": obs,
        "action": action,
        "packaging": {"kind": "pettingzoo_wrappers"},
        "needs_human_confirm": True,
        "limits": (
            "Draft only. Confirm action case completeness (especially MultiDiscrete "
            "logit_layout, Dict param_layout, stochastic Box RNG/transform), "
            "wrapper chain, and preprocess IR before publish. Run "
            "`rlx adapter qualify` before claiming support."
        ),
    }


def capture_draft_from_actor(
    *,
    observation: dict[str, Any] | None = None,
    action_space: Any | None = None,
    action: dict[str, Any] | None = None,
    preferred_payload: str = "torchscript",
) -> dict[str, Any]:
    """Draft a policy contract from spaces (and optional pre-built action case).

    Limits: does not introspect nn.Module graphs; payload preference defaults to
    TorchScript. ``trusted_source`` is never the default.
    """
    ensure_plugins_loaded()
    from rlx.core.registry import PAYLOAD_LOADERS

    if preferred_payload not in PAYLOAD_LOADERS:
        PAYLOAD_LOADERS.get(preferred_payload)  # raise extension recipe
    if action is None:
        if action_space is None:
            raise SchemaError("capture_draft_from_actor requires action or action_space")
        action = capture_action_case_from_space(action_space)
    if observation is None:
        observation = {"type": "Box", "shape": [], "dtype": "float32"}
        observation["needs_human_confirm"] = True
    return {
        "observation": observation,
        "action": action,
        "runtime": {"tier": preferred_payload, "adapter": "custom-pytorch"},
        "needs_human_confirm": True,
        "limits": (
            "Does not script/trace the module. Prefer torchscript; trusted_source "
            "requires explicit --trust-source and is not sandboxed."
        ),
    }
