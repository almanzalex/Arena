"""Fail-loud policy contract checks (architecture ↔ spaces ↔ runtime).

Action schemas are a discriminated union of typed cases (Discrete, MultiDiscrete,
deterministic Box, diagonal-Gaussian Box, recursive Dict). Template categorical
actors remain Discrete-only; BYO TorchScript may use the fuller cases when the
case contract is complete. Cross-field desync and incomplete cases are rejected —
never silently padded, flattened, or coerced into a different action type.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rlx.core.action_cases import (
    validate_action_case,
    validate_expected_action,
)
from rlx.core.errors import CompatibilityIssue, ConformanceError, SchemaError

# Architectures the custom-pytorch adapter can actually execute.
_CATEGORICAL_ARCHS = frozenset({"mlp_categorical", "gru_categorical"})
_SERIALIZED_ARCHS = frozenset({"serialized_module", "trusted_source_module"})


def observation_flat_size(observation: dict[str, Any]) -> int | None:
    """Return expected flat observation length, or None if inexpressible."""
    otype = observation.get("type")
    if otype == "Discrete":
        n = observation.get("n")
        return int(n) if n is not None else None
    if otype == "Box":
        shape = observation.get("shape")
        if shape is None:
            return None
        return int(np.prod([int(x) for x in shape]))
    return None


def validate_architecture_spaces(
    *,
    observation: dict[str, Any],
    action: dict[str, Any],
    architecture: dict[str, Any],
    adapter: str | None = None,
) -> None:
    """Reject architecture / space lies that the runtime cannot honor.

    Raises SchemaError with an actionable repair message.
    """
    if adapter is not None and adapter != "custom-pytorch":
        return
    atype = architecture.get("type")
    if atype in _SERIALIZED_ARCHS:
        _validate_serialized_spaces(observation, action, architecture)
        return
    if atype not in _CATEGORICAL_ARCHS:
        # Unknown arch types are rejected elsewhere (build_module); skip here.
        return

    obs_dim = architecture.get("observation_dim")
    action_n = architecture.get("action_n")
    if obs_dim is None or action_n is None:
        raise SchemaError(
            "architecture must declare observation_dim and action_n for "
            f"{atype!r} (got observation_dim={obs_dim!r}, action_n={action_n!r})"
        )
    obs_dim = int(obs_dim)
    action_n = int(action_n)

    # Templates: Discrete-only. Other action types fail with case-specific guidance.
    validate_action_case(action, architecture=architecture, adapter=adapter)
    act_type = action.get("type")
    if act_type != "Discrete":
        # validate_action_case already raised for MultiDiscrete/Box/Dict on templates.
        raise SchemaError(
            f"custom-pytorch {atype} only supports Discrete categorical int actions; "
            f"got action.type={act_type!r}. Re-export with a Discrete action space, "
            "or use a BYO TorchScript actor with a complete action-type case."
        )
    if int(action["n"]) != int(action_n):
        raise SchemaError(
            f"architecture.action_n ({action_n}) disagrees with action.n ({int(action['n'])}). "
            "Align both fields to the true Discrete action count before export/verify."
        )

    otype = observation.get("type")
    if otype == "Box":
        shape = observation.get("shape")
        if shape is None:
            raise SchemaError("Box observation requires 'shape'")
        shape_t = tuple(int(x) for x in shape)
        if len(shape_t) >= 3 and "layout" not in observation:
            raise SchemaError(
                f"Box observation shape {list(shape_t)} is rank≥3 without an explicit "
                "'layout' (e.g. CHW or HWC). Flatten order is ambiguous — declare "
                "layout or export a flat vector observation (see RFC 002)."
            )
        flat = int(np.prod(shape_t))
        if flat != obs_dim:
            raise SchemaError(
                f"architecture.observation_dim ({obs_dim}) disagrees with "
                f"observation.shape product ({flat} from shape={list(shape_t)}). "
                "Align both to the true flat observation size."
            )
    elif otype == "Discrete":
        n = observation.get("n")
        if n is None:
            raise SchemaError("Discrete observation requires 'n'")
        if int(n) != obs_dim:
            raise SchemaError(
                f"architecture.observation_dim ({obs_dim}) disagrees with "
                f"observation.n ({int(n)}) for Discrete one-hot encoding. "
                "Set observation_dim == observation.n."
            )
    else:
        raise SchemaError(
            f"custom-pytorch {atype} only supports Box or Discrete observations; "
            f"got observation.type={otype!r}."
        )


def _validate_serialized_spaces(
    observation: dict[str, Any], action: dict[str, Any], architecture: dict[str, Any]
) -> None:
    """Validate the explicit I/O contract for a captured Torch actor."""
    if observation.get("type") != "Box":
        raise SchemaError("serialized_module currently requires a Box observation contract")
    shape = observation.get("shape")
    if not isinstance(shape, list) or not shape or any(int(x) <= 0 for x in shape):
        raise SchemaError("serialized_module Box observation requires a non-empty positive shape")
    layout = observation.get("layout")
    if len(shape) >= 3 and layout not in {"CHW", "HWC"}:
        raise SchemaError("rank≥3 serialized_module observations require layout CHW or HWC")
    if layout == "CHW" and len(shape) != 3:
        raise SchemaError("layout CHW requires exactly three image dimensions")
    if layout == "HWC" and len(shape) != 3:
        raise SchemaError("layout HWC requires exactly three image dimensions")
    validate_action_case(
        action,
        architecture=architecture,
        adapter="custom-pytorch",
        require_byo_layout=True,
    )


def architecture_space_issues(policy: dict[str, Any]) -> list[CompatibilityIssue]:
    """Compatibility-report form of :func:`validate_architecture_spaces`."""
    adapter = policy.get("runtime", {}).get("adapter")
    architecture = policy.get("architecture")
    if not isinstance(architecture, dict):
        return []
    if adapter not in {None, "custom-pytorch"}:
        return []
    if architecture.get("type") not in _CATEGORICAL_ARCHS | _SERIALIZED_ARCHS:
        return []
    try:
        validate_architecture_spaces(
            observation=policy.get("observation") or {},
            action=policy.get("action") or {},
            architecture=architecture,
            adapter="custom-pytorch",
        )
    except SchemaError as e:
        return [
            CompatibilityIssue(
                code="ARCHITECTURE_CONTRACT",
                message=str(e),
                evidence={
                    "architecture": architecture,
                    "observation": policy.get("observation"),
                    "action": policy.get("action"),
                },
                repairs=[
                    "align architecture.observation_dim with observation shape/n",
                    "align architecture.action_n with action.n for Discrete templates",
                    "declare a complete action-type case (Discrete / MultiDiscrete / "
                    "Box / Dict) — never flatten or coerce spaces",
                ],
            )
        ]
    return []


def validate_reference_case_action(
    case: dict[str, Any],
    *,
    action: dict[str, Any],
    index: int | None = None,
) -> None:
    """Reject illegal expected_action values (OOB or masked-illegal).

    Must run before stamping conformance verified.
    """
    validate_expected_action(case, action=action, index=index)


def check_observation_vector(
    observation: Any,
    *,
    obs_space: dict[str, Any],
    obs_dim: int,
) -> np.ndarray:
    """Encode an observation fail-loud (no pad/truncate/clip-to-fit)."""
    otype = obs_space.get("type")
    arr = np.asarray(observation)

    if otype == "Discrete":
        n = int(obs_space["n"])
        if obs_dim != n:
            raise ConformanceError(
                f"architecture.observation_dim ({obs_dim}) != observation.n ({n})"
            )
        if arr.ndim == 0 or arr.shape == ():
            try:
                idx = int(arr)
            except (TypeError, ValueError) as e:
                raise ConformanceError(
                    f"Discrete observation {observation!r} is not an integer index"
                ) from e
            if not (0 <= idx < n):
                raise ConformanceError(
                    f"Discrete observation index {idx} out of range [0, {n})"
                )
            x = np.zeros(obs_dim, dtype=np.float32)
            x[idx] = 1.0
            return x
        flat = np.asarray(observation, dtype=np.float32).reshape(-1)
        # Exact length-n vectors are treated as already-encoded features.
        # Length-1 arrays are NOT silently unwrapped (that was a silent-wrong trap).
        if flat.size == n:
            return flat
        raise ConformanceError(
            f"Discrete observation must be a scalar index in [0, {n}) or a "
            f"length-{n} vector; got shape {arr.shape}. Refusing to pad/truncate "
            "or coerce shape(1,) to a scalar."
        )

    # Box / vector path
    if arr.ndim == 0 or arr.shape == ():
        raise ConformanceError(
            f"expected a length-{obs_dim} observation vector "
            f"(observation.type={otype!r}, shape={obs_space.get('shape')!r}); "
            f"got scalar {observation!r}. Refusing to pad."
        )
    if otype == "Box":
        shape = tuple(int(x) for x in (obs_space.get("shape") or ()))
        if shape and arr.shape != shape and arr.reshape(-1).size == int(np.prod(shape)):
            # Allow flat lists that match the declared product, but not reshape lies
            # that change rank ambiguously for rank≥3 without layout (already banned
            # at export). For 1D contracts, require exact flat size below.
            pass
        elif shape and len(shape) > 1 and arr.shape != shape:
            raise ConformanceError(
                f"observation shape {tuple(arr.shape)} != declared Box shape {shape}. "
                "Pass an array with the declared shape (or its exact flat length)."
            )
    x = np.asarray(observation, dtype=np.float32).reshape(-1)
    if x.size != obs_dim:
        raise ConformanceError(
            f"observation length {x.size} != architecture.observation_dim / "
            f"declared flat size {obs_dim}. Refusing to pad or truncate."
        )
    return x
