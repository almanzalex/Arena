"""Gymnasium-compatible space descriptors for compatibility checks."""

from __future__ import annotations

import math
from typing import Any


def normalize_bound_value(value: Any) -> Any:
    """Encode unbounded Box limits without non-finite JSON numbers."""

    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [normalize_bound_value(item) for item in value]
    if isinstance(value, float) and math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


def decode_bound_value(value: Any) -> Any:
    """Decode Arena's explicit JSON-safe Box-bound sentinels."""

    if isinstance(value, list):
        return [decode_bound_value(item) for item in value]
    if value == "inf":
        return math.inf
    if value == "-inf":
        return -math.inf
    return value


def normalize_space_descriptor(data: dict[str, Any]) -> dict[str, Any]:
    """Copy a typed space and normalize all nested Box bounds."""

    result = dict(data)
    if result.get("type") == "Box":
        for key in ("low", "high"):
            if key in result:
                result[key] = normalize_bound_value(result[key])
    if result.get("type") == "Dict" and isinstance(result.get("spaces"), dict):
        result["spaces"] = {
            str(key): normalize_space_descriptor(value)
            for key, value in result["spaces"].items()
        }
    return result


def space_from_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize a space dict; unknown fields preserved."""
    if "type" not in data:
        raise ValueError("space requires 'type'")
    return normalize_space_descriptor(data)


def spaces_compatible(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    """Return list of mismatch messages (empty if compatible)."""
    expected = normalize_space_descriptor(expected)
    actual = normalize_space_descriptor(actual)
    mismatches: list[str] = []
    if expected.get("type") != actual.get("type"):
        mismatches.append(f"type {actual.get('type')!r} != {expected.get('type')!r}")
        return mismatches

    stype = expected["type"]
    if stype == "Discrete":
        if expected.get("n") != actual.get("n"):
            mismatches.append(f"n {actual.get('n')!r} != {expected.get('n')!r}")
    elif stype == "Box":
        for key in ("shape", "dtype"):
            if key in expected and expected[key] != actual.get(key):
                mismatches.append(f"{key} {actual.get(key)!r} != {expected[key]!r}")
        for key in ("low", "high"):
            if key in expected and key in actual and expected[key] != actual[key]:
                mismatches.append(f"{key} mismatch")
        exp_dist = expected.get("distribution", "deterministic")
        act_dist = actual.get("distribution", "deterministic")
        if exp_dist != act_dist:
            mismatches.append(f"distribution {act_dist!r} != {exp_dist!r}")
        # Layout is part of the image contract when the task declares it.
        # HWC vs CHW must not silently pass; policy-only layout is ignored so
        # legacy non-image tasks keep working.
        exp_layout = expected.get("layout")
        act_layout = actual.get("layout")
        if exp_layout is not None and exp_layout != act_layout:
            mismatches.append(
                f"layout {act_layout!r} != {exp_layout!r} "
                "(declare matching policy observation.layout / task.observation_layout)"
            )
    elif stype == "MultiDiscrete":
        if expected.get("nvec") != actual.get("nvec"):
            mismatches.append("nvec mismatch")
        # Optional case-contract fields: when either side declares them, they must match.
        for key in ("logit_layout", "sampling_order", "masks"):
            if key in expected and key in actual and expected[key] != actual.get(key):
                mismatches.append(f"{key} mismatch")
    elif stype == "Dict":
        exp_order = expected.get("key_order")
        act_order = actual.get("key_order")
        if exp_order is not None and exp_order != act_order:
            mismatches.append(f"key_order {act_order!r} != {exp_order!r}")
        exp_spaces = expected.get("spaces") or {}
        act_spaces = actual.get("spaces") or {}
        if set(exp_spaces) != set(act_spaces):
            mismatches.append(
                f"Dict fields {sorted(act_spaces)!r} != {sorted(exp_spaces)!r}"
            )
        else:
            for key in exp_order or sorted(exp_spaces):
                mismatches.extend(
                    f"field {key}: {m}"
                    for m in spaces_compatible(exp_spaces[key], act_spaces[key])
                )
    elif stype == "MultiBinary":
        if expected.get("n") != actual.get("n"):
            mismatches.append(f"n {actual.get('n')!r} != {expected.get('n')!r}")
    else:
        # Best-effort equality for unknown types
        if expected != actual:
            mismatches.append(f"space descriptors differ for type {stype!r}")
    return mismatches


def gymnasium_space_to_dict(space: Any) -> dict[str, Any]:
    """Convert a Gymnasium space object to an Arena space dict."""
    import numpy as np

    name = type(space).__name__
    if name == "Discrete":
        return {"type": "Discrete", "n": int(space.n), "dtype": "int64"}
    if name == "Box":
        shape = tuple(int(x) for x in space.shape)
        low = space.low
        high = space.high
        # Collapse scalars when uniform
        low_v: Any
        high_v: Any
        if np.allclose(low, low.flat[0]) and np.allclose(high, high.flat[0]):
            low_v = normalize_bound_value(float(low.flat[0]))
            high_v = normalize_bound_value(float(high.flat[0]))
        else:
            low_v = normalize_bound_value(low)
            high_v = normalize_bound_value(high)
        return {
            "type": "Box",
            "shape": list(shape),
            "dtype": str(space.dtype),
            "low": low_v,
            "high": high_v,
        }
    if name == "MultiDiscrete":
        return {"type": "MultiDiscrete", "nvec": [int(x) for x in space.nvec]}
    if name == "Dict":
        # Preserve insertion order from the Gymnasium Dict as a suggested key_order;
        # policies must still declare an explicit canonical key_order.
        keys = list(space.spaces.keys())
        return {
            "type": "Dict",
            "key_order": keys,
            "spaces": {k: gymnasium_space_to_dict(space.spaces[k]) for k in keys},
        }
    if name == "MultiBinary":
        return {"type": "MultiBinary", "n": int(space.n)}
    return {"type": name, "repr": repr(space)}
