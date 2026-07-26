"""Discriminated action-type cases for portable Arena policies.

Action schemas are a registry of typed cases. Incomplete or mismatched
declarations fail before publish/load — never silently flatten MultiDiscrete to
Discrete, coerce Dict↔vector, or treat stochastic Box as deterministic.
Unknown kinds fail loud with an extension recipe (see ``arena.core.registry``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from arena.core.errors import ConformanceError, SchemaError

# Back-compat snapshots; live membership is ``ACTION_CASES`` / ``DISTRIBUTIONS``.
SUPPORTED_ACTION_TYPES = frozenset({"Discrete", "MultiDiscrete", "Box", "Dict"})
BOX_DISTRIBUTIONS = frozenset({"deterministic", "diagonal_gaussian"})
STOCHASTIC_BOX_TRANSFORMS = ("sample", "tanh", "affine")
RNG_ALGORITHMS = frozenset({"numpy_generator"})


def action_type(action: dict[str, Any]) -> str:
    atype = action.get("type")
    if not isinstance(atype, str) or not atype:
        raise SchemaError("action.type is required")
    return atype


def require_masks(action: dict[str, Any]) -> str:
    masks = action.get("masks")
    if masks not in {"none", "optional", "required"}:
        raise SchemaError("action.masks must be none|optional|required")
    return str(masks)


def _action_case(kind: str):
    from arena.core.registry import ACTION_CASES, ensure_plugins_loaded

    ensure_plugins_loaded()
    return ACTION_CASES.get(kind)


def multidiscrete_nvec(action: dict[str, Any]) -> list[int]:
    nvec = action.get("nvec")
    if not isinstance(nvec, list) or not nvec or any(int(x) <= 0 for x in nvec):
        raise SchemaError("MultiDiscrete requires a non-empty positive nvec")
    return [int(x) for x in nvec]


def multidiscrete_slices(action: dict[str, Any]) -> list[tuple[int, int]]:
    """Return exclusive logit slices per factor under the declared layout."""
    nvec = multidiscrete_nvec(action)
    layout = action.get("logit_layout")
    if not isinstance(layout, dict):
        raise SchemaError(
            "MultiDiscrete case is incomplete: declare logit_layout "
            "(e.g. {kind: concatenated} with slices matching nvec). "
            "Arena will not flatten MultiDiscrete to Discrete."
        )
    kind = layout.get("kind")
    if kind != "concatenated":
        raise SchemaError(
            f"unsupported MultiDiscrete logit_layout.kind={kind!r}; "
            "supported: 'concatenated' (contiguous per-factor logit blocks)"
        )
    raw_slices = layout.get("slices")
    if raw_slices is None:
        # Derived contiguous layout is allowed when kind is explicitly declared.
        start = 0
        slices: list[tuple[int, int]] = []
        for n in nvec:
            slices.append((start, start + n))
            start += n
        return slices
    if not isinstance(raw_slices, list) or len(raw_slices) != len(nvec):
        raise SchemaError(
            f"MultiDiscrete logit_layout.slices length {len(raw_slices) if isinstance(raw_slices, list) else 'n/a'} "
            f"must equal len(nvec)={len(nvec)}"
        )
    slices = []
    expected_start = 0
    for i, (spec, n) in enumerate(zip(raw_slices, nvec, strict=True)):
        if not isinstance(spec, (list, tuple)) or len(spec) != 2:
            raise SchemaError(f"MultiDiscrete logit slice {i} must be [start, end)")
        start, end = int(spec[0]), int(spec[1])
        if start != expected_start or end - start != n:
            raise SchemaError(
                f"MultiDiscrete logit slice {i}={spec!r} disagrees with contiguous "
                f"nvec layout (expected [{expected_start}, {expected_start + n}))"
            )
        slices.append((start, end))
        expected_start = end
    return slices


def multidiscrete_sampling_order(action: dict[str, Any]) -> list[int]:
    nvec = multidiscrete_nvec(action)
    order = action.get("sampling_order", "sequential")
    if order == "sequential":
        return list(range(len(nvec)))
    if not isinstance(order, list) or sorted(int(x) for x in order) != list(range(len(nvec))):
        raise SchemaError(
            "MultiDiscrete sampling_order must be 'sequential' or a permutation of factor indices"
        )
    return [int(x) for x in order]


def multidiscrete_logit_width(action: dict[str, Any]) -> int:
    slices = multidiscrete_slices(action)
    return int(slices[-1][1]) if slices else 0


def box_distribution(action: dict[str, Any]) -> str:
    from arena.core.registry import DISTRIBUTIONS, ensure_plugins_loaded

    ensure_plugins_loaded()
    dist = action.get("distribution", "deterministic")
    # Fail loud with extension recipe for unknown kinds (e.g. mixtures).
    DISTRIBUTIONS.get(str(dist))
    return str(dist)


def box_shape(action: dict[str, Any]) -> tuple[int, ...]:
    shape = action.get("shape")
    if not isinstance(shape, list) or not shape or any(int(x) <= 0 for x in shape):
        raise SchemaError("Box action requires a non-empty positive shape")
    return tuple(int(x) for x in shape)


def box_bounds(action: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    shape = box_shape(action)
    size = int(np.prod(shape))
    low = np.asarray(action.get("low"), dtype=np.float64)
    high = np.asarray(action.get("high"), dtype=np.float64)
    if low.size not in {1, size} or high.size not in {1, size}:
        raise SchemaError("Box action.low/high must be scalar or match action.shape")
    if np.any(low >= high):
        raise SchemaError("Box action requires low < high elementwise")
    return low, high


def validate_stochastic_box_contract(action: dict[str, Any]) -> None:
    """Require the full diagonal-Gaussian portable contract via the distribution registry."""
    from arena.core.registry import DISTRIBUTIONS, ensure_plugins_loaded

    ensure_plugins_loaded()
    if box_distribution(action) != "diagonal_gaussian":
        return
    DISTRIBUTIONS.get("diagonal_gaussian").validate(action)


def dict_key_order(action: dict[str, Any]) -> list[str]:
    order = action.get("key_order")
    spaces = action.get("spaces")
    if not isinstance(spaces, dict) or not spaces:
        raise SchemaError(
            "Dict action case is incomplete: declare typed spaces for each field. "
            "Open/dynamic untyped Dict actions are rejected."
        )
    if not isinstance(order, list) or not order:
        raise SchemaError(
            "Dict action case is incomplete: declare canonical key_order. "
            "Unstable key order is rejected (Arena will not invent a sort)."
        )
    keys = [str(k) for k in order]
    if len(keys) != len(set(keys)):
        raise SchemaError("Dict key_order contains duplicates")
    missing = [k for k in keys if k not in spaces]
    extra = [k for k in spaces if k not in keys]
    if missing or extra:
        raise SchemaError(
            f"Dict key_order/spaces mismatch (missing_in_spaces={missing}, "
            f"extra_in_spaces={extra}). Declare every field exactly once."
        )
    return keys


def dict_field_actions(action: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    keys = dict_key_order(action)
    spaces = action["spaces"]
    return [(k, spaces[k]) for k in keys]


def dict_param_slices(action: dict[str, Any]) -> list[tuple[str, str, int, int]]:
    """Return (key, kind, start, end) slices into the flat actor output."""
    layout = action.get("param_layout")
    if not isinstance(layout, dict) or layout.get("kind") != "concatenated_fields":
        raise SchemaError(
            "Dict action case is incomplete for BYO/serialized actors: declare "
            "param_layout {kind: concatenated_fields, fields: {key: {kind, slice}}}. "
            "Arena will not coerce Dict↔vector."
        )
    fields = layout.get("fields")
    if not isinstance(fields, dict):
        raise SchemaError("Dict param_layout.fields must be a mapping")
    out: list[tuple[str, str, int, int]] = []
    expected_start = 0
    for key, field in dict_field_actions(action):
        spec = fields.get(key)
        if not isinstance(spec, dict):
            raise SchemaError(f"Dict param_layout missing field {key!r}")
        kind = spec.get("kind")
        raw = spec.get("slice")
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise SchemaError(f"Dict param_layout.fields[{key!r}].slice must be [start, end)")
        start, end = int(raw[0]), int(raw[1])
        if start != expected_start or end <= start:
            raise SchemaError(
                f"Dict param_layout field {key!r} slice [{start}, {end}) is not contiguous "
                f"(expected start {expected_start})"
            )
        width = end - start
        expected = _field_param_width(field)
        if kind != expected[0] or width != expected[1]:
            raise SchemaError(
                f"Dict field {key!r} param_layout kind/width {kind!r}/{width} disagrees "
                f"with nested action case (expected {expected[0]!r}/{expected[1]})"
            )
        out.append((key, str(kind), start, end))
        expected_start = end
    extra = sorted(set(fields) - {k for k, _ in dict_field_actions(action)})
    if extra:
        raise SchemaError(f"Dict param_layout has unknown fields: {extra}")
    return out


def _field_param_width(field: dict[str, Any]) -> tuple[str, int]:
    ftype = action_type(field)
    if ftype == "Discrete":
        n = int(field.get("n", 0))
        if n <= 0:
            raise SchemaError("Discrete field requires positive n")
        return "logits", n
    if ftype == "MultiDiscrete":
        return "logits", multidiscrete_logit_width(field)
    if ftype == "Box":
        dim = int(np.prod(box_shape(field)))
        if box_distribution(field) == "diagonal_gaussian":
            return "mean_log_std", 2 * dim
        return "box_values", dim
    if ftype == "Dict":
        # Nested Dict uses its own concatenated layout width.
        return "nested_dict", sum(end - start for _, _, start, end in dict_param_slices(field))
    # Unknown nested field kinds fail via the action registry (extension recipe).
    _action_case(ftype)
    raise SchemaError(f"unsupported Dict field action.type={ftype!r}")



def validate_action_case(
    action: dict[str, Any],
    *,
    architecture: dict[str, Any] | None = None,
    adapter: str | None = None,
    require_byo_layout: bool = False,
) -> None:
    """Validate a complete action-type case via the action-case registry."""
    del adapter  # reserved for future adapter-specific case subsets
    atype = action_type(action)
    _action_case(atype).validate(
        action,
        architecture=architecture,
        require_byo_layout=require_byo_layout,
    )


def actions_equal(got: Any, expected: Any, *, action: dict[str, Any]) -> bool:
    """Compare actions under the declared case (exact for ints, atol for floats)."""
    return _action_case(action_type(action)).actions_equal(got, expected, action=action)


def validate_expected_action(
    case: dict[str, Any], *, action: dict[str, Any], index: int | None = None
) -> None:
    """Reject illegal expected_action values for any registered action case."""
    if "expected_action" not in case:
        return
    _action_case(action_type(action)).validate_expected(case, action=action, index=index)


def decode_action_from_params(
    params: np.ndarray,
    *,
    action: dict[str, Any],
    mode: str,
    rng: np.random.Generator | None = None,
    action_mask: np.ndarray | list | dict | None = None,
) -> Any:
    """Decode a flat actor parameter vector into a typed action under the case contract."""
    return _action_case(action_type(action)).decode(
        params, action=action, mode=mode, rng=rng, action_mask=action_mask
    )


def validate_runtime_action(
    value: Any,
    *,
    action: dict[str, Any],
    agent: str | None = None,
) -> None:
    """Validate a concrete action against the declared case (match/runtime gate)."""
    _action_case(action_type(action)).validate_runtime(value, action=action, agent=agent)


def _validate_multidiscrete_mask(mask: Any, *, nvec: list[int], prefix: str = "") -> None:
    arr = np.asarray(mask, dtype=bool)
    total = int(sum(nvec))
    if arr.shape == (total,):
        return
    if isinstance(mask, list) and len(mask) == len(nvec):
        for i, (m, n) in enumerate(zip(mask, nvec, strict=True)):
            flat = np.asarray(m, dtype=bool).reshape(-1)
            if flat.size != n:
                raise ConformanceError(
                    f"{prefix}action_mask factor[{i}] length {flat.size} != nvec[{i}]={n}"
                )
        return
    raise ConformanceError(
        f"{prefix}MultiDiscrete action_mask must be length-sum(nvec)={total} flat bools "
        f"or a per-factor list matching nvec={nvec}"
    )


def _flatten_factor_masks(mask: Any, *, nvec: list[int]) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    total = int(sum(nvec))
    if arr.shape == (total,):
        return arr.reshape(-1)
    parts = [np.asarray(m, dtype=bool).reshape(-1) for m in mask]
    return np.concatenate(parts, axis=0)
