"""Concrete action-case implementations (registered via ``arena.plugins.actions``)."""

from __future__ import annotations

from typing import Any

import numpy as np

from arena.core import action_cases as ac
from arena.core.errors import ConformanceError, SchemaError
from arena.core.registry import ACTION_CASES, DISTRIBUTIONS, ensure_plugins_loaded


class DiscreteActionCase:
    kind = "Discrete"

    def validate(
        self,
        action: dict[str, Any],
        *,
        architecture: dict[str, Any] | None = None,
        adapter: str | None = None,
        require_byo_layout: bool = False,
    ) -> None:
        del adapter, require_byo_layout
        ac.require_masks(action)
        n = action.get("n")
        if n is None or int(n) <= 0:
            raise SchemaError("Discrete action requires a positive n")
        if architecture is not None and architecture.get("type") in {
            "mlp_categorical",
            "gru_categorical",
        }:
            action_n = architecture.get("action_n")
            if action_n is None or int(action_n) != int(n):
                raise SchemaError(
                    f"architecture.action_n ({action_n}) disagrees with action.n ({int(n)}). "
                    "Align both fields to the true Discrete action count before export/verify."
                )

    def decode(
        self,
        params: np.ndarray,
        *,
        action: dict[str, Any],
        mode: str,
        rng: np.random.Generator | None = None,
        action_mask: Any = None,
    ) -> Any:
        flat = np.asarray(params, dtype=np.float64).reshape(-1)
        n = int(action["n"])
        if flat.size != n:
            raise ConformanceError(f"Discrete actor emitted {flat.size} logits, expected {n}")
        logits = flat.copy()
        if action_mask is not None:
            mask_arr = np.asarray(action_mask, dtype=bool).reshape(-1)
            if mask_arr.size != n:
                raise ConformanceError("action mask length mismatch")
            if not bool(mask_arr.any()):
                raise ConformanceError("action mask has no legal actions")
            logits = np.where(mask_arr, logits, -1e9)
        if mode == "deterministic":
            action_i = int(np.argmax(logits))
        elif mode == "stochastic":
            if rng is None:
                rng = np.random.default_rng()
            z = logits - np.max(logits)
            probs = np.exp(z)
            probs = probs / probs.sum()
            action_i = int(rng.choice(n, p=probs))
        else:
            raise SchemaError(f"unknown inference mode: {mode}")
        if action_mask is not None and not bool(np.asarray(action_mask)[action_i]):
            raise ConformanceError("illegal action selected despite mask")
        return action_i

    def actions_equal(self, got: Any, expected: Any, *, action: dict[str, Any]) -> bool:
        del action
        try:
            return int(got) == int(expected)
        except (TypeError, ValueError):
            return False

    def validate_expected(
        self, case: dict[str, Any], *, action: dict[str, Any], index: int | None = None
    ) -> None:
        if "expected_action" not in case:
            return
        prefix = f"reference case {index}: " if index is not None else ""
        raw = case["expected_action"]
        n = int(action["n"])
        if isinstance(raw, bool) or not isinstance(raw, (int, np.integer)):
            raise ConformanceError(f"{prefix}expected_action {raw!r} must be an integer")
        expected = int(raw)
        if not (0 <= expected < n):
            raise ConformanceError(
                f"{prefix}expected_action {expected} is illegal under Discrete(n={n}) "
                f"(legal range [0, {n})). Refuse to stamp verified."
            )
        mask = case.get("action_mask")
        if mask is not None:
            mask_arr = np.asarray(mask, dtype=bool).reshape(-1)
            if mask_arr.size != n:
                raise ConformanceError(
                    f"{prefix}action_mask length {mask_arr.size} != action.n {n}"
                )
            if not bool(mask_arr[expected]):
                raise ConformanceError(
                    f"{prefix}expected_action {expected} is illegal under the case "
                    "action_mask. Refuse to stamp verified."
                )

    def validate_runtime(
        self, value: Any, *, action: dict[str, Any], agent: str | None = None
    ) -> None:
        prefix = f"policy for {agent}: " if agent else ""
        n = int(action["n"])
        try:
            idx = int(value)
        except (TypeError, ValueError) as e:
            raise ConformanceError(f"{prefix}non-integer Discrete action {value!r}") from e
        if isinstance(value, bool) or (isinstance(value, float) and float(value) != int(value)):
            raise ConformanceError(f"{prefix}non-integer Discrete action {value!r}")
        if not (0 <= idx < n):
            raise ConformanceError(
                f"{prefix}out-of-bounds Discrete action {idx} (legal range [0, {n}))"
            )


class MultiDiscreteActionCase:
    kind = "MultiDiscrete"

    def validate(
        self,
        action: dict[str, Any],
        *,
        architecture: dict[str, Any] | None = None,
        adapter: str | None = None,
        require_byo_layout: bool = False,
    ) -> None:
        del adapter
        ac.require_masks(action)
        nvec = ac.multidiscrete_nvec(action)
        slices = ac.multidiscrete_slices(action)
        ac.multidiscrete_sampling_order(action)
        if architecture is not None and architecture.get("type") in {
            "mlp_categorical",
            "gru_categorical",
        }:
            raise SchemaError(
                "MultiDiscrete actions are deliberately rejected for template actors "
                "(mlp_categorical/gru_categorical): templates cannot preserve factor "
                "boundaries. Use Discrete templates, or export a BYO TorchScript actor "
                "with a complete MultiDiscrete case (nvec + logit_layout + sampling_order)."
            )
        if require_byo_layout and action.get("logit_layout") is None:
            raise SchemaError(
                "MultiDiscrete BYO case requires logit_layout "
                "(kind: concatenated). Refusing incomplete claim."
            )
        width = slices[-1][1]
        io = (architecture or {}).get("io") if architecture else None
        if isinstance(io, dict) and "action_logit_width" in io:
            if int(io["action_logit_width"]) != width:
                raise SchemaError(
                    f"architecture.io.action_logit_width ({io['action_logit_width']}) "
                    f"disagrees with MultiDiscrete logit width {width} from nvec={nvec}"
                )

    def decode(
        self,
        params: np.ndarray,
        *,
        action: dict[str, Any],
        mode: str,
        rng: np.random.Generator | None = None,
        action_mask: Any = None,
    ) -> Any:
        flat = np.asarray(params, dtype=np.float64).reshape(-1)
        nvec = ac.multidiscrete_nvec(action)
        slices = ac.multidiscrete_slices(action)
        order = ac.multidiscrete_sampling_order(action)
        width = ac.multidiscrete_logit_width(action)
        if flat.size != width:
            raise ConformanceError(
                f"MultiDiscrete actor emitted {flat.size} logits, expected {width}"
            )
        if action_mask is None:
            factor_masks: list[np.ndarray | None] = [None] * len(nvec)
        else:
            ac._validate_multidiscrete_mask(action_mask, nvec=nvec)
            flat_mask = ac._flatten_factor_masks(action_mask, nvec=nvec)
            factor_masks = []
            cursor = 0
            for n in nvec:
                factor_masks.append(flat_mask[cursor : cursor + n])
                cursor += n
        chosen = [0] * len(nvec)
        disc = ACTION_CASES.get("Discrete")
        for factor_i in order:
            start, end = slices[factor_i]
            sub = {
                "type": "Discrete",
                "n": nvec[factor_i],
                "masks": action.get("masks", "none"),
            }
            chosen[factor_i] = disc.decode(
                flat[start:end],
                action=sub,
                mode=mode,
                rng=rng,
                action_mask=factor_masks[factor_i],
            )
        return np.asarray(chosen, dtype=np.int64)

    def actions_equal(self, got: Any, expected: Any, *, action: dict[str, Any]) -> bool:
        g = np.asarray(got, dtype=np.int64).reshape(-1)
        e = np.asarray(expected, dtype=np.int64).reshape(-1)
        nvec = ac.multidiscrete_nvec(action)
        return g.shape == (len(nvec),) and e.shape == (len(nvec),) and np.array_equal(g, e)

    def validate_expected(
        self, case: dict[str, Any], *, action: dict[str, Any], index: int | None = None
    ) -> None:
        if "expected_action" not in case:
            return
        prefix = f"reference case {index}: " if index is not None else ""
        raw = case["expected_action"]
        nvec = ac.multidiscrete_nvec(action)
        value = np.asarray(raw)
        if value.shape != (len(nvec),) or not np.issubdtype(value.dtype, np.integer):
            raise ConformanceError(
                f"{prefix}MultiDiscrete expected_action must be an int vector of length "
                f"{len(nvec)}; got shape={value.shape} dtype={value.dtype}"
            )
        for i, (v, n) in enumerate(zip(value.tolist(), nvec, strict=True)):
            if not (0 <= int(v) < int(n)):
                raise ConformanceError(
                    f"{prefix}expected_action factor[{i}]={v} is illegal under "
                    f"MultiDiscrete nvec[{i}]={n} (legal range [0, {n})). "
                    "Refuse to stamp verified."
                )
        mask = case.get("action_mask")
        if mask is not None:
            ac._validate_multidiscrete_mask(mask, nvec=nvec, prefix=prefix)
            flat_mask = ac._flatten_factor_masks(mask, nvec=nvec)
            flat_action = value.astype(np.int64)
            offsets = np.cumsum([0] + nvec[:-1])
            for i, (a, off) in enumerate(zip(flat_action.tolist(), offsets.tolist(), strict=True)):
                if not bool(flat_mask[off + int(a)]):
                    raise ConformanceError(
                        f"{prefix}expected_action factor[{i}]={a} is illegal under "
                        "the case action_mask. Refuse to stamp verified."
                    )

    def validate_runtime(
        self, value: Any, *, action: dict[str, Any], agent: str | None = None
    ) -> None:
        prefix = f"policy for {agent}: " if agent else ""
        nvec = ac.multidiscrete_nvec(action)
        arr = np.asarray(value)
        if arr.shape != (len(nvec),) or not np.issubdtype(arr.dtype, np.integer):
            raise ConformanceError(
                f"{prefix}MultiDiscrete action must be int vector length {len(nvec)}; "
                f"got shape={arr.shape} dtype={arr.dtype}. Refusing scalar/flatten coercion."
            )
        for i, (v, n) in enumerate(zip(arr.tolist(), nvec, strict=True)):
            if not (0 <= int(v) < int(n)):
                raise ConformanceError(
                    f"{prefix}MultiDiscrete factor[{i}]={v} out of range [0, {n})"
                )


class BoxActionCase:
    kind = "Box"

    def validate(
        self,
        action: dict[str, Any],
        *,
        architecture: dict[str, Any] | None = None,
        adapter: str | None = None,
        require_byo_layout: bool = False,
    ) -> None:
        del adapter, require_byo_layout
        ensure_plugins_loaded()
        ac.require_masks(action)
        shape = ac.box_shape(action)
        if action.get("dtype", "float32") not in {"float32", "float64"}:
            raise SchemaError("Box action dtype must be float32 or float64")
        ac.box_bounds(action)
        dist_name = str(action.get("distribution", "deterministic"))
        dist = DISTRIBUTIONS.get(dist_name)
        if architecture is not None and architecture.get("type") in {
            "mlp_categorical",
            "gru_categorical",
        }:
            raise SchemaError(
                "custom-pytorch categorical templates only supports Discrete actions; "
                f"got Box (distribution={dist_name!r}). Use a BYO TorchScript actor for Box."
            )
        dist.validate(action)
        if dist_name == "diagonal_gaussian":
            io = (architecture or {}).get("io") if architecture else None
            expected = 2 * int(np.prod(shape))
            if isinstance(io, dict) and "action_param_width" in io:
                if int(io["action_param_width"]) != expected:
                    raise SchemaError(
                        f"architecture.io.action_param_width ({io['action_param_width']}) "
                        f"disagrees with diagonal_gaussian width {expected}"
                    )
        elif action.get("param_layout") is not None or action.get("transform") is not None:
            raise SchemaError(
                "deterministic Box must not declare stochastic param_layout/transform; "
                "set distribution: diagonal_gaussian for the stochastic case"
            )

    def decode(
        self,
        params: np.ndarray,
        *,
        action: dict[str, Any],
        mode: str,
        rng: np.random.Generator | None = None,
        action_mask: Any = None,
    ) -> Any:
        del action_mask
        ensure_plugins_loaded()
        dist_name = str(action.get("distribution", "deterministic"))
        return DISTRIBUTIONS.get(dist_name).decode(params, action=action, mode=mode, rng=rng)

    def actions_equal(self, got: Any, expected: Any, *, action: dict[str, Any]) -> bool:
        del action
        g = np.asarray(got, dtype=np.float64)
        e = np.asarray(expected, dtype=np.float64)
        return g.shape == e.shape and bool(np.allclose(g, e, atol=1e-5, rtol=1e-5))

    def validate_expected(
        self, case: dict[str, Any], *, action: dict[str, Any], index: int | None = None
    ) -> None:
        if "expected_action" not in case:
            return
        prefix = f"reference case {index}: " if index is not None else ""
        raw = case["expected_action"]
        shape = ac.box_shape(action)
        value = np.asarray(raw)
        if value.shape != shape:
            raise ConformanceError(
                f"{prefix}expected_action shape {value.shape} != Box action shape {shape}"
            )
        if not np.issubdtype(value.dtype, np.floating):
            raise ConformanceError(f"{prefix}Box expected_action must be floating-point")
        low, high = ac.box_bounds(action)
        if not np.all(np.isfinite(value)) or np.any(value < low) or np.any(value > high):
            raise ConformanceError(f"{prefix}expected_action violates Box bounds")

    def validate_runtime(
        self, value: Any, *, action: dict[str, Any], agent: str | None = None
    ) -> None:
        prefix = f"policy for {agent}: " if agent else ""
        shape = ac.box_shape(action)
        arr = np.asarray(value)
        if arr.shape != shape or not np.issubdtype(arr.dtype, np.floating):
            raise ConformanceError(
                f"{prefix}Box action shape/dtype {arr.shape}/{arr.dtype}, expected "
                f"{shape}/floating"
            )
        low, high = ac.box_bounds(action)
        if not np.all(np.isfinite(arr)) or np.any(arr < low) or np.any(arr > high):
            raise ConformanceError(f"{prefix}Box action outside declared bounds")


class DictActionCase:
    kind = "Dict"

    def validate(
        self,
        action: dict[str, Any],
        *,
        architecture: dict[str, Any] | None = None,
        adapter: str | None = None,
        require_byo_layout: bool = False,
    ) -> None:
        del adapter
        ensure_plugins_loaded()
        ac.require_masks(action)
        if architecture is not None and architecture.get("type") in {
            "mlp_categorical",
            "gru_categorical",
        }:
            raise SchemaError(
                "Dict actions are deliberately rejected for template actors. "
                "Use a BYO TorchScript actor with a complete Dict case "
                "(canonical key_order + typed spaces + param_layout)."
            )
        for key, field in ac.dict_field_actions(action):
            if "masks" not in field:
                field = {**field, "masks": "none"}
            try:
                ACTION_CASES.get(ac.action_type(field)).validate(
                    field,
                    architecture=None,
                    require_byo_layout=True,
                )
            except SchemaError as e:
                raise SchemaError(f"Dict field {key!r}: {e}") from e
        if require_byo_layout or (architecture or {}).get("type") in {
            "serialized_module",
            "trusted_source_module",
        }:
            ac.dict_param_slices(action)

    def decode(
        self,
        params: np.ndarray,
        *,
        action: dict[str, Any],
        mode: str,
        rng: np.random.Generator | None = None,
        action_mask: Any = None,
    ) -> Any:
        ensure_plugins_loaded()
        flat = np.asarray(params, dtype=np.float64).reshape(-1)
        slices = ac.dict_param_slices(action)
        width = slices[-1][3] if slices else 0
        if flat.size != width:
            raise ConformanceError(
                f"Dict actor emitted {flat.size} params, expected concatenated width {width}"
            )
        out: dict[str, Any] = {}
        mask_map = action_mask if isinstance(action_mask, dict) else None
        for key, _kind, start, end in slices:
            field = action["spaces"][key]
            sub_mask = None if mask_map is None else mask_map.get(key)
            out[key] = ACTION_CASES.get(ac.action_type(field)).decode(
                flat[start:end],
                action=field,
                mode=mode,
                rng=rng,
                action_mask=sub_mask,
            )
        return out

    def actions_equal(self, got: Any, expected: Any, *, action: dict[str, Any]) -> bool:
        ensure_plugins_loaded()
        if not isinstance(got, dict) or not isinstance(expected, dict):
            return False
        for key, field in ac.dict_field_actions(action):
            if key not in got or key not in expected:
                return False
            if not ACTION_CASES.get(ac.action_type(field)).actions_equal(
                got[key], expected[key], action=field
            ):
                return False
        return True

    def validate_expected(
        self, case: dict[str, Any], *, action: dict[str, Any], index: int | None = None
    ) -> None:
        if "expected_action" not in case:
            return
        ensure_plugins_loaded()
        prefix = f"reference case {index}: " if index is not None else ""
        raw = case["expected_action"]
        if not isinstance(raw, dict):
            raise ConformanceError(f"{prefix}Dict expected_action must be a mapping")
        keys = ac.dict_key_order(action)
        missing = [k for k in keys if k not in raw]
        extra = [k for k in raw if k not in keys]
        if missing or extra:
            raise ConformanceError(
                f"{prefix}Dict expected_action key mismatch "
                f"(missing={missing}, extra={extra}). Refuse to stamp verified."
            )
        for key, field in ac.dict_field_actions(action):
            nested = {"expected_action": raw[key]}
            if isinstance(case.get("action_mask"), dict) and key in case["action_mask"]:
                nested["action_mask"] = case["action_mask"][key]
            ACTION_CASES.get(ac.action_type(field)).validate_expected(
                nested, action=field, index=index
            )

    def validate_runtime(
        self, value: Any, *, action: dict[str, Any], agent: str | None = None
    ) -> None:
        ensure_plugins_loaded()
        prefix = f"policy for {agent}: " if agent else ""
        if not isinstance(value, dict):
            raise ConformanceError(
                f"{prefix}Dict action must be a mapping; got {type(value).__name__}. "
                "Refusing Dict↔vector coercion."
            )
        keys = ac.dict_key_order(action)
        missing = [k for k in keys if k not in value]
        extra = [k for k in value if k not in keys]
        if missing or extra:
            raise ConformanceError(
                f"{prefix}Dict action key mismatch (missing={missing}, extra={extra})"
            )
        for key, field in ac.dict_field_actions(action):
            ACTION_CASES.get(ac.action_type(field)).validate_runtime(
                value[key], action=field, agent=agent
            )
