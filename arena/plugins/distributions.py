"""Box distribution cases registered on the distribution axis."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from arena.core.errors import ConformanceError, SchemaError
from arena.core.registry import DISTRIBUTIONS

STOCHASTIC_BOX_TRANSFORMS = ("sample", "tanh", "affine")
RNG_ALGORITHMS = frozenset({"numpy_generator"})


class DistributionCase(Protocol):
    kind: str

    def validate(self, action: dict[str, Any]) -> None: ...

    def decode(
        self,
        params: np.ndarray,
        *,
        action: dict[str, Any],
        mode: str,
        rng: np.random.Generator | None = None,
    ) -> Any: ...


def register_distribution(
    kind: str, case: DistributionCase, *, replace: bool = False
) -> DistributionCase:
    return DISTRIBUTIONS.register(kind, case, replace=replace)


class DeterministicBoxDistribution:
    kind = "deterministic"

    def validate(self, action: dict[str, Any]) -> None:
        # Stochastic fields must not appear on the deterministic case.
        if action.get("param_layout") is not None or action.get("transform") is not None:
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
    ) -> Any:
        del rng
        from arena.core.action_cases import box_bounds, box_shape

        flat = np.asarray(params, dtype=np.float64).reshape(-1)
        shape = box_shape(action)
        dim = int(np.prod(shape))
        dtype = np.dtype(action.get("dtype", "float32"))
        low, high = box_bounds(action)
        if mode != "deterministic":
            raise SchemaError(
                "stochastic Box action requested, but this policy declares "
                "distribution=deterministic. Export a diagonal_gaussian case with "
                "param_layout/transform/rng to enable stochastic mode."
            )
        if flat.size != dim:
            raise ConformanceError(
                f"Box actor emitted {flat.size} values, expected action shape {shape}"
            )
        value = flat.reshape(shape).astype(dtype)
        if not np.all(np.isfinite(value)) or np.any(value < low) or np.any(value > high):
            raise ConformanceError(
                "Box actor emitted action outside declared bounds; refusing to clip"
            )
        return value


class DiagonalGaussianDistribution:
    kind = "diagonal_gaussian"

    def validate(self, action: dict[str, Any]) -> None:
        from arena.core.action_cases import box_bounds, box_shape

        shape = box_shape(action)
        dim = int(np.prod(shape))
        layout = action.get("param_layout")
        if not isinstance(layout, dict) or layout.get("kind") != "mean_log_std_concat":
            raise SchemaError(
                "stochastic Box case is incomplete: declare param_layout "
                "{kind: mean_log_std_concat} (actor emits [mean | log_std], length 2*prod(shape)). "
                "Do not treat stochastic Box as deterministic without declaring distribution."
            )
        expected = 2 * dim
        width = layout.get("width", expected)
        if int(width) != expected:
            raise SchemaError(
                f"diagonal_gaussian param_layout.width ({width}) must equal "
                f"2*prod(shape)={expected}"
            )
        transform = action.get("transform")
        if not isinstance(transform, dict):
            raise SchemaError(
                "stochastic Box case is incomplete: declare transform.order "
                f"(supported portable order: {list(STOCHASTIC_BOX_TRANSFORMS)})"
            )
        order = transform.get("order")
        if list(order or []) != list(STOCHASTIC_BOX_TRANSFORMS):
            raise SchemaError(
                f"unsupported Box transform.order={order!r}; portable diagonal_gaussian "
                f"requires exactly {list(STOCHASTIC_BOX_TRANSFORMS)}. Wrong order is a "
                "different case and will not be silently rewritten."
            )
        rng = action.get("rng")
        if not isinstance(rng, dict) or rng.get("algorithm") not in RNG_ALGORITHMS:
            raise SchemaError(
                "stochastic Box case is incomplete: declare rng.algorithm "
                f"(supported: {sorted(RNG_ALGORITHMS)}). Seeded source conformance uses "
                "numpy.random.Generator.standard_normal under the match/verify seed contract."
            )
        if action.get("deterministic_mode", "mean") != "mean":
            raise SchemaError(
                "stochastic Box deterministic_mode must be 'mean' (mean path through the "
                "same transform, skipping the noise sample)"
            )
        box_bounds(action)
        if action.get("dtype", "float32") not in {"float32", "float64"}:
            raise SchemaError("Box action dtype must be float32 or float64")

    def decode(
        self,
        params: np.ndarray,
        *,
        action: dict[str, Any],
        mode: str,
        rng: np.random.Generator | None = None,
    ) -> Any:
        from arena.core.action_cases import box_bounds, box_shape

        self.validate(action)
        flat = np.asarray(params, dtype=np.float64).reshape(-1)
        shape = box_shape(action)
        dim = int(np.prod(shape))
        dtype = np.dtype(action.get("dtype", "float32"))
        low, high = box_bounds(action)
        if flat.size != 2 * dim:
            raise ConformanceError(
                f"diagonal_gaussian actor emitted {flat.size} params, expected {2 * dim}"
            )
        mean = flat[:dim]
        log_std = flat[dim:]
        std = np.exp(log_std)
        if mode == "deterministic":
            pre_tanh = mean
        elif mode == "stochastic":
            if rng is None:
                raise ConformanceError(
                    "stochastic Box requires an explicit numpy.random.Generator under "
                    "the declared rng.algorithm contract"
                )
            noise = rng.standard_normal(size=dim)
            pre_tanh = mean + std * noise
        else:
            raise SchemaError(f"unknown inference mode: {mode}")
        squashed = np.tanh(pre_tanh)
        low_b = np.broadcast_to(low, shape).astype(np.float64).reshape(-1)
        high_b = np.broadcast_to(high, shape).astype(np.float64).reshape(-1)
        scaled = low_b + 0.5 * (high_b - low_b) * (squashed + 1.0)
        value = scaled.reshape(shape).astype(dtype)
        if not np.all(np.isfinite(value)) or np.any(value < low) or np.any(value > high):
            raise ConformanceError(
                "stochastic Box transform produced values outside declared bounds; "
                "refusing to clip"
            )
        return value


def register_builtins() -> None:
    register_distribution("deterministic", DeterministicBoxDistribution(), replace=True)
    register_distribution("diagonal_gaussian", DiagonalGaussianDistribution(), replace=True)
