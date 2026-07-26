"""Task wrapper-op cases registered on the wrapper axis."""

from __future__ import annotations

from typing import Any, Protocol

from arena.core.errors import ArenaError, SchemaError
from arena.core.registry import WRAPPER_OPS

_OP_ALIASES = {
    "color_reduction": "color_reduction",
    "color_reduction_v0": "color_reduction",
    "resize": "resize",
    "resize_v1": "resize",
    "frame_stack": "frame_stack",
    "frame_stack_v1": "frame_stack",
}


class WrapperOp(Protocol):
    kind: str
    aliases: tuple[str, ...]

    def normalize(self, step: dict[str, Any], *, index: int) -> dict[str, Any]: ...

    def apply(self, env: Any, step: dict[str, Any], *, index: int, supersuit: Any) -> Any: ...

    def identity_part(self, step: dict[str, Any]) -> str: ...


def register_wrapper_op(kind: str, op: WrapperOp, *, replace: bool = False) -> WrapperOp:
    return WRAPPER_OPS.register(kind, op, replace=replace)


def resolve_wrapper_kind(raw_op: str) -> str:
    """Map alias → canonical kind, then resolve via registry (fail-loud)."""
    from arena.core.registry import ensure_plugins_loaded

    ensure_plugins_loaded()
    canonical = _OP_ALIASES.get(str(raw_op), str(raw_op))
    return WRAPPER_OPS.get(canonical).kind


class ColorReductionOp:
    kind = "color_reduction"
    aliases = ("color_reduction", "color_reduction_v0")

    def normalize(self, step: dict[str, Any], *, index: int) -> dict[str, Any]:
        mode = step.get("mode", "full")
        if mode not in {"full", "R", "G", "B"}:
            raise SchemaError(
                f"wrappers[{index}] color_reduction.mode must be full|R|G|B, got {mode!r}"
            )
        out = dict(step)
        out["mode"] = mode
        return out

    def apply(self, env: Any, step: dict[str, Any], *, index: int, supersuit: Any) -> Any:
        del index
        return supersuit.color_reduction_v0(env, mode=step.get("mode", "full"))

    def identity_part(self, step: dict[str, Any]) -> str:
        return f"color_reduction({step.get('mode', 'full')})"


class ResizeOp:
    kind = "resize"
    aliases = ("resize", "resize_v1")

    def normalize(self, step: dict[str, Any], *, index: int) -> dict[str, Any]:
        x = step.get("x_size", step.get("width", step.get("x")))
        y = step.get("y_size", step.get("height", step.get("y")))
        if x is None or y is None:
            raise SchemaError(
                f"wrappers[{index}] resize requires x_size and y_size (or width/height)"
            )
        out = dict(step)
        try:
            out["x_size"] = int(x)
            out["y_size"] = int(y)
        except (TypeError, ValueError) as e:
            raise SchemaError(f"wrappers[{index}] resize sizes must be integers") from e
        if out["x_size"] < 1 or out["y_size"] < 1:
            raise SchemaError(f"wrappers[{index}] resize sizes must be >= 1")
        return out

    def apply(self, env: Any, step: dict[str, Any], *, index: int, supersuit: Any) -> Any:
        del index
        return supersuit.resize_v1(env, step["x_size"], step["y_size"])

    def identity_part(self, step: dict[str, Any]) -> str:
        return f"resize({step['x_size']},{step['y_size']})"


class FrameStackWrapperOp:
    kind = "frame_stack"
    aliases = ("frame_stack", "frame_stack_v1")

    def normalize(self, step: dict[str, Any], *, index: int) -> dict[str, Any]:
        k = step.get("stack_size", step.get("k", step.get("num_frames")))
        if k is None:
            raise SchemaError(f"wrappers[{index}] frame_stack requires stack_size (or k)")
        out = dict(step)
        try:
            out["stack_size"] = int(k)
        except (TypeError, ValueError) as e:
            raise SchemaError(f"wrappers[{index}] frame_stack.stack_size must be an integer") from e
        if out["stack_size"] < 1:
            raise SchemaError(f"wrappers[{index}] frame_stack.stack_size must be >= 1")
        return out

    def apply(self, env: Any, step: dict[str, Any], *, index: int, supersuit: Any) -> Any:
        del index
        return supersuit.frame_stack_v1(env, stack_size=step["stack_size"])

    def identity_part(self, step: dict[str, Any]) -> str:
        return f"frame_stack({step['stack_size']})"


def register_builtins() -> None:
    for op in (ColorReductionOp(), ResizeOp(), FrameStackWrapperOp()):
        register_wrapper_op(op.kind, op, replace=True)
        for alias in op.aliases:
            _OP_ALIASES[alias] = op.kind


def require_supersuit():
    try:
        import supersuit
    except ImportError as e:
        raise ArenaError(
            "task.wrappers requires SuperSuit to apply the declared PettingZoo "
            "observation chain. Install with: pip install supersuit  "
            "(or pip install 'arena[wrappers]'). Refusing to run unwrapped."
        ) from e
    return supersuit
