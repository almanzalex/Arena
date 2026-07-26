"""Preprocess-op cases registered on the preprocess axis."""

from __future__ import annotations

from collections import deque
from typing import Any, Protocol

import numpy as np

from arena.core.errors import ConformanceError, SchemaError
from arena.core.registry import PREPROCESS_OPS


class PreprocessOp(Protocol):
    kind: str

    def apply(
        self,
        x: np.ndarray,
        step: dict[str, Any],
        *,
        agent_id: str,
        state: dict[str, Any],
    ) -> np.ndarray: ...


def register_preprocess_op(kind: str, op: PreprocessOp, *, replace: bool = False) -> PreprocessOp:
    return PREPROCESS_OPS.register(kind, op, replace=replace)


class LayoutOp:
    kind = "layout"

    def apply(
        self,
        x: np.ndarray,
        step: dict[str, Any],
        *,
        agent_id: str,
        state: dict[str, Any],
    ) -> np.ndarray:
        del agent_id, state
        source, target = step.get("from"), step.get("to")
        if source not in {"CHW", "HWC"} or target not in {"CHW", "HWC"}:
            raise SchemaError("layout op requires from/to CHW or HWC")
        if x.ndim != 3:
            raise ConformanceError(f"layout conversion requires rank-3 image, got {x.shape}")
        return x if source == target else np.transpose(x, (2, 0, 1))


class RunningNormOp:
    kind = "running_norm"

    def apply(
        self,
        x: np.ndarray,
        step: dict[str, Any],
        *,
        agent_id: str,
        state: dict[str, Any],
    ) -> np.ndarray:
        del agent_id, state
        mean, std = np.asarray(step["mean"], np.float32), np.asarray(step["std"], np.float32)
        if np.any(std <= 0):
            raise SchemaError("running_norm std must be positive")
        try:
            return (x - mean) / np.maximum(std, float(step.get("eps", 1e-8)))
        except ValueError as e:
            raise ConformanceError(f"running_norm shape mismatch for {x.shape}") from e


class ClipOp:
    kind = "clip"

    def apply(
        self,
        x: np.ndarray,
        step: dict[str, Any],
        *,
        agent_id: str,
        state: dict[str, Any],
    ) -> np.ndarray:
        del agent_id, state
        return np.clip(x, float(step["low"]), float(step["high"]))


class FrameStackOp:
    kind = "frame_stack"

    def apply(
        self,
        x: np.ndarray,
        step: dict[str, Any],
        *,
        agent_id: str,
        state: dict[str, Any],
    ) -> np.ndarray:
        k = int(step.get("k", 0))
        axis = int(step.get("axis", 0))
        if k < 1 or not (-x.ndim <= axis < x.ndim):
            raise SchemaError("frame_stack requires k >= 1 and an in-range axis")
        frames_map: dict[str, deque[np.ndarray]] = state.setdefault("frames", {})
        frames = frames_map.setdefault(agent_id, deque(maxlen=k))
        if not frames:
            pad = step.get("pad", "zeros")
            if pad == "zeros":
                frames.extend(np.zeros_like(x) for _ in range(k - 1))
            elif pad == "repeat_first":
                frames.extend(x.copy() for _ in range(k - 1))
            else:
                raise SchemaError("frame_stack.pad must be zeros or repeat_first")
        if frames and frames[-1].shape != x.shape:
            raise ConformanceError(
                f"frame_stack input shape changed from {frames[-1].shape} to {x.shape}"
            )
        frames.append(x.copy())
        return np.concatenate(tuple(frames), axis=axis)


class FlattenOp:
    kind = "flatten"

    def apply(
        self,
        x: np.ndarray,
        step: dict[str, Any],
        *,
        agent_id: str,
        state: dict[str, Any],
    ) -> np.ndarray:
        del step, agent_id, state
        return x.reshape(-1)


class ConcatOp:
    kind = "concat"

    def apply(
        self,
        x: np.ndarray,
        step: dict[str, Any],
        *,
        agent_id: str,
        state: dict[str, Any],
    ) -> np.ndarray:
        del x, step, agent_id, state
        raise ConformanceError(
            "concat preprocessing requires an adapter-supplied structured input; "
            "this portable runtime accepts one raw observation tensor"
        )


def register_builtins() -> None:
    for op in (
        LayoutOp(),
        RunningNormOp(),
        ClipOp(),
        FrameStackOp(),
        FlattenOp(),
        ConcatOp(),
    ):
        register_preprocess_op(op.kind, op, replace=True)
