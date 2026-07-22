"""Serializable, shape-safe preprocessing pipeline for portable actors."""

from __future__ import annotations

from typing import Any

import numpy as np

from rlx.core.errors import ConformanceError, SchemaError
from rlx.core.registry import PREPROCESS_OPS, ensure_plugins_loaded


class PreprocessPipeline:
    """Execute ``rlx.preprocess/v1`` without implicit reshapes or coercions."""

    def __init__(self, spec: dict[str, Any], observation: dict[str, Any]) -> None:
        ensure_plugins_loaded()
        self.spec = spec
        self.observation = observation
        self.steps = list(spec.get("pipeline", {}).get("steps", []))
        version = spec.get("pipeline", {}).get("version", "rlx.preprocess/v1")
        if version != "rlx.preprocess/v1":
            raise SchemaError(f"unsupported preprocessing pipeline version {version!r}")
        self._state: dict[str, Any] = {"frames": {}}
        # Ops are resolved on apply so construction stays cheap; unknown kinds
        # still fail loud with an extension recipe before producing tensors.

    def reset(self, agent_id: str = "default") -> None:
        frames = self._state.get("frames")
        if isinstance(frames, dict):
            frames.pop(agent_id, None)

    def __call__(self, observation: Any, agent_id: str = "default") -> np.ndarray:
        ensure_plugins_loaded()
        x = np.asarray(observation, dtype=np.float32)
        declared = tuple(int(v) for v in self.observation.get("shape", []))
        if declared and tuple(x.shape) != declared:
            raise ConformanceError(
                f"raw observation shape {tuple(x.shape)} != declared shape {declared}; "
                "refusing implicit reshape"
            )
        for step in self.steps:
            op = PREPROCESS_OPS.get(str(step.get("op")))
            x = op.apply(x, step, agent_id=agent_id, state=self._state)
        return np.asarray(x, dtype=np.float32)


def canonical_pipeline(
    preprocessing: dict[str, Any] | None, observation: dict[str, Any]
) -> dict[str, Any]:
    """Upgrade legacy elementwise preprocessing into explicit v1 pipeline."""
    del observation  # retained for call-site compatibility / future layout defaults
    preprocessing = dict(preprocessing or {})
    if "pipeline" in preprocessing:
        out = dict(preprocessing)
        out.setdefault("included", True)
        out.setdefault("id", out.get("id", "pipeline_v1"))
        return out
    steps: list[dict[str, Any]] = []
    if preprocessing.get("mean") is not None or preprocessing.get("std") is not None:
        steps.append(
            {
                "op": "running_norm",
                "mean": preprocessing.get("mean", 0.0),
                "std": preprocessing.get("std", 1.0),
                "eps": 1e-8,
            }
        )
    if preprocessing.get("clip") is not None:
        low, high = preprocessing["clip"]
        steps.append({"op": "clip", "low": low, "high": high})
    # Legacy templates expect a vector. A declared image is never silently flattened:
    # an author must include the flatten operation in their new pipeline.
    return {
        "included": True,
        "id": preprocessing.get("id", "normalize_v0"),
        "pipeline": {"version": "rlx.preprocess/v1", "steps": steps},
    }
