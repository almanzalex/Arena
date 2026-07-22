"""Regression gates for RFC 002 portability residuals."""

from __future__ import annotations

import numpy as np
import pytest

from rlx.adapters.policy_custom_torch.preprocess import PreprocessPipeline
from rlx.core.errors import ConformanceError, SchemaError
from rlx.runtime.match import _validate_action


def test_hwc_to_chw_norm_and_frame_stack_are_explicit_and_stateful() -> None:
    pipe = PreprocessPipeline(
        {
            "pipeline": {
                "version": "rlx.preprocess/v1",
                "steps": [
                    {"op": "layout", "from": "HWC", "to": "CHW"},
                    {"op": "running_norm", "mean": 1.0, "std": 2.0},
                    {"op": "frame_stack", "k": 2, "axis": 0, "pad": "zeros"},
                ],
            }
        },
        {"type": "Box", "shape": [2, 3, 1], "layout": "HWC"},
    )
    first = pipe(np.ones((2, 3, 1), dtype=np.float32), "a")
    assert first.shape == (2, 2, 3)
    assert np.all(first[0] == 0.0) and np.all(first[1] == 0.0)
    second = pipe(np.full((2, 3, 1), 3, dtype=np.float32), "a")
    assert np.all(second[0] == 0.0) and np.all(second[1] == 1.0)
    pipe.reset("a")
    assert np.all(pipe(np.ones((2, 3, 1), dtype=np.float32), "a")[0] == 0.0)


def test_image_pipeline_refuses_shape_or_layout_lies() -> None:
    pipe = PreprocessPipeline(
        {"pipeline": {"version": "rlx.preprocess/v1", "steps": [{"op": "layout", "from": "HWC", "to": "CHW"}]}},
        {"type": "Box", "shape": [2, 3, 1], "layout": "HWC"},
    )
    with pytest.raises(ConformanceError, match="raw observation shape"):
        pipe(np.ones((1, 2, 3), dtype=np.float32))
    with pytest.raises(ConformanceError, match="rank-3"):
        PreprocessPipeline(
            {"pipeline": {"version": "rlx.preprocess/v1", "steps": [{"op": "layout", "from": "HWC", "to": "CHW"}]}},
            {"type": "Box", "shape": [6], "layout": None},
        )(np.ones(6, dtype=np.float32))


def test_box_actions_are_checked_without_clipping() -> None:
    task = {"roles": {"agent": {"action": {"type": "Box", "shape": [2], "low": [-1, -1], "high": [1, 1]}}}}
    _validate_action(
        np.array([0.1, -0.4], dtype=np.float32),
        agent="agent",
        task_info=task,
        episode_index=0,
        step_i=0,
    )
    with pytest.raises(Exception, match="outside"):
        _validate_action(
            np.array([2.0, 0.0], dtype=np.float32),
            agent="agent",
            task_info=task,
            episode_index=0,
            step_i=0,
        )


def test_unknown_preprocess_op_fails_loudly() -> None:
    pipe = PreprocessPipeline(
        {"pipeline": {"version": "rlx.preprocess/v1", "steps": [{"op": "invented"}]}},
        {"type": "Box", "shape": [2]},
    )
    with pytest.raises(SchemaError, match="Unknown preprocess|unknown preprocessing|extension"):
        pipe(np.zeros(2, dtype=np.float32))
