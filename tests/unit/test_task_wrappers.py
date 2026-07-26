"""Unit coverage for declarative PettingZoo / SuperSuit task wrappers."""

from __future__ import annotations

import pytest

from arena.adapters.task_pettingzoo.wrappers import (
    SUPPORTED_WRAPPER_OPS,
    normalize_wrappers,
    wrappers_provenance,
)
from arena.core.errors import SchemaError
from arena.core.spaces import spaces_compatible


def test_normalize_wrappers_accepts_supersuit_aliases() -> None:
    steps = normalize_wrappers(
        [
            {"op": "color_reduction_v0", "mode": "full"},
            {"op": "resize_v1", "x_size": 64, "y_size": 64},
            {"op": "frame_stack_v1", "stack_size": 4},
        ]
    )
    assert [s["op"] for s in steps] == ["color_reduction", "resize", "frame_stack"]
    assert steps[1]["x_size"] == 64
    assert steps[2]["stack_size"] == 4
    assert SUPPORTED_WRAPPER_OPS == {"color_reduction", "resize", "frame_stack"}


def test_unknown_wrapper_op_fails_loud() -> None:
    with pytest.raises(SchemaError, match="unknown task wrapper op"):
        normalize_wrappers([{"op": "normalize_observation"}])


def test_resize_requires_sizes() -> None:
    with pytest.raises(SchemaError, match="resize requires"):
        normalize_wrappers([{"op": "resize"}])


def test_wrappers_provenance_identity() -> None:
    prov = wrappers_provenance(
        [
            {"op": "color_reduction", "mode": "full"},
            {"op": "resize", "width": 64, "height": 64},
            {"op": "frame_stack", "k": 4},
        ]
    )
    assert prov["identity"] == "color_reduction(full)>resize(64,64)>frame_stack(4)"
    assert len(prov["chain"]) == 3


def test_layout_mismatch_is_incompatible() -> None:
    expected = {"type": "Box", "shape": [64, 64, 4], "layout": "HWC", "dtype": "uint8"}
    actual = {"type": "Box", "shape": [64, 64, 4], "layout": "CHW", "dtype": "uint8"}
    mism = spaces_compatible(expected, actual)
    assert any("layout" in m for m in mism)


def test_shape_mismatch_without_wrappers_is_incompatible() -> None:
    expected = {"type": "Box", "shape": [457, 120, 3], "dtype": "uint8"}
    actual = {"type": "Box", "shape": [64, 64, 4], "layout": "HWC", "dtype": "uint8"}
    mism = spaces_compatible(expected, actual)
    assert any("shape" in m for m in mism)
