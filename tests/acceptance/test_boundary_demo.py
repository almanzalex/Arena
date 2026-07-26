from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("pettingzoo")
pytest.importorskip("pyspiel")


def _demo_module():
    path = Path("examples/boundaries/run_demo.py").resolve()
    spec = importlib.util.spec_from_file_location("arena_boundary_demo", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.acceptance
def test_former_boundaries_execute_as_user_flows(tmp_path: Path) -> None:
    summary = _demo_module().run_demo(tmp_path / "demo")
    assert summary["dynamic"]["outcome"]["failure_count"] == 0
    assert summary["dynamic"]["resolver"] == "role"
    assert summary["dynamic"]["agent_0_segments"] == 2
    assert summary["dynamic"]["lifecycle_events"] >= 6
    assert summary["training"]["selected_episodes"] == 12
    assert sum(summary["training"]["split_counts"].values()) == 12
    assert set(summary["training"]["split_counts"]) == {"train", "validation"}
    assert summary["training"]["resumed_from_epoch"] == 25
    assert summary["training"]["loss_final"] < summary["training"]["loss_initial"]
    assert summary["training"]["verify_mode"] == "source-conformance"
    assert summary["training"]["reuse_outcome"]["failure_count"] == 0
    assert set(summary["openspiel"]) == {
        "connect_four",
        "kuhn_poker",
        "matrix_rps",
    }
    assert all(
        result["equivalence_ok"] and result["outcome"]["failure_count"] == 0
        for result in summary["openspiel"].values()
    )
    assert summary["authenticity"]["verified"] is True
    assert all(item["verified"] for item in summary["stores"].values())
    assert all(item["identity_equal"] for item in summary["stores"].values())
    assert all(item["signature_valid"] for item in summary["stores"].values())
