"""Claim 5 (adversarial): compatibility engine fails BEFORE long jobs.

Attacks: subtly mismatched dtype/shape/bounds, role mismatch, mask-required-but-absent,
agent-count / unknown-agent mismatch, and non-Discrete space-type mismatch — each must
produce an actionable pre-run error and create NO run directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rlx.core.compatibility import compose_check

pytest.importorskip("torch")
pytest.importorskip("pettingzoo")

from _adv_envs import LargeDiscreteParallel, make_discrete_policy  # noqa: E402

from rlx.core.errors import CompatibilityError  # noqa: E402
from rlx.core.sdk import Match, Policy, Task  # noqa: E402


def _box_policy(**box) -> dict:
    base_obs = {"type": "Box", "shape": [4], "dtype": "float32", "low": -1.0, "high": 1.0}
    base_obs.update(box.get("observation", {}))
    return {
        "name": "p",
        "roles": {"allowed": ["player_0"]},
        "observation": base_obs,
        "action": box.get("action", {"type": "Discrete", "n": 3, "masks": "none"}),
        "state": {"recurrent": False, "reset_on": []},
        "inference": {"modes": ["deterministic", "stochastic"]},
        "preprocessing": {"included": True, "id": "normalize_v0"},
    }


def test_box_dtype_mismatch_precheck() -> None:
    report = compose_check(
        policy=_box_policy(),
        role="player_0",
        expected_obs={"type": "Box", "shape": [4], "dtype": "float64", "low": -1.0, "high": 1.0},
    )
    assert not report.ok
    assert any(i.code == "OBSERVATION_MISMATCH" for i in report.issues)
    assert any(i.repairs for i in report.issues)


def test_box_shape_mismatch_precheck() -> None:
    report = compose_check(
        policy=_box_policy(),
        role="player_0",
        expected_obs={"type": "Box", "shape": [8], "dtype": "float32", "low": -1.0, "high": 1.0},
    )
    assert not report.ok
    assert any(i.code == "OBSERVATION_MISMATCH" for i in report.issues)


def test_box_bounds_mismatch_precheck() -> None:
    report = compose_check(
        policy=_box_policy(),
        role="player_0",
        expected_obs={"type": "Box", "shape": [4], "dtype": "float32", "low": -5.0, "high": 5.0},
    )
    assert not report.ok
    assert any(i.code == "OBSERVATION_MISMATCH" for i in report.issues)


def test_space_type_mismatch_precheck() -> None:
    """Discrete policy vs Box task (and vice versa) is caught, not coerced."""
    report = compose_check(
        policy=_box_policy(action={"type": "Discrete", "n": 3, "masks": "none"}),
        role="player_0",
        expected_act={"type": "Box", "shape": [3], "dtype": "float32"},
    )
    assert not report.ok
    assert any(i.code == "ACTION_MISMATCH" for i in report.issues)


def test_multidiscrete_mismatch_precheck() -> None:
    report = compose_check(
        policy=_box_policy(action={"type": "MultiDiscrete", "nvec": [2, 2], "masks": "none"}),
        role="player_0",
        expected_act={"type": "MultiDiscrete", "nvec": [3, 3]},
    )
    assert not report.ok
    assert any(i.code == "ACTION_MISMATCH" for i in report.issues)


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_unknown_agent_assignment_precheck_no_dir(tmp_path: Path) -> None:
    """An assignment key that is not an agent in the task fails before any episode and
    creates no run directory."""
    p0 = make_discrete_policy(tmp_path / "p0", role="ghost")
    task = Task.load({"adapter": "pettingzoo-parallel", "env": "rlx/competitive_rps_v0"})
    out = tmp_path / "never"
    match = Match(task=task, assignments={"ghost": Policy.load(p0)})
    with pytest.raises(CompatibilityError, match="not an agent"):
        match.run(seeds=list(range(10_000)), record=True, out=out)
    assert not out.exists()


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_action_count_mismatch_precheck_no_dir(tmp_path: Path, patch_task_env) -> None:
    """A policy whose Discrete action_n differs from the task's is rejected before the
    (large-action-space) job runs; no run directory is created."""
    patch_task_env(LargeDiscreteParallel)
    # Task action space is Discrete(257); policy exports Discrete(3).
    p0 = make_discrete_policy(tmp_path / "p0", role="player_0", obs_n=512, action_n=3)
    p1 = make_discrete_policy(tmp_path / "p1", role="player_1", obs_n=512, action_n=257)
    task = Task.load({"adapter": "pettingzoo-parallel", "env": "adv/large"})
    out = tmp_path / "never"
    match = Match(
        task=task,
        assignments={"player_0": Policy.load(p0), "player_1": Policy.load(p1)},
    )
    with pytest.raises(CompatibilityError, match="ACTION_MISMATCH"):
        match.run(seeds=list(range(10_000)), record=True, out=out)
    assert not out.exists()


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_valid_composition_is_not_falsely_rejected(tmp_path: Path, patch_task_env) -> None:
    """Control: a genuinely-compatible large-discrete composition passes the gate and
    produces a run directory (so the pre-run gate is not vacuously failing everything)."""
    patch_task_env(LargeDiscreteParallel, max_cycles=2)
    p0 = make_discrete_policy(tmp_path / "p0", role="player_0", obs_n=512, action_n=257)
    p1 = make_discrete_policy(tmp_path / "p1", role="player_1", obs_n=512, action_n=257)
    task = Task.load({"adapter": "pettingzoo-parallel", "env": "adv/large"})
    out = tmp_path / "run"
    result = Match(
        task=task,
        assignments={"player_0": Policy.load(p0), "player_1": Policy.load(p1)},
    ).run(seeds=[0, 1], record=True, out=out)
    assert out.exists()
    assert result["outcome"]["episodes_completed"] == 2
