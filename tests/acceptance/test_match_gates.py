"""Acceptance tests M-01, M-02, D-01 + F5 PettingZoo Parallel match."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("pettingzoo")

import torch  # noqa: E402

from rlx.adapters.policy_custom_torch import build_module, export_policy  # noqa: E402
from rlx.conformance.fixtures import build_rps_policy  # noqa: E402
from rlx.core.errors import CompatibilityError  # noqa: E402
from rlx.core.sdk import Match, Policy, Task, check  # noqa: E402
from rlx.runtime.trajectory import inspect_trajectory  # noqa: E402

_PILOT = "rlx/competitive_rps_v0"


def _build_match(
    tmp_path: Path,
    *,
    action_mode: str = "deterministic",
    max_cycles: int | None = None,
    seed0: int = 10,
    seed1: int = 20,
    assign_both: bool = True,
) -> Match:
    p0 = build_rps_policy(tmp_path / "p0", role="player_0", seed=seed0)
    p1 = build_rps_policy(tmp_path / "p1", role="player_1", seed=seed1)
    task_spec: dict[str, Any] = {"adapter": "pettingzoo-parallel", "env": _PILOT}
    if max_cycles is not None:
        task_spec["config"] = {"max_cycles": max_cycles}
    assignments = {"player_0": Policy.load(p0)}
    if assign_both:
        assignments["player_1"] = Policy.load(p1)
    return Match(
        task=Task.load(task_spec),
        assignments=assignments,
        action_mode=action_mode,
        failure_policy={"timeout_seconds": 30, "retain_incomplete": True, "retry": 0},
    )


def _build_policy(
    path: Path,
    *,
    role: str,
    action_n: int = 3,
    obs_n: int = 4,
    masks: str = "none",
    seed: int = 0,
) -> Path:
    """Export a minimal Discrete policy with configurable (possibly incompatible) spaces."""
    arch = {"type": "mlp_categorical", "observation_dim": obs_n, "hidden_dims": [16], "action_n": action_n}
    torch.manual_seed(seed)
    return export_policy(
        out_dir=path,
        name=f"p-{role}-a{action_n}-o{obs_n}-{masks}",
        roles=[role],
        observation={"type": "Discrete", "n": obs_n, "dtype": "int64"},
        action={"type": "Discrete", "n": action_n, "dtype": "int64", "masks": masks},
        architecture=arch,
        state_dict=build_module(arch).state_dict(),
        preprocessing={"id": "normalize_v0", "mean": 0.0, "std": 1.0},
        modes=["deterministic", "stochastic"],
    )


def _actions(run_dir: Path, i: int) -> list[dict[str, int]]:
    ep = json.loads((run_dir / "trajectories" / f"episode_{i:04d}.json").read_text())
    return [step["actions"] for step in ep["steps"]]


def _normalize_run(record: dict[str, Any]) -> dict[str, Any]:
    """Drop volatile provenance (timestamped run id + creation time) for equality."""
    norm = dict(record)
    norm.pop("run_id", None)
    norm.pop("created_at", None)
    return norm


@pytest.mark.acceptance
@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_f5_parallel_match_deterministic(tmp_path: Path) -> None:
    """F5: simultaneous joint actions applied atomically; deterministic replay is stable."""
    match = _build_match(tmp_path, action_mode="deterministic", max_cycles=4)
    seeds = [0, 1, 2, 3, 4]
    r1 = match.run(seeds=seeds, record=True, out=tmp_path / "run1")
    r2 = match.run(seeds=seeds, record=True, out=tmp_path / "run2")
    assert r1["outcome"]["episodes_completed"] == len(seeds)
    assert r2["outcome"]["episodes_completed"] == len(seeds)
    for i in range(len(seeds)):
        assert _actions(tmp_path / "run1", i) == _actions(tmp_path / "run2", i)


@pytest.mark.acceptance
@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_m01_reproducibility_stochastic(tmp_path: Path) -> None:
    """M-01: repeated seeded matches with a *stochastic* policy over multiple cycles
    (non-trivial trajectories) are element-/byte-identical across runs and record files,
    while *distinct* seeds yield *distinct* trajectories (so reproducibility is not vacuous)."""
    match = _build_match(tmp_path, action_mode="stochastic", max_cycles=8)
    seeds = [0, 1, 2, 3, 4, 5, 6, 7]

    r1 = match.run(seeds=seeds, record=True, out=tmp_path / "run1")
    r2 = match.run(seeds=seeds, record=True, out=tmp_path / "run2")

    # Sanity: the trajectories must be non-trivial (multi-step, seed-dependent) — otherwise
    # a constant single-step episode would make "reproducibility" meaningless.
    ep0 = json.loads((tmp_path / "run1" / "trajectories" / "episode_0000.json").read_text())
    assert len(ep0["steps"]) == 8

    # Element-identical run records across repeated runs (modulo timestamp/run id).
    assert _normalize_run(r1) == _normalize_run(r2)

    # Byte-identical trajectory episode files across repeated runs.
    per_seed_actions: dict[int, str] = {}
    for i, s in enumerate(seeds):
        f1 = (tmp_path / "run1" / "trajectories" / f"episode_{i:04d}.json").read_bytes()
        f2 = (tmp_path / "run2" / "trajectories" / f"episode_{i:04d}.json").read_bytes()
        assert f1 == f2, f"episode {i} (seed {s}) not byte-identical across repeated runs"
        per_seed_actions[s] = json.dumps(_actions(tmp_path / "run1", i))

    # Byte-identical bundle manifests across repeated runs.
    b1 = (tmp_path / "run1" / "trajectories" / "bundle.json").read_bytes()
    b2 = (tmp_path / "run2" / "trajectories" / "bundle.json").read_bytes()
    assert b1 == b2

    # Non-vacuous: different seeds must produce different action trajectories.
    distinct = set(per_seed_actions.values())
    assert len(distinct) >= 2, "different seeds produced identical trajectories (seeding is vacuous)"


@pytest.mark.acceptance
@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_m02_adapter_runtime_error_recorded(tmp_path: Path) -> None:
    """M-02 (adapter runtime error): a policy that raises during act() is recorded as a
    per-episode failure with an actionable kind/agent/message; healthy episodes still complete."""
    from rlx.adapters import policy_custom_torch as pct
    from rlx.runtime import match as match_mod

    match = _build_match(tmp_path, max_cycles=2)
    real_load = pct.load_runtime

    class Boom:
        def reset(self, *a, **k):
            return None

        def reset_agent(self, *a, **k):
            return None

        def act(self, *a, **k):
            raise RuntimeError("adapter blew up")

    def fake_load(path):
        return Boom() if "p0" in str(path) else real_load(path)

    match_mod.load_runtime = fake_load  # type: ignore[assignment]
    pct.load_runtime = fake_load  # type: ignore[assignment]
    try:
        result = match.run(seeds=[0, 1], record=True, out=tmp_path / "failrun")
    finally:
        match_mod.load_runtime = real_load  # type: ignore[assignment]
        pct.load_runtime = real_load

    assert result["outcome"]["failure_count"] == 2
    assert result["outcome"]["episodes_completed"] == 0
    for f in result["failures"]:
        assert f["kind"] == "policy_failure"
        assert f["agent"] == "player_0"
        assert "adapter blew up" in f["message"]
    # Never silently dropped: every requested seed is accounted for in the record.
    assert {f["seed"] for f in result["failures"]} == {0, 1}
    assert len(result["episodes"]) == 2


@pytest.mark.acceptance
@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_m02_crash_recorded(tmp_path: Path) -> None:
    """M-02 (crash): an unexpected environment error is recorded as a `crash` failure with a
    traceback, and subsequent episodes still run — failures are never swallowed."""
    from rlx.runtime import match as match_mod

    match = _build_match(tmp_path, max_cycles=2)
    real_make_env = match_mod.make_env
    calls = {"n": 0}

    def crashing_make_env(spec):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("simulated environment crash")
        return real_make_env(spec)

    match_mod.make_env = crashing_make_env  # type: ignore[assignment]
    try:
        result = match.run(seeds=[0, 1], record=True, out=tmp_path / "crashrun")
    finally:
        match_mod.make_env = real_make_env  # type: ignore[assignment]

    crashes = [f for f in result["failures"] if f["kind"] == "crash"]
    assert len(crashes) == 1
    assert "simulated environment crash" in crashes[0]["message"]
    assert crashes[0].get("traceback")
    assert result["outcome"]["episodes_completed"] == 1
    assert len(result["episodes"]) == 2


@pytest.mark.acceptance
@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_m02_timeout_recorded_with_partial(tmp_path: Path) -> None:
    """M-02 (timeout + incomplete episode): exceeding the budget records a `timeout` failure
    and retains the partial episode when the failure policy asks for it."""
    p0 = build_rps_policy(tmp_path / "p0", role="player_0", seed=10)
    p1 = build_rps_policy(tmp_path / "p1", role="player_1", seed=20)
    match = Match(
        task=Task.load(
            {"adapter": "pettingzoo-parallel", "env": _PILOT, "config": {"max_cycles": 100}}
        ),
        assignments={"player_0": Policy.load(p0), "player_1": Policy.load(p1)},
        action_mode="deterministic",
        failure_policy={"timeout_seconds": 1e-6, "retain_incomplete": True, "retry": 0},
    )
    out = tmp_path / "timeoutrun"
    result = match.run(seeds=[0], record=True, out=out)

    assert result["outcome"]["failure_count"] == 1
    fail = result["failures"][0]
    assert fail["kind"] == "timeout"
    assert "timeout" in fail["message"]
    # Incomplete episode is retained rather than silently dropped.
    assert (out / "trajectories" / "episode_0000.json").exists()


@pytest.mark.acceptance
@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_m02_missing_agent_recorded(tmp_path: Path) -> None:
    """M-02 (missing action for a required agent): if a required agent has no assigned policy,
    the runtime records an `invalid_action` failure naming the agent (never a silent no-op)."""
    match = _build_match(tmp_path, assign_both=False)
    out = tmp_path / "missingrun"
    result = match.run(seeds=[0], record=True, out=out)

    assert result["outcome"]["failure_count"] == 1
    fail = result["failures"][0]
    assert fail["kind"] == "invalid_action"
    assert fail["agent"] == "player_1"
    assert "no policy assigned" in fail["message"]


@pytest.mark.acceptance
@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_m02_precheck_incompatibilities_before_run(tmp_path: Path) -> None:
    """M-02 (pre-run accounting): incompatible action/observation space, role mismatch, and a
    policy that requires masks the task cannot supply are all caught with actionable reports
    *before* any episode runs (no run directory is created)."""
    good1 = build_rps_policy(tmp_path / "g1", role="player_1", seed=2)

    # (a) incompatible action space (Discrete n=5 vs task Discrete n=3)
    bad_action = _build_policy(tmp_path / "bad_action", role="player_0", action_n=5)
    # (b) incompatible observation space (Discrete n=6 vs task Discrete n=4)
    bad_obs = _build_policy(tmp_path / "bad_obs", role="player_0", obs_n=6)
    # (c) role mismatch: a player_1-only policy assigned to agent player_0
    good0_as_p1 = build_rps_policy(tmp_path / "g0", role="player_1", seed=1)
    # (d) missing required mask: policy needs masks, RPS task provides none
    mask_pol = _build_policy(tmp_path / "mask", role="player_0", masks="required")

    task = Task.load({"adapter": "pettingzoo-parallel", "env": _PILOT})

    cases: list[tuple[str, dict[str, Policy], str]] = [
        ("action", {"player_0": Policy.load(bad_action), "player_1": Policy.load(good1)}, "ACTION_MISMATCH"),
        ("obs", {"player_0": Policy.load(bad_obs), "player_1": Policy.load(good1)}, "OBSERVATION_MISMATCH"),
        ("role", {"player_0": Policy.load(good0_as_p1), "player_1": Policy.load(good1)}, "ROLE_MISMATCH"),
        ("mask", {"player_0": Policy.load(mask_pol), "player_1": Policy.load(good1)}, "MASK_REQUIRED"),
    ]

    for label, assignments, expected_code in cases:
        match = Match(task=task, assignments=assignments, action_mode="deterministic")
        out = tmp_path / f"never_{label}"
        with pytest.raises(CompatibilityError) as exc:
            match.run(seeds=list(range(1000)), record=True, out=out)
        assert expected_code in str(exc.value), f"{label}: {exc.value}"
        # The long job never started: no output directory was created.
        assert not out.exists(), f"{label}: run directory created despite pre-run failure"

    # The compatibility report is actionable (carries repair suggestions), independent of run().
    report = check(task, Policy.load(bad_action).as_role("player_0"), action_mode="deterministic")
    assert not report.ok
    assert any(issue.repairs for issue in report.issues)


@pytest.mark.acceptance
@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_d01_trajectory_completeness(tmp_path: Path) -> None:
    match = _build_match(tmp_path)
    out = tmp_path / "traj"
    match.run(seeds=[0, 1], record=True, out=out)
    info = inspect_trajectory(out / "trajectories")
    assert info["completeness"]["ok"]
    assert info["completeness"]["checked"] >= 1
    ep = json.loads((out / "trajectories" / "episode_0000.json").read_text())
    assert ep["seed"] is not None
    assert ep["task"]["env"]
    assert ep["policies"]
    assert ep["role_map"]
    for step in ep["steps"]:
        for agent in step["actions"]:
            assert agent in step["observations"]
            assert agent in step["rewards"]
            assert agent in step["terminations"]
            assert agent in step["truncations"]
