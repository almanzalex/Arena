"""Claim 6 (adversarial): failure accounting completeness.

Attacks: a policy that returns an out-of-bounds action / a non-integer action / NaN
logits; an env that raises mid-episode; a required agent missing; immediate-termination
and zero-length episodes — every one must be recorded with a cause, never silently
dropped, and every requested seed must be accounted for.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("pettingzoo")

from _adv_envs import (  # noqa: E402
    ImmediateTermParallel,
    MidEpisodeCrashParallel,
    ZeroLengthParallel,
    make_discrete_policy,
)

from arena.core.sdk import Match, Policy, Task  # noqa: E402

_PILOT = "arena/competitive_rps_v0"


def _pilot_match(tmp_path: Path, *, action_mode: str = "deterministic", max_cycles: int = 2) -> Match:
    p0 = make_discrete_policy(tmp_path / "p0", role="player_0", action_n=3, seed=10)
    p1 = make_discrete_policy(tmp_path / "p1", role="player_1", action_n=3, seed=20)
    return Match(
        task=Task.load(
            {"adapter": "pettingzoo-parallel", "env": _PILOT, "config": {"max_cycles": max_cycles}}
        ),
        assignments={"player_0": Policy.load(p0), "player_1": Policy.load(p1)},
        action_mode=action_mode,
        failure_policy={"timeout_seconds": 30, "retain_incomplete": True, "retry": 0},
    )


def _rogue_load(real_load, target_substr: str, rogue):
    def fake(path):
        return rogue if target_substr in str(path) else real_load(path)

    return fake


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_out_of_bounds_action_recorded(tmp_path: Path) -> None:
    """A rogue policy that returns an out-of-range action is recorded as invalid_action
    (the pilot env would otherwise apply it silently)."""
    from arena.adapters import policy_custom_torch as pct
    from arena.runtime import match as match_mod

    match = _pilot_match(tmp_path)
    real = pct.load_runtime

    class Rogue:
        def reset(self, *a, **k): ...
        def reset_agent(self, *a, **k): ...
        def act(self, *a, **k):
            return 99

    match_mod.load_runtime = _rogue_load(real, "p0", Rogue())
    pct.load_runtime = match_mod.load_runtime
    try:
        result = match.run(seeds=[0, 1], record=True, out=tmp_path / "oob")
    finally:
        match_mod.load_runtime = real
        pct.load_runtime = real

    assert result["outcome"]["failure_count"] == 2
    assert result["outcome"]["episodes_completed"] == 0
    for f in result["failures"]:
        assert f["kind"] == "invalid_action"
        assert f["agent"] == "player_0"
        assert "out-of-bounds" in f["message"]
    assert {f["seed"] for f in result["failures"]} == {0, 1}


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_non_integer_action_recorded(tmp_path: Path) -> None:
    from arena.adapters import policy_custom_torch as pct
    from arena.runtime import match as match_mod

    match = _pilot_match(tmp_path)
    real = pct.load_runtime

    class Rogue:
        def reset(self, *a, **k): ...
        def reset_agent(self, *a, **k): ...
        def act(self, *a, **k):
            return "not-an-int"

    match_mod.load_runtime = _rogue_load(real, "p0", Rogue())
    pct.load_runtime = match_mod.load_runtime
    try:
        result = match.run(seeds=[0], record=True, out=tmp_path / "nonint")
    finally:
        match_mod.load_runtime = real
        pct.load_runtime = real

    assert result["outcome"]["failure_count"] == 1
    assert result["failures"][0]["kind"] == "invalid_action"
    assert "non-integer" in result["failures"][0]["message"]


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_nan_logits_recorded_as_policy_failure(tmp_path: Path) -> None:
    """A policy whose network emits non-finite logits is recorded as a policy_failure
    end-to-end (the runtime refuses to emit an action from NaN/Inf)."""
    import torch

    from arena.adapters import policy_custom_torch as pct
    from arena.runtime import match as match_mod

    match = _pilot_match(tmp_path)
    real = pct.load_runtime

    def fake(path):
        rt = real(path)
        if "p0" in str(path):
            with torch.no_grad():
                for p in rt.module.parameters():
                    p.mul_(float("nan"))
        return rt

    match_mod.load_runtime = fake
    pct.load_runtime = fake
    try:
        result = match.run(seeds=[0, 1], record=True, out=tmp_path / "nan")
    finally:
        match_mod.load_runtime = real
        pct.load_runtime = real

    assert result["outcome"]["failure_count"] == 2
    for f in result["failures"]:
        assert f["kind"] == "policy_failure"
        assert "non-finite" in f["message"]


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_env_raises_mid_episode_recorded_as_crash(tmp_path: Path, patch_task_env) -> None:
    patch_task_env(MidEpisodeCrashParallel, max_cycles=5)
    p0 = make_discrete_policy(tmp_path / "p0", role="player_0", action_n=3)
    p1 = make_discrete_policy(tmp_path / "p1", role="player_1", action_n=3)
    match = Match(
        task=Task.load({"adapter": "pettingzoo-parallel", "env": "adv/crash"}),
        assignments={"player_0": Policy.load(p0), "player_1": Policy.load(p1)},
    )
    result = match.run(seeds=[0, 1, 2], record=True, out=tmp_path / "crash")
    crashes = [f for f in result["failures"] if f["kind"] == "crash"]
    assert len(crashes) == 3
    for c in crashes:
        assert "mid-episode environment failure" in c["message"]
        assert c.get("traceback")
    assert {e["seed"] for e in result["episodes"]} == {0, 1, 2}  # every seed accounted


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_missing_required_agent_recorded(tmp_path: Path) -> None:
    """A required agent with no assigned policy is recorded as invalid_action naming the
    agent — never a silent no-op."""
    p0 = make_discrete_policy(tmp_path / "p0", role="player_0", action_n=3)
    match = Match(
        task=Task.load({"adapter": "pettingzoo-parallel", "env": _PILOT}),
        assignments={"player_0": Policy.load(p0)},
    )
    result = match.run(seeds=[0], record=True, out=tmp_path / "missing")
    assert result["outcome"]["failure_count"] == 1
    fail = result["failures"][0]
    assert fail["kind"] == "invalid_action"
    assert fail["agent"] == "player_1"
    assert "no policy assigned" in fail["message"]


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_immediate_termination_recorded_complete(tmp_path: Path, patch_task_env) -> None:
    patch_task_env(ImmediateTermParallel)
    p0 = make_discrete_policy(tmp_path / "p0", role="player_0", action_n=3)
    p1 = make_discrete_policy(tmp_path / "p1", role="player_1", action_n=3)
    match = Match(
        task=Task.load({"adapter": "pettingzoo-parallel", "env": "adv/immterm"}),
        assignments={"player_0": Policy.load(p0), "player_1": Policy.load(p1)},
    )
    out = tmp_path / "immterm"
    result = match.run(seeds=[0, 1], record=True, out=out)
    assert result["outcome"]["episodes_completed"] == 2
    ep = json.loads((out / "trajectories" / "episode_0000.json").read_text())
    assert len(ep["steps"]) == 1
    assert all(ep["steps"][0]["terminations"].values())


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_zero_length_episode_recorded_and_accounted(tmp_path: Path, patch_task_env) -> None:
    ZeroLengthParallel.reset_counter()
    patch_task_env(ZeroLengthParallel)
    p0 = make_discrete_policy(tmp_path / "p0", role="player_0", action_n=3)
    match = Match(
        task=Task.load({"adapter": "pettingzoo-parallel", "env": "adv/zerolen"}),
        assignments={"player_0": Policy.load(p0)},
    )
    out = tmp_path / "zero"
    result = match.run(seeds=[0], record=True, out=out)
    # The zero-length episode is not dropped: it is recorded and the seed is accounted.
    assert len(result["episodes"]) == 1
    assert result["episodes"][0]["seed"] == 0
    ep = json.loads((out / "trajectories" / "episode_0000.json").read_text())
    assert ep["steps"] == []
    assert ep["seed"] == 0


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_process_budget_timeout_is_accounted_not_silent(tmp_path: Path) -> None:
    """Claim 6 + hard budgets: a timed-out cell accounts every seed as failed."""
    from arena.conformance.fixtures import build_fixed_action_rps_policy
    from arena.core.errors import IncompleteExecutionError
    from arena.runtime.evaluation import build_eval_report, run_evaluation

    left = build_fixed_action_rps_policy(
        tmp_path / "p0", role=["player_0", "player_1"], action=0
    )
    right = build_fixed_action_rps_policy(
        tmp_path / "p1", role=["player_0", "player_1"], action=1
    )
    suite = {
        "schema": "arena.evaluation/v0alpha1",
        "name": "c6-budget",
        "task": {
            "adapter": "pettingzoo-parallel",
            "env": _PILOT,
            "interaction": "parallel",
            "config": {"max_cycles": 1},
        },
        "assignments": {
            "player_0": str(left.resolve()),
            "player_1": str(right.resolve()),
        },
        "seeds": [0, 1],
        "action_mode": "deterministic",
        "metrics": ["mean_return"],
        "budgets": {"executor": "process", "timeout_seconds": 0.000001},
    }
    result = run_evaluation(
        suite, policy_index={}, out_dir=tmp_path / "c6-budget", record=False
    )
    assert result["denominators"]["attempted"] == 2
    assert result["denominators"]["failed"] == 2
    assert result["denominators"]["completed"] == 0
    seeds = {
        f["seed"]
        for cell in result["cell_results"]
        for f in (cell.get("run") or {}).get("failures") or []
    }
    assert seeds == {0, 1}
    with pytest.raises(IncompleteExecutionError):
        build_eval_report(result)
