"""Claim 7 (adversarial): trajectory completeness (D-01) in edge environments.

Attacks: assert every transition carries task/agent/role/policy/seed/obs/action/
reward/terminal provenance even for single-agent Parallel, a large discrete action
space, and immediate-termination tasks.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("pettingzoo")

from _adv_envs import (  # noqa: E402
    ImmediateTermParallel,
    LargeDiscreteParallel,
    SingleAgentParallel,
    make_discrete_policy,
)

from rlx.core.sdk import Match, Policy, Task  # noqa: E402
from rlx.runtime.trajectory import inspect_trajectory  # noqa: E402

_REQUIRED_EP = {"schema", "seed", "task", "agents", "role_map", "policies", "steps"}
_REQUIRED_STEP = {
    "observations",
    "actions",
    "rewards",
    "terminations",
    "truncations",
    "action_masks",
}


def _assert_complete(out: Path, agents: list[str]) -> dict:
    info = inspect_trajectory(out / "trajectories")
    assert info["completeness"]["ok"], info["completeness"]
    ep = json.loads((out / "trajectories" / "episode_0000.json").read_text())
    for key in _REQUIRED_EP:
        assert key in ep, f"episode missing {key}"
    assert ep["seed"] is not None
    assert ep["task"]["env"]
    assert ep["policies"]
    assert ep["role_map"]
    for agent in agents:
        assert agent in ep["role_map"]
        assert agent in ep["policies"]
    for step in ep["steps"]:
        for key in _REQUIRED_STEP:
            assert key in step, f"step missing {key}"
        for agent in step["actions"]:
            assert agent in step["observations"]
            assert agent in step["rewards"]
            assert agent in step["terminations"]
            assert agent in step["truncations"]
    return ep


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_single_agent_parallel_completeness(tmp_path: Path, patch_task_env) -> None:
    patch_task_env(SingleAgentParallel, max_cycles=4)
    p = make_discrete_policy(tmp_path / "solo", role="solo", obs_n=4, action_n=3)
    match = Match(
        task=Task.load({"adapter": "pettingzoo-parallel", "env": "adv/solo"}),
        assignments={"solo": Policy.load(p)},
    )
    out = tmp_path / "solo_run"
    match.run(seeds=[0, 1], record=True, out=out)
    ep = _assert_complete(out, ["solo"])
    assert len(ep["steps"]) == 4
    assert ep["agents"] == ["solo"]


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_large_discrete_completeness(tmp_path: Path, patch_task_env) -> None:
    patch_task_env(LargeDiscreteParallel, max_cycles=3)
    p0 = make_discrete_policy(tmp_path / "p0", role="player_0", obs_n=512, action_n=257)
    p1 = make_discrete_policy(tmp_path / "p1", role="player_1", obs_n=512, action_n=257)
    match = Match(
        task=Task.load({"adapter": "pettingzoo-parallel", "env": "adv/large"}),
        assignments={"player_0": Policy.load(p0), "player_1": Policy.load(p1)},
    )
    out = tmp_path / "large_run"
    match.run(seeds=[0], record=True, out=out)
    ep = _assert_complete(out, ["player_0", "player_1"])
    for step in ep["steps"]:
        for a, act in step["actions"].items():
            assert 0 <= int(act) < 257, f"{a} action {act} out of the declared range"


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_immediate_termination_completeness(tmp_path: Path, patch_task_env) -> None:
    patch_task_env(ImmediateTermParallel)
    p0 = make_discrete_policy(tmp_path / "p0", role="player_0", action_n=3)
    p1 = make_discrete_policy(tmp_path / "p1", role="player_1", action_n=3)
    match = Match(
        task=Task.load({"adapter": "pettingzoo-parallel", "env": "adv/immterm"}),
        assignments={"player_0": Policy.load(p0), "player_1": Policy.load(p1)},
    )
    out = tmp_path / "imm_run"
    match.run(seeds=[0], record=True, out=out)
    ep = _assert_complete(out, ["player_0", "player_1"])
    assert len(ep["steps"]) == 1
    step = ep["steps"][0]
    assert all(step["terminations"].values())
    assert set(step["rewards"]) == {"player_0", "player_1"}
