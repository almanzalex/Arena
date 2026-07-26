"""AEC match runner + F6 conformance (Phase 5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("pettingzoo")

from arena.adapters.task_pettingzoo.adapter import describe_task
from arena.conformance.fixtures import build_fixed_action_rps_policy
from arena.core.errors import SchemaError
from arena.core.sdk import Policy
from arena.runtime.aec_match import run_aec_match
from arena.runtime.match import run_match


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_f6_aec_match_parity_with_parallel(tmp_path: Path) -> None:
    """F6: AEC twin of RPS yields same per-episode returns as Parallel for fixed actions."""
    rock = build_fixed_action_rps_policy(
        tmp_path / "rock", role=["player_0", "player_1"], action=0, name="rock"
    )
    paper = build_fixed_action_rps_policy(
        tmp_path / "paper", role=["player_0", "player_1"], action=1, name="paper"
    )
    assignments = {
        "player_0": Policy.load(rock),
        "player_1": Policy.load(paper),
    }
    parallel_spec = {
        "adapter": "pettingzoo-parallel",
        "env": "arena/competitive_rps_v0",
        "interaction": "parallel",
        "config": {"max_cycles": 1},
    }
    aec_spec = {
        "adapter": "pettingzoo-parallel",
        "env": "arena/competitive_rps_aec_v0",
        "interaction": "aec",
        "config": {"max_cycles": 1},
    }
    seeds = [0, 1, 2]
    p = run_match(
        task_spec=parallel_spec,
        assignments=assignments,
        seeds=seeds,
        action_mode="deterministic",
        record=True,
        out_dir=tmp_path / "parallel",
    )
    a = run_aec_match(
        task_spec=aec_spec,
        assignments=assignments,
        seeds=seeds,
        action_mode="deterministic",
        record=True,
        out_dir=tmp_path / "aec",
    )
    assert p["outcome"]["episodes_completed"] == len(seeds)
    assert a["outcome"]["episodes_completed"] == len(seeds)
    for i in range(len(seeds)):
        ep_p = json.loads(
            (tmp_path / "parallel" / "trajectories" / f"episode_{i:04d}.json").read_text()
        )
        ep_a = json.loads((tmp_path / "aec" / "trajectories" / f"episode_{i:04d}.json").read_text())
        assert ep_p["returns"] == ep_a["returns"]
        # Rock vs paper: player_0 loses.
        assert ep_p["returns"]["player_0"] == -1.0
        assert ep_p["returns"]["player_1"] == 1.0


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_aec_describe_task_and_dynamic_agents_flag() -> None:
    info = describe_task(
        {
            "adapter": "pettingzoo-parallel",
            "env": "arena/competitive_rps_aec_v0",
            "interaction": "aec",
        }
    )
    assert info["interaction"] == "aec"
    assert info["dynamic_agents"] is False
    assert set(info["agents"]) == {"player_0", "player_1"}


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_dynamic_agents_fail_loud(tmp_path: Path, monkeypatch) -> None:
    rock = build_fixed_action_rps_policy(
        tmp_path / "rock", role=["player_0", "player_1"], action=0
    )
    paper = build_fixed_action_rps_policy(
        tmp_path / "paper", role=["player_0", "player_1"], action=1
    )

    def _fake_describe(spec):
        base = describe_task(spec)
        return {**base, "dynamic_agents": True}

    monkeypatch.setattr("arena.runtime.aec_match.describe_task", _fake_describe)
    with pytest.raises(SchemaError, match="Dynamic agent"):
        run_aec_match(
            task_spec={
                "adapter": "pettingzoo-parallel",
                "env": "arena/competitive_rps_aec_v0",
                "interaction": "aec",
            },
            assignments={
                "player_0": Policy.load(rock),
                "player_1": Policy.load(paper),
            },
            seeds=[0],
            out_dir=tmp_path / "dyn",
        )
