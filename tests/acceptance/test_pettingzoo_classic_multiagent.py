"""Acceptance: PettingZoo classic/rps_v2 multi-agent path with portable policies."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("pettingzoo")

from arena.conformance.fixtures import build_fixed_action_rps_policy
from arena.core.manifests import load_manifest, validate_task_manifest
from arena.core.sdk import Match, Policy, Task
from arena.runtime.aec_match import run_aec_match
from arena.runtime.match import run_match

ROOT = Path(__file__).resolve().parents[2]
PARALLEL_TASK = ROOT / "examples" / "tasks" / "pettingzoo-classic-rps.yaml"
AEC_TASK = ROOT / "examples" / "tasks" / "pettingzoo-classic-rps-aec.yaml"


@pytest.mark.acceptance
@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_classic_rps_task_manifests_have_stable_digests() -> None:
    for path in (PARALLEL_TASK, AEC_TASK):
        data = load_manifest(path)
        validate_task_manifest(data)
        assert data["env"] == "classic/rps_v2"
        assert data["digest"].startswith("sha256:")


@pytest.mark.acceptance
@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_classic_rps_parallel_and_aec_match_with_portable_policies(tmp_path: Path) -> None:
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
    seeds = [0, 1, 2]
    parallel_spec = Task.load(PARALLEL_TASK).spec
    aec_spec = Task.load(AEC_TASK).spec

    parallel = run_match(
        task_spec=parallel_spec,
        assignments=assignments,
        seeds=seeds,
        action_mode="deterministic",
        record=True,
        out_dir=tmp_path / "parallel",
    )
    aec = run_aec_match(
        task_spec=aec_spec,
        assignments=assignments,
        seeds=seeds,
        action_mode="deterministic",
        record=True,
        out_dir=tmp_path / "aec",
    )
    assert parallel["outcome"]["episodes_completed"] == len(seeds)
    assert aec["outcome"]["episodes_completed"] == len(seeds)
    assert assignments["player_0"].digest.startswith("sha256:")
    assert assignments["player_1"].digest.startswith("sha256:")
    assert assignments["player_0"].digest != assignments["player_1"].digest

    for i in range(len(seeds)):
        ep_p = json.loads(
            (tmp_path / "parallel" / "trajectories" / f"episode_{i:04d}.json").read_text()
        )
        ep_a = json.loads((tmp_path / "aec" / "trajectories" / f"episode_{i:04d}.json").read_text())
        assert ep_p["returns"] == ep_a["returns"]
        assert ep_p["returns"]["player_0"] == -1.0
        assert ep_p["returns"]["player_1"] == 1.0


def _demo_module():
    import importlib.util

    path = (ROOT / "examples" / "multiagent" / "run_demo.py").resolve()
    spec = importlib.util.spec_from_file_location("arena_multiagent_demo", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.acceptance
@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_multiagent_demo_script(tmp_path: Path) -> None:
    summary = _demo_module().run_multiagent_demo(out=tmp_path / "demo", seeds=[0, 1])
    assert summary["ok"] is True
    assert summary["parity"]["episode_returns_equal"] is True
    assert (tmp_path / "demo" / "summary.json").is_file()
    assert summary["policies"]["player_0"]["digest"].startswith("sha256:")
    assert summary["parallel"]["outcome_digest"].startswith("sha256:")
    assert summary["aec"]["outcome_digest"].startswith("sha256:")


@pytest.mark.acceptance
@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_sdk_match_loads_classic_task_yaml(tmp_path: Path) -> None:
    rock = build_fixed_action_rps_policy(
        tmp_path / "rock", role=["player_0", "player_1"], action=0, name="rock"
    )
    paper = build_fixed_action_rps_policy(
        tmp_path / "paper", role=["player_0", "player_1"], action=1, name="paper"
    )
    match = Match(
        task=Task.load(PARALLEL_TASK),
        assignments={
            "player_0": Policy.load(rock),
            "player_1": Policy.load(paper),
        },
        action_mode="deterministic",
    )
    result = match.run(seeds=[0], record=True, out=tmp_path / "sdk")
    assert result["outcome"]["episodes_completed"] == 1
