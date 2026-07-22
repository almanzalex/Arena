"""Dataset slice + eval release bundle (Phase 6)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("pettingzoo")

from rlx.conformance.fixtures import build_fixed_action_rps_policy
from rlx.core.dataset import select_episodes
from rlx.core.eval_bundle import build_eval_bundle
from rlx.core.population import create_population
from rlx.core.sdk import Policy
from rlx.core.store import LocalStore
from rlx.runtime.evaluation import build_eval_report, run_evaluation


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_data_select_and_eval_bundle_roundtrip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    LocalStore(tmp_path).init()
    store = LocalStore(tmp_path)
    rock = build_fixed_action_rps_policy(
        tmp_path / "rock", role=["player_0", "player_1"], action=0, name="rock"
    )
    paper = build_fixed_action_rps_policy(
        tmp_path / "paper", role=["player_0", "player_1"], action=1, name="paper"
    )
    pop = create_population(
        name="slice-pop",
        members=[{"policy": str(rock)}, {"policy": str(paper)}],
        store=store,
    )
    suite = {
        "schema": "rlx.evaluation/v0alpha1",
        "name": "slice-suite",
        "interaction": "parallel",
        "task": {
            "adapter": "pettingzoo-parallel",
            "env": "rlx/competitive_rps_v0",
            "config": {"max_cycles": 1},
        },
        "assignments": {
            "player_0": {"kind": "crossplay", "population": pop["digest"]},
            "player_1": {"kind": "crossplay", "population": pop["digest"]},
        },
        "seeds": {"start": 0, "count": 1},
        "action_mode": "deterministic",
        "metrics": ["payoff_matrix", "mean_return"],
    }
    policy_index = {
        Policy.load(rock).digest: Path(rock),
        Policy.load(paper).digest: Path(paper),
    }
    result = run_evaluation(
        suite,
        policy_index=policy_index,
        populations={pop["digest"]: pop},
        out_dir=tmp_path / "eval-run",
    )
    report = build_eval_report(result)

    rock_d = Policy.load(rock).digest
    dataset = select_episodes(
        source_runs=[result["run_dir"]],
        query={"policy": rock_d, "role": "player_0", "outcome": "loss"},
        name="rock-losses",
        out_dir=tmp_path / "dataset",
    )
    assert dataset["schema"] == "rlx.dataset/v0alpha1"
    assert (tmp_path / "dataset" / "dataset.yaml").exists()
    # Rock vs paper: rock loses as player_0 — at least one episode selected.
    assert len(dataset["episodes"]) >= 1
    for ep in dataset["episodes"]:
        assert Path(ep["path"]).exists()
        assert ep["digest"].startswith("sha256:")

    bundle_dir = tmp_path / "bundle"
    bundle = build_eval_bundle(
        eval_run_dir=result["run_dir"],
        report=report,
        out_dir=bundle_dir,
    )
    assert bundle["digest"].startswith("sha256:")
    locked = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))
    assert locked["evaluation_digest"] == result["evaluation_digest"]
    assert "eval_run.json" in locked["artifacts"] or "eval_run.yaml" in locked["artifacts"]
    assert (bundle_dir / "report.json").exists()
    # Hermetic inspect: locked digests present without trainer repos.
    assert locked["reproduce"]["mode"] == "reaggregate_from_locked_rollouts"
