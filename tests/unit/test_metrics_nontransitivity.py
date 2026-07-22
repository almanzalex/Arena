"""Metrics + non-transitivity guard (Phase 4 / E-04)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("pettingzoo")

from rlx.conformance.fixtures import build_fixed_action_rps_policy
from rlx.core.population import create_population
from rlx.core.sdk import Policy
from rlx.core.store import LocalStore
from rlx.plugins.metrics import PayoffMatrixMetric, detect_nontransitivity
from rlx.runtime.evaluation import build_eval_report, run_evaluation


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_cyclic_rps_population_nontransitivity_warning(tmp_path: Path, monkeypatch) -> None:
    """E-04: cyclic rock/paper/scissors matrix emits warning; matrix retained; no silent ranking."""
    monkeypatch.chdir(tmp_path)
    LocalStore(tmp_path).init()
    store = LocalStore(tmp_path)
    roles = ["player_0", "player_1"]
    rock = build_fixed_action_rps_policy(tmp_path / "rock", role=roles, action=0, name="rock")
    paper = build_fixed_action_rps_policy(tmp_path / "paper", role=roles, action=1, name="paper")
    scissors = build_fixed_action_rps_policy(
        tmp_path / "scissors", role=roles, action=2, name="scissors"
    )
    digests = {
        "rock": Policy.load(rock).digest,
        "paper": Policy.load(paper).digest,
        "scissors": Policy.load(scissors).digest,
    }
    pop = create_population(
        name="cyclic-rps",
        members=[
            {"policy": str(rock), "tags": ["rock"]},
            {"policy": str(paper), "tags": ["paper"]},
            {"policy": str(scissors), "tags": ["scissors"]},
        ],
        store=store,
        ref="populations/cyclic-rps",
    )
    suite = {
        "schema": "rlx.evaluation/v0alpha1",
        "name": "cyclic-matrix",
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
        "metrics": ["payoff_matrix", "mean_return", "win_rate"],
    }
    policy_index = {
        digests["rock"]: Path(rock),
        digests["paper"]: Path(paper),
        digests["scissors"]: Path(scissors),
    }
    result = run_evaluation(
        suite,
        policy_index=policy_index,
        populations={pop["digest"]: pop},
        out_dir=tmp_path / "eval-cyclic",
    )
    assert len(result["cells"]) == 9
    report = build_eval_report(result)
    payoff = report["metrics"]["payoff_matrix"]
    assert payoff["matrix"] is not None
    assert report["nontransitivity_warning"]
    assert "Non-transitive" in report["nontransitivity_warning"]
    assert payoff["ranking"] is None
    # Every summary cell carries evidence refs.
    for refs in payoff["evidence_refs"].values():
        assert refs


def test_detect_nontransitivity_unit() -> None:
    labels = ["a", "b", "c"]
    # Square matrix: a>b, b>c, c>a
    matrix = [[0.0, 1.0, -1.0], [-1.0, 0.0, 1.0], [1.0, -1.0, 0.0]]
    warn = detect_nontransitivity(matrix, labels)
    assert warn is not None
    out = PayoffMatrixMetric().compute(
        [
            {
                "candidate_policy": "a",
                "opponent_policy": "b",
                "episodes": [{"returns": {"player_0": 1.0}}],
                "evidence_refs": ["e1"],
            },
            {
                "candidate_policy": "b",
                "opponent_policy": "c",
                "episodes": [{"returns": {"player_0": 1.0}}],
                "evidence_refs": ["e2"],
            },
            {
                "candidate_policy": "c",
                "opponent_policy": "a",
                "episodes": [{"returns": {"player_0": 1.0}}],
                "evidence_refs": ["e3"],
            },
            {
                "candidate_policy": "a",
                "opponent_policy": "c",
                "episodes": [{"returns": {"player_0": -1.0}}],
                "evidence_refs": ["e4"],
            },
            {
                "candidate_policy": "b",
                "opponent_policy": "a",
                "episodes": [{"returns": {"player_0": -1.0}}],
                "evidence_refs": ["e5"],
            },
            {
                "candidate_policy": "c",
                "opponent_policy": "b",
                "episodes": [{"returns": {"player_0": -1.0}}],
                "evidence_refs": ["e6"],
            },
        ]
    )
    assert out["nontransitivity_warning"]
    assert out["ranking"] is None
