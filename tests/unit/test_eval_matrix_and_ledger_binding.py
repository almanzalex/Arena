"""Heavy coverage: sampling ledger stability + report binding digests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("pettingzoo")

from arena.conformance.fixtures import build_fixed_action_rps_policy
from arena.core.identity import canonical_json, digest_uri, sha256_bytes
from arena.core.population import create_population
from arena.core.sdk import Policy
from arena.core.store import LocalStore
from arena.runtime.eval_matrix import run_crossplay_matrix
from arena.runtime.evaluation import (
    build_eval_report,
    expand_evaluation_cells,
    run_evaluation,
)


def _dual_role_policy(root: Path, *, action: int, name: str) -> Path:
    return build_fixed_action_rps_policy(
        root / f"{name}.arena",
        role=["player_0", "player_1"],
        action=action,
        name=name,
    )


def _cartesian_suite(pop_digest: str) -> dict:
    return {
        "schema": "arena.evaluation/v0alpha1",
        "name": "ledger-binding",
        "interaction": "parallel",
        "task": {
            "adapter": "pettingzoo-parallel",
            "env": "arena/competitive_rps_v0",
            "config": {"max_cycles": 1},
        },
        "assignments": {
            "player_0": {"kind": "crossplay", "population": pop_digest},
            "player_1": {"kind": "crossplay", "population": pop_digest},
        },
        "seeds": {"start": 0, "count": 1},
        "action_mode": "deterministic",
        "metrics": ["payoff_matrix", "mean_return", "win_rate"],
        "sampling": {"kind": "enumerated_crossplay", "seed": 0},
    }


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_sampling_ledger_stable_across_workers_and_reruns(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    LocalStore(tmp_path).init()
    store = LocalStore(tmp_path)

    rock = _dual_role_policy(tmp_path, action=0, name="rock")
    paper = _dual_role_policy(tmp_path, action=1, name="paper")
    scissors = _dual_role_policy(tmp_path, action=2, name="scissors")
    pop = create_population(
        name="cyclic",
        members=[
            {"policy": str(rock)},
            {"policy": str(paper)},
            {"policy": str(scissors)},
        ],
        store=store,
        ref="populations/cyclic",
    )
    suite = _cartesian_suite(pop["digest"])
    policy_index = {
        Policy.load(rock).digest: Path(rock),
        Policy.load(paper).digest: Path(paper),
        Policy.load(scissors).digest: Path(scissors),
    }
    populations = {pop["digest"]: pop}

    cells_a, ledger_a = expand_evaluation_cells(suite, populations=populations)
    cells_b, ledger_b = expand_evaluation_cells(suite, populations=populations)
    assert ledger_a == ledger_b
    assert len(cells_a) == 9
    assert len(ledger_a) == 6  # 3 members × 2 roles
    assert all("policy" in e and "seed" in e and "stream" in e and "role" in e for e in ledger_a)

    r1 = run_evaluation(
        suite,
        policy_index=policy_index,
        populations=populations,
        store=store,
        out_dir=tmp_path / "run-w1",
        workers=1,
    )
    r2 = run_evaluation(
        suite,
        policy_index=policy_index,
        populations=populations,
        store=None,
        out_dir=tmp_path / "run-w4",
        workers=4,
    )
    assert r1["sampling_ledger"] == ledger_a
    assert r2["sampling_ledger"] == ledger_a
    assert r1["sampling_ledger"] == r2["sampling_ledger"]
    assert r1["evaluation_digest"] == r2["evaluation_digest"]
    assert r1["evaluation_intent_digest"] == r2["evaluation_intent_digest"]
    # Binding may include worker count — assert both runs record a binding digest.
    assert r1["execution_binding_digest"].startswith("sha256:")
    assert r2["execution_binding_digest"].startswith("sha256:")


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_report_binds_ledger_and_intent_binding_digests(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    LocalStore(tmp_path).init()
    store = LocalStore(tmp_path)

    rock = _dual_role_policy(tmp_path, action=0, name="rock")
    paper = _dual_role_policy(tmp_path, action=1, name="paper")
    scissors = _dual_role_policy(tmp_path, action=2, name="scissors")
    pop = create_population(
        name="cyclic",
        members=[
            {"policy": str(rock)},
            {"policy": str(paper)},
            {"policy": str(scissors)},
        ],
        store=store,
    )
    suite = _cartesian_suite(pop["digest"])
    policy_index = {
        Policy.load(rock).digest: Path(rock),
        Policy.load(paper).digest: Path(paper),
        Policy.load(scissors).digest: Path(scissors),
    }
    run = run_evaluation(
        suite,
        policy_index=policy_index,
        populations={pop["digest"]: pop},
        store=store,
        out_dir=tmp_path / "eval-run",
        workers=2,
    )
    report = build_eval_report(run)

    expected_ledger_digest = digest_uri(
        sha256_bytes(canonical_json(run["sampling_ledger"]))
    )
    assert report["sampling_ledger"] == run["sampling_ledger"]
    assert report["sampling_ledger_digest"] == expected_ledger_digest
    assert report["population_digests"] == [pop["digest"]]

    for key in (
        "evaluation_digest",
        "evaluation_intent_digest",
        "execution_binding_digest",
        "semantic_result_digest",
        "eval_run_digest",
        "sampling_ledger_digest",
    ):
        assert report.get(key), f"missing {key}"
        assert str(report[key]).startswith("sha256:")

    # Binding digests on the report must match the eval-run record exactly.
    assert report["evaluation_digest"] == run["evaluation_digest"]
    assert report["evaluation_intent_digest"] == run["evaluation_intent_digest"]
    assert report["execution_binding_digest"] == run["execution_binding_digest"]
    assert report["semantic_result_digest"] == run["semantic_result_digest"]

    assert report["nontransitivity_warning"]
    assert report["metrics"]["payoff_matrix"]["ranking"] is None

    # Rebuilding the report from on-disk eval_run.json must preserve ledger binding.
    disk = json.loads((Path(run["run_dir"]) / "eval_run.json").read_text(encoding="utf-8"))
    disk["cell_results"] = run["cell_results"]
    disk["suite"] = run["suite"]
    rebuilt = build_eval_report(disk)
    assert rebuilt["sampling_ledger_digest"] == report["sampling_ledger_digest"]
    assert rebuilt["sampling_ledger"] == report["sampling_ledger"]
    assert rebuilt["execution_binding_digest"] == report["execution_binding_digest"]


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_eval_matrix_two_policies_to_bound_report(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    LocalStore(tmp_path).init()

    a = _dual_role_policy(tmp_path, action=0, name="alpha")
    b = _dual_role_policy(tmp_path, action=1, name="beta")
    out = tmp_path / "matrix-out"
    result = run_crossplay_matrix(
        [a, b],
        out_dir=out,
        env="arena/competitive_rps_v0",
        config={"max_cycles": 1},
        name="two-policy-matrix",
        workers=2,
    )
    assert result["cells"] == 4  # 2×2
    assert result["population_digest"].startswith("sha256:")
    assert (out / "population.yaml").exists()
    assert (out / "evaluation.yaml").exists()
    assert (out / "report.json").exists()
    assert (out / "eval_run.json").exists()

    report = result["report"]
    assert report["sampling_ledger_digest"] == result["sampling_ledger_digest"]
    assert report["population_digests"] == [result["population_digest"]]
    for key in (
        "evaluation_digest",
        "evaluation_intent_digest",
        "execution_binding_digest",
        "semantic_result_digest",
        "eval_run_digest",
        "sampling_ledger_digest",
    ):
        assert result.get(key) and str(result[key]).startswith("sha256:")

    # Ledger is role-annotated and stable under a second identical matrix run.
    again = run_crossplay_matrix(
        [a, b],
        out_dir=tmp_path / "matrix-out-2",
        env="arena/competitive_rps_v0",
        config={"max_cycles": 1},
        name="two-policy-matrix",
        workers=1,
    )
    assert again["sampling_ledger"] == result["sampling_ledger"]
    assert again["sampling_ledger_digest"] == result["sampling_ledger_digest"]
    assert again["evaluation_digest"] == result["evaluation_digest"]
    assert again["evaluation_intent_digest"] == result["evaluation_intent_digest"]
