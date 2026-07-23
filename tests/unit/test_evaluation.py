"""Parallel evaluation expansion, sampling ledger, cross-play (Phase 3)."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("pettingzoo")

from rlx.conformance.fixtures import build_fixed_action_rps_policy, build_rps_policy
from rlx.core.errors import SchemaError
from rlx.core.population import create_population
from rlx.core.sdk import Policy
from rlx.core.store import LocalStore
from rlx.runtime.evaluation import (
    expand_evaluation_cells,
    run_evaluation,
    validate_evaluation,
)


def _suite(
    *,
    candidate: str,
    population_ref: str,
    seeds: dict | None = None,
    role_swaps: list | None = None,
) -> dict:
    suite = {
        "schema": "rlx.evaluation/v0alpha1",
        "name": "rps-crossplay",
        "interaction": "parallel",
        "task": {
            "adapter": "pettingzoo-parallel",
            "env": "rlx/competitive_rps_v0",
            "version": "rlx-pilot",
            "config": {"max_cycles": 1},
        },
        "assignments": {
            "player_0": {"kind": "policy", "policy": candidate},
            "player_1": {"kind": "crossplay", "population": population_ref},
        },
        "seeds": seeds or {"start": 0, "count": 2},
        "action_mode": "deterministic",
        "metrics": ["payoff_matrix", "mean_return", "win_rate"],
        "sampling": {"kind": "enumerated_crossplay", "seed": 0},
    }
    if role_swaps is not None:
        suite["role_swaps"] = role_swaps
    return suite


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_eval_crossplay_matrix_and_ledger_stable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    LocalStore(tmp_path).init()
    store = LocalStore(tmp_path)
    cand = build_fixed_action_rps_policy(tmp_path / "cand", role="player_0", action=0, name="cand-rock")
    rock = build_fixed_action_rps_policy(tmp_path / "rock", role="player_1", action=0)
    paper = build_fixed_action_rps_policy(tmp_path / "paper", role="player_1", action=1)
    scissors = build_fixed_action_rps_policy(tmp_path / "scissors", role="player_1", action=2)
    pop = create_population(
        name="opp",
        members=[
            {"policy": str(rock), "weight": 1.0},
            {"policy": str(paper), "weight": 1.0},
            {"policy": str(scissors), "weight": 1.0},
        ],
        store=store,
        ref="populations/opp",
    )
    cand_pol = Policy.load(cand)
    suite = _suite(candidate=cand_pol.digest, population_ref=pop["digest"])
    validate_evaluation(suite, populations={pop["digest"]: pop})
    policy_index = {
        cand_pol.digest: Path(cand),
        Policy.load(rock).digest: Path(rock),
        Policy.load(paper).digest: Path(paper),
        Policy.load(scissors).digest: Path(scissors),
    }
    populations = {pop["digest"]: pop}

    r1 = run_evaluation(
        suite,
        policy_index=policy_index,
        populations=populations,
        store=store,
        out_dir=tmp_path / "eval1",
        workers=1,
    )
    from rlx.plugins import interactions

    real_run = interactions.get_interaction("parallel").run_match
    rendezvous = threading.Barrier(2, timeout=5)
    worker_threads: set[int] = set()
    worker_lock = threading.Lock()
    arrivals = 0

    def overlapping_run(**kwargs):
        nonlocal arrivals
        with worker_lock:
            worker_threads.add(threading.get_ident())
            arrivals += 1
            ordinal = arrivals
        if ordinal <= 2:
            rendezvous.wait()
        return real_run(**kwargs)

    monkeypatch.setattr(
        interactions,
        "get_interaction",
        lambda _kind: SimpleNamespace(run_match=overlapping_run),
    )
    r2 = run_evaluation(
        suite,
        policy_index=policy_index,
        populations=populations,
        store=None,
        out_dir=tmp_path / "eval2",
        workers=4,
    )
    assert len(r1["cells"]) == 3
    assert len(worker_threads) >= 2
    assert r1["sampling_ledger"] == r2["sampling_ledger"]
    assert r1["evaluation_digest"] == r2["evaluation_digest"]

    # Episode action digests identical modulo run timestamps.
    def action_stream(run_dir: Path) -> list:
        out = []
        for cell in sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("cell-")):
            for ep in sorted((cell / "trajectories").glob("episode_*.json")):
                data = json.loads(ep.read_text(encoding="utf-8"))
                out.append([s["actions"] for s in data["steps"]])
        return out

    assert action_stream(Path(r1["run_dir"])) == action_stream(Path(r2["run_dir"]))


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_eval_incompatible_role_swap_fails_before_run_dir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    LocalStore(tmp_path).init()
    store = LocalStore(tmp_path)
    cand = build_rps_policy(tmp_path / "cand", role="player_0", seed=0)
    opp = build_rps_policy(tmp_path / "opp", role="player_1", seed=1)
    pop = create_population(
        name="opp",
        members=[{"policy": str(opp)}],
        store=store,
    )
    suite = _suite(
        candidate=Policy.load(cand).digest,
        population_ref=pop["digest"],
        role_swaps=[{"map": {"player_0": "player_1"}, "transform": "bogus"}],
    )
    out = tmp_path / "should-not-exist"
    with pytest.raises(SchemaError, match="unsupported role_swaps.transform"):
        run_evaluation(
            suite,
            policy_index={
                Policy.load(cand).digest: Path(cand),
                Policy.load(opp).digest: Path(opp),
            },
            populations={pop["digest"]: pop},
            out_dir=out,
        )
    assert not out.exists()


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_expand_cartesian_two_populations(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    LocalStore(tmp_path).init()
    store = LocalStore(tmp_path)
    a = build_fixed_action_rps_policy(
        tmp_path / "a", role=["player_0", "player_1"], action=0, name="a"
    )
    b = build_fixed_action_rps_policy(
        tmp_path / "b", role=["player_0", "player_1"], action=1, name="b"
    )
    pop = create_population(
        name="both",
        members=[{"policy": str(a)}, {"policy": str(b)}],
        store=store,
    )
    suite = {
        "schema": "rlx.evaluation/v0alpha1",
        "name": "cart",
        "interaction": "parallel",
        "task": {"adapter": "pettingzoo-parallel", "env": "rlx/competitive_rps_v0"},
        "assignments": {
            "player_0": {"kind": "crossplay", "population": pop["digest"]},
            "player_1": {"kind": "crossplay", "population": pop["digest"]},
        },
        "seeds": {"start": 0, "count": 1},
        "action_mode": "deterministic",
        "metrics": ["payoff_matrix"],
    }
    cells, ledger = expand_evaluation_cells(suite, populations={pop["digest"]: pop})
    assert len(cells) == 4
    assert len(ledger) == 4  # 2 roles × 2 members
