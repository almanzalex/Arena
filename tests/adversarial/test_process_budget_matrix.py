"""Process-budget eval failure / missingness matrix (T-502–T-511 style).

Proves hard-budget supervision kills hung reset/action/step/close process groups,
timeouts never fake success, denominators stay coherent, and ``eval report`` exits
6 unless identity-bearing missingness explicitly permits a diagnostic report.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("pettingzoo")

from arena.cli.main import main
from arena.conformance.fixtures import build_fixed_action_rps_policy
from arena.core.errors import ExternalUnavailableError, IncompleteExecutionError
from arena.core.supervisor import run_supervised
from arena.runtime.evaluation import build_eval_report, run_evaluation


def _pair(tmp_path: Path) -> tuple[Path, Path]:
    left = build_fixed_action_rps_policy(
        tmp_path / "left.arena",
        role=["player_0", "player_1"],
        action=0,
    )
    right = build_fixed_action_rps_policy(
        tmp_path / "right.arena",
        role=["player_0", "player_1"],
        action=1,
    )
    return left, right


def _suite(
    *,
    left: Path,
    right: Path,
    seeds: list[int] | dict,
    budgets: dict | None = None,
    failure_policy: dict | None = None,
    name: str = "budget-matrix",
) -> dict:
    suite: dict = {
        "schema": "arena.evaluation/v0alpha1",
        "name": name,
        "task": {
            "adapter": "pettingzoo-parallel",
            "env": "arena/competitive_rps_v0",
            "interaction": "parallel",
            "config": {"max_cycles": 1},
        },
        "assignments": {
            "player_0": str(left.resolve()),
            "player_1": str(right.resolve()),
        },
        "seeds": seeds,
        "action_mode": "deterministic",
        "metrics": ["mean_return"],
    }
    if budgets is not None:
        suite["budgets"] = budgets
    if failure_policy is not None:
        suite["failure_policy"] = failure_policy
    return suite


@pytest.mark.parametrize("boundary", ["reset", "action", "step", "close"])
def test_supervisor_kills_hang_at_boundary_with_grandchild(
    tmp_path: Path, boundary: str
) -> None:
    """T-508: hung reset/action/step/close + grandchild → process-group kill."""
    marker = tmp_path / f"{boundary}-grandchild-survived"
    child = (
        "import pathlib,sys,time;"
        "time.sleep(1.5);"
        "pathlib.Path(sys.argv[1]).write_text('bad')"
    )
    parent = (
        "import subprocess,sys,time;"
        f"subprocess.Popen([sys.executable,'-c',{child!r},sys.argv[1]]);"
        f"print({boundary!r}, flush=True);"
        "time.sleep(60)"
    )
    with pytest.raises(ExternalUnavailableError, match="wall time") as caught:
        run_supervised(
            [sys.executable, "-c", parent, str(marker)],
            timeout_seconds=0.2,
            grace_seconds=0.5,
        )
    assert caught.value.code == "EXTERNAL_TIMEOUT"
    time.sleep(1.8)
    assert not marker.exists(), f"grandchild survived after {boundary} hang kill"


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
@pytest.mark.parametrize("boundary", ["reset", "action", "step", "close"])
def test_eval_process_budget_hang_at_boundary_kills_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    """T-503/T-508 via eval worker: hang boundary + grandchild → failed ledger."""
    left, right = _pair(tmp_path)
    marker = tmp_path / f"eval-{boundary}-grandchild"
    monkeypatch.setenv("ARENA_TEST_HANG_BOUNDARY", boundary)
    monkeypatch.setenv("ARENA_TEST_HANG_MARKER", str(marker))
    suite = _suite(
        left=left,
        right=right,
        seeds=[0],
        budgets={"executor": "process", "timeout_seconds": 0.35},
        failure_policy={"missingness": "fail", "max_failed_episodes": 0},
        name=f"hang-{boundary}",
    )
    result = run_evaluation(
        suite,
        policy_index={},
        out_dir=tmp_path / f"hang-{boundary}",
        record=False,
    )
    assert result["state"] == "failed"
    assert result["denominators"] == {"attempted": 1, "completed": 0, "failed": 1}
    failures = (result["cell_results"][0].get("run") or {}).get("failures") or []
    assert failures
    assert failures[0]["kind"] == "timeout"
    assert failures[0].get("code") == "EXTERNAL_TIMEOUT"
    with pytest.raises(IncompleteExecutionError, match="incomplete evaluation"):
        build_eval_report(result)
    time.sleep(1.8)
    assert not marker.exists(), f"eval worker grandchild survived {boundary} hang"


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_hard_timeout_no_fake_success_and_report_exit_6(tmp_path: Path) -> None:
    """Timeout publishes failed denominators; eval report CLI exits 6."""
    left, right = _pair(tmp_path)
    out = tmp_path / "timed-out"
    suite = _suite(
        left=left,
        right=right,
        seeds=[0, 1],
        budgets={"executor": "process", "timeout_seconds": 0.000001},
        failure_policy={"missingness": "fail", "max_failed_episodes": 0},
    )
    result = run_evaluation(suite, policy_index={}, out_dir=out, record=False)
    assert result["state"] == "failed"
    assert result["denominators"] == {"attempted": 2, "completed": 0, "failed": 2}
    assert (out / "eval_run.json").is_file()
    assert not (out / "report.json").exists()
    with pytest.raises(IncompleteExecutionError) as caught:
        build_eval_report(result)
    assert caught.value.exit_code == 6
    assert caught.value.code == "EVALUATION_INCOMPLETE"
    code = main(["eval", "report", str(out), "--json"])
    assert code == 6


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_one_cell_fails_others_finish_no_valid_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T-502: one hung cell + one finishing cell → incomplete, report refused."""
    left, right = _pair(tmp_path)
    rock = build_fixed_action_rps_policy(
        tmp_path / "rock.arena", role=["player_0", "player_1"], action=0
    )
    paper = build_fixed_action_rps_policy(
        tmp_path / "paper.arena", role=["player_0", "player_1"], action=1
    )
    # Two cells via distinct opponent paths as two separate one-seed suites run
    # under process budgets: hang env forces first worker path.
    marker = tmp_path / "partial-grandchild"
    hang_calls = {"n": 0}
    real_run = run_evaluation

    def run_once(hang: bool, out: Path) -> dict:
        if hang:
            monkeypatch.setenv("ARENA_TEST_HANG_BOUNDARY", "step")
            monkeypatch.setenv("ARENA_TEST_HANG_MARKER", str(marker))
            budgets = {"executor": "process", "timeout_seconds": 0.35}
        else:
            monkeypatch.delenv("ARENA_TEST_HANG_BOUNDARY", raising=False)
            monkeypatch.delenv("ARENA_TEST_HANG_MARKER", raising=False)
            budgets = {"executor": "process", "timeout_seconds": 30}
        suite = _suite(
            left=left if hang else rock,
            right=right if hang else paper,
            seeds=[0],
            budgets=budgets,
            failure_policy={"missingness": "fail"},
        )
        return real_run(suite, policy_index={}, out_dir=out, record=False)

    # Compose a two-cell suite with population crossplay so one process budget
    # covers both cells; inject hang only for the first scheduled cell via env
    # that the first worker inherits, then clear for subsequent workers.
    from arena.core.population import create_population
    from arena.core.sdk import Policy
    from arena.core.store import LocalStore

    LocalStore(tmp_path).init()
    store = LocalStore(tmp_path)
    pop = create_population(
        name="opp",
        members=[{"policy": str(rock)}, {"policy": str(paper)}],
        store=store,
    )
    cand = Policy.load(left)
    hang_calls = {"n": 0}
    from arena.runtime import evaluation as evaluation_mod

    real_supervised = evaluation_mod._run_cell_supervised

    def selective_supervised(**kwargs):
        hang_calls["n"] += 1
        if hang_calls["n"] == 1:
            monkeypatch.setenv("ARENA_TEST_HANG_BOUNDARY", "action")
            monkeypatch.setenv("ARENA_TEST_HANG_MARKER", str(marker))
            kwargs = {**kwargs, "timeout_seconds": 0.35}
        else:
            monkeypatch.delenv("ARENA_TEST_HANG_BOUNDARY", raising=False)
            monkeypatch.delenv("ARENA_TEST_HANG_MARKER", raising=False)
            kwargs = {**kwargs, "timeout_seconds": 30.0}
        return real_supervised(**kwargs)

    monkeypatch.setattr(evaluation_mod, "_run_cell_supervised", selective_supervised)
    suite = {
        "schema": "arena.evaluation/v0alpha1",
        "name": "partial-fail",
        "task": {
            "adapter": "pettingzoo-parallel",
            "env": "arena/competitive_rps_v0",
            "interaction": "parallel",
            "config": {"max_cycles": 1},
        },
        "assignments": {
            "player_0": {"kind": "policy", "policy": cand.digest},
            "player_1": {"kind": "crossplay", "population": pop["digest"]},
        },
        "seeds": [0],
        "action_mode": "deterministic",
        "metrics": ["mean_return"],
        "budgets": {"executor": "process", "timeout_seconds": 30},
        "failure_policy": {"missingness": "fail", "max_failed_episodes": 0},
        "sampling": {"kind": "enumerated_crossplay", "seed": 0},
    }
    policy_index = {
        cand.digest: Path(left),
        Policy.load(rock).digest: Path(rock),
        Policy.load(paper).digest: Path(paper),
    }
    result = run_evaluation(
        suite,
        policy_index=policy_index,
        populations={pop["digest"]: pop},
        out_dir=tmp_path / "partial",
        workers=1,
        record=False,
    )
    assert result["denominators"]["attempted"] == 2
    assert result["denominators"]["failed"] >= 1
    assert result["denominators"]["completed"] >= 1
    assert result["state"] == "incomplete"
    with pytest.raises(IncompleteExecutionError):
        build_eval_report(result)
    assert main(["eval", "report", str(tmp_path / "partial"), "--json"]) == 6
    time.sleep(1.8)
    assert not marker.exists()


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
@pytest.mark.parametrize(
    ("max_failed", "expect_report"),
    [
        (0, False),
        (1, True),
        (9, True),
    ],
)
def test_missingness_threshold_matrix(
    tmp_path: Path, max_failed: int, expect_report: bool
) -> None:
    """T-509: one failed seed under each missingness threshold."""
    left, right = _pair(tmp_path)
    # Force a single-seed process timeout failure (failed=1, completed=0).
    suite = _suite(
        left=left,
        right=right,
        seeds=[0],
        budgets={"executor": "process", "timeout_seconds": 0.000001},
        failure_policy={
            "missingness": "allow",
            "max_failed_episodes": max_failed,
        },
    )
    result = run_evaluation(
        suite, policy_index={}, out_dir=tmp_path / f"miss-{max_failed}", record=False
    )
    assert result["state"] == "failed"
    assert result["denominators"] == {"attempted": 1, "completed": 0, "failed": 1}
    result["suite"] = suite
    if expect_report:
        report = build_eval_report(result)
        assert report["state"] == "failed"
        assert report["denominators"]["failed"] == 1
        assert report["denominators"]["attempted"] == 1
        code = main(["eval", "report", str(tmp_path / f"miss-{max_failed}"), "--json"])
        assert code == 0
    else:
        with pytest.raises(IncompleteExecutionError):
            build_eval_report(result)
        code = main(["eval", "report", str(tmp_path / f"miss-{max_failed}"), "--json"])
        assert code == 6


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_missingness_fail_default_blocks_partial_metrics(tmp_path: Path) -> None:
    left, right = _pair(tmp_path)
    suite = _suite(
        left=left,
        right=right,
        seeds=list(range(10)),
        budgets={"executor": "process", "timeout_seconds": 0.000001},
        failure_policy={"missingness": "fail"},
    )
    result = run_evaluation(
        suite, policy_index={}, out_dir=tmp_path / "ten-fail", record=False
    )
    assert result["denominators"] == {"attempted": 10, "completed": 0, "failed": 10}
    assert result["state"] == "failed"
    with pytest.raises(IncompleteExecutionError):
        build_eval_report({**result, "suite": suite})


def test_build_eval_report_allow_keeps_denominators() -> None:
    """Missingness allow never invents completed episodes."""
    eval_run = {
        "evaluation_digest": "sha256:" + ("a" * 64),
        "evaluation_intent_digest": "sha256:" + ("b" * 64),
        "execution_binding_digest": "sha256:" + ("c" * 64),
        "semantic_result_digest": "sha256:" + ("d" * 64),
        "state": "incomplete",
        "denominators": {"attempted": 10, "completed": 9, "failed": 1},
        "cells": [],
        "cell_results": [],
        "suite": {
            "metrics": ["mean_return"],
            "failure_policy": {"missingness": "allow", "max_failed_episodes": 1},
        },
        "provider": {"kind": "native"},
        "task_digest": "sha256:" + ("e" * 64),
    }
    report = build_eval_report(eval_run)
    assert report["state"] == "incomplete"
    assert report["denominators"] == {"attempted": 10, "completed": 9, "failed": 1}
