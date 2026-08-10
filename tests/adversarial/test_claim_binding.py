"""Claim-binding / reproducibility hardening.

Proves the lab-facing non-goals for eval claims:
  * same policy+task+seed inputs reproduce digests / match outcomes (exact for discrete RPS)
  * incomplete or mismatched evidence cannot publish as a finished bundle
  * tampered locked artifacts fail verify loudly
  * eval reports bind policy digests + suite identity
  * partial publish crashes leave no finished-looking --out path
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arena.core.errors import IntegrityError, SchemaError
from arena.core.eval_bundle import build_eval_bundle, verify_eval_bundle
from arena.core.identity import digest_uri, sha256_bytes
from arena.core.io import publish_directory
from arena.core.manifests import dump_json, validate_eval_report_manifest
from arena.runtime.evaluation import build_eval_report


def _write_eval_run(
    root: Path,
    *,
    state: str = "complete",
    evaluation_digest: str = "sha256:" + ("a" * 64),
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    dump_json(
        {
            "schema": "arena.eval-run/v1",
            "state": state,
            "evaluation_digest": evaluation_digest,
            "evaluation_name": "claim-binding",
            "cells": [],
            "denominators": {
                "attempted": 1,
                "completed": 1 if state == "complete" else 0,
                "failed": 0 if state == "complete" else 1,
            },
        },
        root / "eval_run.json",
    )
    cell = root / "cell0"
    traj = cell / "trajectories"
    traj.mkdir(parents=True)
    (traj / "episode_0000.json").write_text(
        json.dumps({"steps": [], "seed": 0}), encoding="utf-8"
    )
    return root


def test_refuse_incomplete_eval_bundle(tmp_path: Path) -> None:
    run_dir = _write_eval_run(tmp_path / "run", state="failed")
    with pytest.raises(SchemaError, match="incomplete evaluation"):
        build_eval_bundle(eval_run_dir=run_dir, out_dir=tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_refuse_report_suite_digest_mismatch(tmp_path: Path) -> None:
    digest = "sha256:" + ("b" * 64)
    run_dir = _write_eval_run(tmp_path / "run", evaluation_digest=digest)
    report = {
        "schema": "arena.eval-report/v1",
        "evaluation_digest": "sha256:" + ("c" * 64),
        "evaluation_intent_digest": "sha256:" + ("d" * 64),
        "semantic_result_digest": "sha256:" + ("e" * 64),
        "policy_digests": ["sha256:" + ("f" * 64)],
        "state": "complete",
        "eval_run_digest": "sha256:" + ("1" * 64),
        "metrics": {},
    }
    with pytest.raises(SchemaError, match="evaluation_digest does not match"):
        build_eval_bundle(
            eval_run_dir=run_dir,
            report=report,
            out_dir=tmp_path / "bundle",
        )
    assert not (tmp_path / "bundle").exists()


def test_verify_eval_bundle_rejects_tampered_artifact(tmp_path: Path) -> None:
    digest = "sha256:" + ("a" * 64)
    run_dir = _write_eval_run(tmp_path / "run", evaluation_digest=digest)
    bundle_dir = tmp_path / "bundle"
    build_eval_bundle(eval_run_dir=run_dir, out_dir=bundle_dir)
    ok = verify_eval_bundle(bundle_dir)
    assert ok["ok"] is True
    assert ok["artifact_count"] >= 1

    target = bundle_dir / "eval_run.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["state"] = "tampered"
    target.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(IntegrityError, match="integrity check failed") as exc:
        verify_eval_bundle(bundle_dir)
    assert exc.value.code == "EVAL_BUNDLE_TAMPERED"


def test_verify_eval_bundle_rejects_missing_artifact(tmp_path: Path) -> None:
    digest = "sha256:" + ("a" * 64)
    run_dir = _write_eval_run(tmp_path / "run", evaluation_digest=digest)
    bundle_dir = tmp_path / "bundle"
    build_eval_bundle(eval_run_dir=run_dir, out_dir=bundle_dir)
    (bundle_dir / "eval_run.json").unlink()
    with pytest.raises(IntegrityError, match="missing locked artifact") as exc:
        verify_eval_bundle(bundle_dir)
    assert exc.value.code == "EVAL_BUNDLE_MISSING_ARTIFACT"


def test_partial_publish_crash_leaves_no_finished_out(tmp_path: Path) -> None:
    final = tmp_path / "bundle-out"

    def boom(stage: Path) -> dict:
        (stage / "eval_run.json").write_text('{"state":"complete"}', encoding="utf-8")
        raise RuntimeError("simulated crash before coherent publish")

    with pytest.raises(RuntimeError, match="simulated crash"):
        publish_directory(final, boom, replace=True)
    assert not final.exists(), "partial staging published as finished output"
    assert list(tmp_path.glob(".bundle-out.*.staging")) == []


def test_build_eval_report_requires_bound_policy_digests() -> None:
    digest = "sha256:" + ("a" * 64)
    with pytest.raises(SchemaError, match="no bound policy digests"):
        build_eval_report(
            {
                "state": "complete",
                "evaluation_digest": digest,
                "evaluation_intent_digest": digest,
                "semantic_result_digest": digest,
                "cell_results": [{"assignments": {"player_0": "not-a-digest"}}],
                "suite": {"metrics": ["mean_return"]},
                "denominators": {"attempted": 1, "completed": 1, "failed": 0},
            }
        )


def test_build_eval_report_binds_policy_and_suite_digests() -> None:
    policy_a = "sha256:" + ("1" * 64)
    policy_b = "sha256:" + ("2" * 64)
    suite = "sha256:" + ("3" * 64)
    intent = "sha256:" + ("4" * 64)
    semantic = "sha256:" + ("5" * 64)
    report = build_eval_report(
        {
            "state": "complete",
            "evaluation_digest": suite,
            "evaluation_intent_digest": intent,
            "semantic_result_digest": semantic,
            "object_digest": "sha256:" + ("6" * 64),
            "cell_results": [
                {
                    "assignments": {"player_0": policy_a, "player_1": policy_b},
                    "episodes": [],
                }
            ],
            "suite": {"name": "bound", "metrics": ["mean_return"]},
            "denominators": {"attempted": 1, "completed": 1, "failed": 0},
        }
    )
    validate_eval_report_manifest(report)
    assert report["schema"] == "arena.eval-report/v1"
    assert report["policy_digests"] == sorted([policy_a, policy_b])
    assert report["evaluation_digest"] == suite
    assert report["evaluation_intent_digest"] == intent
    assert report["semantic_result_digest"] == semantic
    assert report["evaluation_name"] == "bound"

    broken = dict(report)
    broken["policy_digests"] = []
    with pytest.raises(SchemaError, match="policy_digests"):
        validate_eval_report_manifest(broken)


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_same_seed_match_outcomes_are_byte_identical(tmp_path: Path) -> None:
    """Same policy+task+seed → identical trajectories.

    Discrete RPS rewards are integers, so numeric tolerance is exact equality.
    """
    pytest.importorskip("torch")
    pytest.importorskip("pettingzoo")

    from arena.conformance.fixtures import build_rps_policy
    from arena.core.sdk import Match, Policy, Task

    p0 = build_rps_policy(tmp_path / "p0", role="player_0", seed=11)
    p1 = build_rps_policy(tmp_path / "p1", role="player_1", seed=22)
    task = Task.load(
        {
            "adapter": "pettingzoo-parallel",
            "env": "arena/competitive_rps_v0",
            "config": {"max_cycles": 4},
        }
    )
    seeds = [0, 1, 2, 3]
    match = Match(
        task=task,
        assignments={"player_0": Policy.load(p0), "player_1": Policy.load(p1)},
        action_mode="stochastic",
    )
    match.run(seeds=seeds, record=True, out=tmp_path / "run-a")
    match.run(seeds=seeds, record=True, out=tmp_path / "run-b")
    for i in range(len(seeds)):
        a = (tmp_path / "run-a" / "trajectories" / f"episode_{i:04d}.json").read_bytes()
        b = (tmp_path / "run-b" / "trajectories" / f"episode_{i:04d}.json").read_bytes()
        assert a == b, f"episode {i} diverged under identical policy+task+seed"

    match.run(seeds=[9, 10, 11, 12], record=True, out=tmp_path / "run-c")
    diverged = any(
        (tmp_path / "run-a" / "trajectories" / f"episode_{i:04d}.json").read_bytes()
        != (tmp_path / "run-c" / "trajectories" / f"episode_{i:04d}.json").read_bytes()
        for i in range(4)
    )
    assert diverged, "changing seeds did not change outcomes (seeding vacuous)"


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_seeded_eval_digests_stable_across_runs_and_workers(
    tmp_path: Path, monkeypatch
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("pettingzoo")

    from arena.conformance.fixtures import build_fixed_action_rps_policy
    from arena.core.population import create_population
    from arena.core.sdk import Policy
    from arena.core.store import LocalStore
    from arena.runtime.evaluation import run_evaluation

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
        name="stable-pop",
        members=[{"policy": str(rock)}, {"policy": str(paper)}],
        store=store,
    )
    suite = {
        "schema": "arena.evaluation/v0alpha1",
        "name": "stable-suite",
        "interaction": "parallel",
        "task": {
            "adapter": "pettingzoo-parallel",
            "env": "arena/competitive_rps_v0",
            "config": {"max_cycles": 1},
        },
        "assignments": {
            "player_0": {"kind": "crossplay", "population": pop["digest"]},
            "player_1": {"kind": "crossplay", "population": pop["digest"]},
        },
        "seeds": {"start": 0, "count": 1},
        "action_mode": "deterministic",
        "metrics": ["payoff_matrix", "mean_return"],
        "sampling": {"kind": "enumerated_crossplay", "seed": 0},
    }
    policy_index = {
        Policy.load(rock).digest: Path(rock),
        Policy.load(paper).digest: Path(paper),
    }

    semantic: list[str] = []
    report_keys: list[tuple[str, tuple[str, ...]]] = []
    bindings: list[str] = []
    claim_artifacts: list[dict[str, str]] = []
    for tag, workers in (("A", 1), ("B", 4)):
        result = run_evaluation(
            suite,
            policy_index=policy_index,
            populations={pop["digest"]: pop},
            out_dir=tmp_path / f"eval-run-{tag}",
            workers=workers,
        )
        assert result["state"] == "complete"
        report = build_eval_report(result)
        bundle = build_eval_bundle(
            eval_run_dir=result["run_dir"],
            report=report,
            out_dir=tmp_path / f"bundle-{tag}",
        )
        locked = json.loads(
            (tmp_path / f"bundle-{tag}" / "bundle.json").read_text(encoding="utf-8")
        )
        semantic.append(result["semantic_result_digest"])
        bindings.append(result["execution_binding_digest"])
        report_keys.append(
            (report["evaluation_digest"], tuple(report["policy_digests"]))
        )
        # Locked trajectories must be byte-stable. report.json may embed the
        # execution binding (worker count), so claim identity is asserted via
        # evaluation/semantic/policy digests above rather than report bytes.
        claim_artifacts.append(
            {
                rel: digest
                for rel, digest in locked["artifacts"].items()
                if rel.startswith("trajectories/")
            }
        )
        verify_eval_bundle(tmp_path / f"bundle-{tag}")
        assert bundle["digest"].startswith("sha256:")

    assert semantic[0] == semantic[1]
    assert report_keys[0] == report_keys[1]
    assert claim_artifacts[0] == claim_artifacts[1]
    assert claim_artifacts[0], "expected locked trajectory claim artifacts"
    assert bindings[0] != bindings[1], "worker count must change execution binding"
    assert report_keys[0][0].startswith("sha256:")
    assert all(d.startswith("sha256:") for d in report_keys[0][1])
    assert semantic[0] != digest_uri(sha256_bytes(b""))


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_hard_timeout_eval_refuses_report_and_bundle(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("pettingzoo")

    from arena.conformance.fixtures import build_fixed_action_rps_policy
    from arena.runtime.evaluation import run_evaluation

    left = build_fixed_action_rps_policy(
        tmp_path / "left.arena", role=["player_0", "player_1"], action=0
    )
    right = build_fixed_action_rps_policy(
        tmp_path / "right.arena", role=["player_0", "player_1"], action=1
    )
    suite = {
        "schema": "arena.evaluation/v0alpha1",
        "name": "forced-timeout",
        "task": {
            "adapter": "pettingzoo-parallel",
            "env": "arena/competitive_rps_v0",
            "interaction": "parallel",
        },
        "assignments": {
            "player_0": str(left.resolve()),
            "player_1": str(right.resolve()),
        },
        "seeds": [0],
        "action_mode": "deterministic",
        "metrics": ["mean_return"],
        "budgets": {"executor": "process", "timeout_seconds": 0.000001},
    }
    result = run_evaluation(suite, policy_index={}, out_dir=tmp_path / "failed-run")
    assert result["state"] in {"failed", "incomplete"}
    with pytest.raises(SchemaError, match="incomplete evaluation"):
        build_eval_report(result)
    out = tmp_path / "should-not-publish"
    with pytest.raises(SchemaError, match="refusing to bundle incomplete"):
        build_eval_bundle(eval_run_dir=result["run_dir"], out_dir=out)
    assert not out.exists()
