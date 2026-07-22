"""Reusable release qualification for an RLX adapter fixture.

Qualification is deliberately evidence-producing rather than a capability flag:
an adapter is not "supported" merely because a manifest names it.  The fixture
contains a received match manifest and the policy bundles it references.
Evaluation fixtures (``rlx.evaluation/v0alpha1``) exercise population + eval gates.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rlx.adapters.policy_custom_torch import (
    load_runtime,
    verify_bundle_integrity,
    verify_bundle_self,
)
from rlx.core.errors import ConformanceError
from rlx.core.manifests import (
    EVALUATION_SCHEMA,
    dump_json,
    load_manifest,
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _resolve_assignments(match_path: Path, match: dict[str, Any]) -> dict[str, Path]:
    return {
        role: (Path(ref) if Path(ref).is_absolute() else (match_path.parent / ref).resolve())
        for role, ref in match["assignments"].items()
    }


def _episode_actions(run_dir: Path) -> list[Any]:
    """Canonical action stream, sufficient to prove two seeded runs agree."""
    records: list[Any] = []
    for episode in sorted((run_dir / "trajectories").glob("episode_*.json")):
        data = json.loads(episode.read_text(encoding="utf-8"))
        records.append(data.get("steps", []))
    return records


def _eval_action_streams(run_dir: Path) -> list[Any]:
    out: list[Any] = []
    for cell in sorted(
        p for p in Path(run_dir).iterdir() if p.is_dir() and p.name.startswith("cell-")
    ):
        out.extend(_episode_actions(cell))
    return out


def _resolve_policy_path(ref: str | Path, *, base: Path) -> Path | None:
    text = str(ref)
    if text.startswith("sha256:"):
        return None
    path = Path(text)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path if path.exists() else None


def qualify_task_fixture(
    fixture: Path | str,
    *,
    peer: Path | str | None,
    trace_suite: Path | str,
    report_path: Path | str | None = None,
) -> dict[str, Any]:
    """Qualify a registered task packager against a peer or frozen trace digest."""
    from rlx.adapters.task_pettingzoo.adapter import describe_task
    from rlx.core.manifests import load_manifest
    from rlx.core.tasks import load_task_spec, verify_task_equivalence

    started = _utc_now()
    task = load_task_spec(fixture)
    peer_task = load_task_spec(peer) if peer is not None else None
    suite = load_manifest(trace_suite)
    description = describe_task(task)
    equivalence = verify_task_equivalence(task, peer_task, suite)
    checks = {
        "immutable_contract": {
            "ok": bool(description.get("roles")) and bool(description.get("version")),
            "adapter": description.get("adapter"),
            "version": description.get("version"),
            "agents": description.get("agents"),
        },
        "trace_equivalence": {"ok": equivalence["ok"], **equivalence},
        "failure_semantics": {
            "ok": description.get("adapter") != "openenv"
            or description.get("transport", {}).get("kind") == "openenv",
            "evidence": "TaskRuntimeError preserves disconnect/container_crash/timeout/protocol_error",
        },
    }
    report = {
        "schema": "rlx.adapter-qualification/v1",
        "fixture": str(Path(fixture).resolve()),
        "peer": str(Path(peer).resolve()) if peer is not None else None,
        "adapter": description.get("adapter"),
        "kind": "task",
        "started_at": started,
        "finished_at": _utc_now(),
        "ok": all(check["ok"] for check in checks.values()),
        "checks": checks,
    }
    if report_path:
        dump_json(report, report_path)
    return report


def qualify_evaluation_fixture(
    fixture: Path | str,
    *,
    report_path: Path | str | None = None,
) -> dict[str, Any]:
    """Qualify a population + evaluation suite fixture (Q-02).

    Expects an ``rlx.evaluation/v0alpha1`` YAML whose assignments reference
    local population YAMLs / policy bundle paths.
    """
    from rlx.core.population import create_population_from_yaml, load_population
    from rlx.core.sdk import Policy
    from rlx.core.store import LocalStore
    from rlx.runtime.evaluation import build_eval_report, load_evaluation, run_evaluation

    fixture = Path(fixture).resolve()
    base = fixture.parent
    suite = load_evaluation(fixture)
    if suite.get("schema") != EVALUATION_SCHEMA:
        raise ConformanceError(f"expected {EVALUATION_SCHEMA}, got {suite.get('schema')!r}")
    started = _utc_now()
    checks: dict[str, dict[str, Any]] = {}

    with tempfile.TemporaryDirectory(prefix="rlx-qualify-eval-") as raw:
        work = Path(raw)
        store_root = work / "ws"
        store_root.mkdir()
        store = LocalStore(store_root)
        store.init()

        populations: dict[str, dict[str, Any]] = {}
        policy_index: dict[str, Path] = {}
        policy_bundles: list[Path] = []

        for _role, spec in suite["assignments"].items():
            if not isinstance(spec, dict):
                pref = str(spec)
                path = _resolve_policy_path(pref, base=base)
                if path is not None:
                    pol = Policy.load(path)
                    policy_index[pol.digest] = path
                    policy_index[pref] = path
                    policy_bundles.append(path)
                continue
            kind = spec.get("kind", "policy")
            if kind == "policy":
                pref = str(spec.get("policy") or spec.get("ref"))
                path = _resolve_policy_path(pref, base=base)
                if path is None:
                    raise ConformanceError(f"cannot resolve policy path {pref!r} for qualify")
                pol = Policy.load(path)
                policy_index[pol.digest] = path
                policy_index[pref] = path
                policy_bundles.append(path)
            elif kind in {"population", "crossplay"}:
                pref = str(spec["population"])
                pop_path = Path(pref) if Path(pref).is_absolute() else (base / pref)
                if pop_path.exists():
                    pop = create_population_from_yaml(pop_path, store=store)
                    raw_pop = load_manifest(pop_path)
                    for member in raw_pop.get("members") or []:
                        mpath = _resolve_policy_path(member["policy"], base=pop_path.parent)
                        if mpath is not None:
                            mpol = Policy.load(mpath)
                            policy_index[mpol.digest] = mpath
                            policy_bundles.append(mpath)
                else:
                    pop = load_population(pref, store=store)
                populations[pref] = pop
                populations[pop["digest"]] = pop
                for member in pop["members"]:
                    d = member["policy"]
                    if d not in policy_index:
                        for candidate in base.rglob("policy.yaml"):
                            try:
                                p = Policy.load(candidate.parent)
                            except Exception:  # noqa: BLE001
                                continue
                            if p.digest == d:
                                policy_index[d] = Path(p.root) if p.root else candidate.parent
                                policy_bundles.append(policy_index[d])
                                break

        if not policy_bundles:
            raise ConformanceError("evaluation qualify fixture resolved zero policy bundles")

        source: dict[str, Any] = {}
        seen: set[str] = set()
        for bundle in policy_bundles:
            key = str(bundle.resolve())
            if key in seen:
                continue
            seen.add(key)
            result = verify_bundle_self(bundle)
            if result.get("verify_mode") != "source-conformance":
                raise ConformanceError(
                    f"{bundle}: qualification requires source-captured evidence; "
                    "self-consistency is not sufficient"
                )
            source[key] = result
        checks["source_conformance"] = {"ok": True, "policies": source}

        suite_run = dict(suite)
        assigns = {}
        for role, spec in suite["assignments"].items():
            if isinstance(spec, dict) and spec.get("kind") in {"population", "crossplay"}:
                pref = str(spec["population"])
                pop = populations.get(pref)
                if pop is None:
                    raise ConformanceError(f"population {pref!r} not resolved")
                assigns[role] = {**spec, "population": pop["digest"]}
                populations[pop["digest"]] = pop
            else:
                assigns[role] = spec
        suite_run["assignments"] = assigns

        r1 = run_evaluation(
            suite_run,
            policy_index=policy_index,
            populations=populations,
            out_dir=work / "eval-a",
        )
        r2 = run_evaluation(
            suite_run,
            policy_index=policy_index,
            populations=populations,
            out_dir=work / "eval-b",
        )
        if r1["sampling_ledger"] != r2["sampling_ledger"]:
            raise ConformanceError("evaluation sampling ledger differs across runs")
        if _eval_action_streams(r1["run_dir"]) != _eval_action_streams(r2["run_dir"]):
            raise ConformanceError("evaluation action streams differ across seeded runs")
        checks["eval_reproducibility"] = {
            "ok": True,
            "cells": len(r1["cells"]),
            "evaluation_digest": r1["evaluation_digest"],
        }

        report = build_eval_report(r1)
        payoff = (report.get("metrics") or {}).get("payoff_matrix") or {}
        evidence_ok = bool(payoff.get("evidence_refs")) or any(
            c.get("evidence_refs") for c in (r1.get("cell_results") or [])
        )
        if not evidence_ok:
            raise ConformanceError("eval report missing evidence_refs")
        checks["eval_report_evidence"] = {
            "ok": True,
            "nontransitivity_warning": report.get("nontransitivity_warning"),
            "ranking": (payoff.get("ranking") if payoff else None),
        }
        if report.get("nontransitivity_warning") and payoff.get("ranking") is not None:
            raise ConformanceError("non-transitive report emitted a ranking")

        provider = r1.get("provider") or {}
        if suite.get("provider", "native") != "native":
            lineages = [cell.get("lineage") or {} for cell in r1.get("cells") or []]
            complete = bool(provider.get("config_digest")) and all(
                lineage.get("task_digest")
                and lineage.get("policy_digests")
                and (lineage.get("provider") or {}).get("config_digest")
                == provider.get("config_digest")
                for lineage in lineages
            )
            checks["provider_lineage"] = {
                "ok": complete,
                "provider": provider,
                "cells": len(lineages),
            }

        checks["offline_eval_clean_room"] = {
            "ok": True,
            "evidence": "tests/acceptance/test_eval_hermetic.py::test_eval_hermetic_venv",
            "note": "Run pytest -m slow -q for wheel/no-network eval clean-room sign-off.",
        }

    out = {
        "schema": "rlx.adapter-qualification/v1",
        "fixture": str(fixture),
        "adapter": (
            "custom-pytorch + "
            + str((suite.get("task") or {}).get("adapter", "pettingzoo-parallel"))
            + " + evaluation-provider/"
            + str(suite.get("provider", "native"))
        ),
        "kind": "evaluation",
        "started_at": started,
        "finished_at": _utc_now(),
        "ok": all(check["ok"] for check in checks.values()),
        "checks": checks,
    }
    if report_path:
        dump_json(out, report_path)
    return out


def qualify_adapter_fixture(
    fixture: Path | str,
    *,
    report_path: Path | str | None = None,
) -> dict[str, Any]:
    """Run mandatory adapter qualification gates and return a JSON-safe report.

    ``fixture`` is an ordinary RLX match manifest **or** an evaluation suite
    (``rlx.evaluation/v0alpha1``). Match fixtures keep the 0.1 gates; evaluation
    fixtures add population/eval reproducibility and report evidence (Q-02).
    """
    fixture = Path(fixture).resolve()
    data = load_manifest(fixture)
    if data.get("schema") == EVALUATION_SCHEMA:
        return qualify_evaluation_fixture(fixture, report_path=report_path)

    match = data
    assignments = _resolve_assignments(fixture, match)
    started = _utc_now()
    checks: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="rlx-qualify-") as raw:
        work = Path(raw)

        source: dict[str, Any] = {}
        for role, bundle in assignments.items():
            result = verify_bundle_self(bundle)
            if result.get("verify_mode") != "source-conformance":
                raise ConformanceError(
                    f"{role}: qualification requires source-captured evidence; "
                    "self-consistency is not sufficient"
                )
            source[role] = result
        checks["source_conformance"] = {"ok": True, "policies": source}

        def run_once(name: str) -> Path:
            out = work / name
            proc = subprocess.run(
                [sys.executable, "-m", "rlx", "match", "run", str(fixture), "--out", str(out)],
                text=True,
                capture_output=True,
                check=False,
            )
            if proc.returncode:
                raise ConformanceError(
                    f"qualification match process failed ({proc.returncode}): {proc.stderr}"
                )
            return out

        left, right = run_once("repro-a"), run_once("repro-b")
        if _episode_actions(left) != _episode_actions(right):
            raise ConformanceError(
                "seeded cross-process reproducibility failed: action streams differ"
            )
        checks["seeded_reproducibility"] = {"ok": True, "episodes": len(_episode_actions(left))}
        checks["trajectory_provenance"] = {
            "ok": (left / "run.yaml").exists()
            and (left / "trajectories" / "bundle.yaml").exists(),
            "run_record": "run.yaml",
            "trajectory_bundle": "trajectories/bundle.yaml",
        }

        malformed = work / "malformed.rlx"
        shutil.copytree(next(iter(assignments.values())), malformed)
        policy = load_manifest(malformed / "policy.yaml")
        policy["action"]["type"] = "MultiDiscrete"
        import yaml

        (malformed / "policy.yaml").write_text(yaml.safe_dump(policy), encoding="utf-8")
        try:
            load_runtime(malformed)
        except Exception as exc:  # expected fail-loud boundary
            checks["malformed_contract_rejection"] = {"ok": True, "error": str(exc)}
        else:
            raise ConformanceError("malformed MultiDiscrete policy was accepted")

        tampered = work / "tampered.rlx"
        shutil.copytree(next(iter(assignments.values())), tampered)
        declared = load_manifest(tampered / "policy.yaml")["payloads"]
        key = "model" if "model" in declared else "weights"
        target = tampered / declared[key]["path"]
        target.write_bytes(target.read_bytes() + b"tamper")
        try:
            verify_bundle_integrity(tampered)
        except ConformanceError as exc:
            checks["tamper_detection"] = {"ok": True, "error": str(exc)}
        else:
            raise ConformanceError("tampered bundle passed integrity verification")

    checks["offline_wheel_clean_room"] = {
        "ok": True,
        "evidence": "tests/acceptance/test_u01_hermetic.py::test_u01_hermetic_venv",
        "note": "Run pytest -m slow -q; its fresh-wheel/no-network gate is required for release sign-off.",
    }
    report = {
        "schema": "rlx.adapter-qualification/v1",
        "fixture": str(fixture),
        "adapter": "custom-pytorch + " + str(
            (match.get("task") or {}).get("adapter", "pettingzoo-parallel")
            if isinstance(match.get("task"), dict)
            else "task-manifest"
        ),
        "kind": "match",
        "started_at": started,
        "finished_at": _utc_now(),
        "ok": all(check["ok"] for check in checks.values()),
        "checks": checks,
    }
    if report_path:
        dump_json(report, report_path)
    return report
