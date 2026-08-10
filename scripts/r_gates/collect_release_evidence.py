"""Collect local R-01…R-14 evidence and emit a release-index skeleton.

This is deliberately *not* ``arena.release-evidence/v1``. A signed release index
must be assembled outside the source commit from complete gate files. This
collector only:

* runs doctor, hermetic-capable inventory, golden fixture digests, perf-smoke
  baselines, and schema-registry / release-truth checks;
* writes per-gate local proof under ``evidence/local/`` when honest;
* emits ``evidence/release-index.json`` with filled vs missing slots;
* accepts ``--attach GATE=path`` for real CI / HF / OpenEnv / Gimitest files;

and **never** invents live HF, separately deployed OpenEnv, or release-CI
Gimitest as passed without an attached real evidence file.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .gates import (
    EXTERNAL_FLOOR_GATES,
    MANDATORY_GATE_IDS,
    NEVER_AUTO_PASS_GATES,
    gate_by_id,
)

COLLECTOR_SCHEMA = "arena.release-evidence-skeleton/v1"
ROOT = Path(__file__).resolve().parents[2]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _git_commit(repo: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = proc.stdout.strip().lower()
    return commit if len(commit) == 40 and all(c in "0123456789abcdef" for c in commit) else None


def _write_json(path: Path, document: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")
    return _digest_bytes(payload.encode("utf-8"))


def _digest_bytes(data: bytes) -> str:
    from arena.core.identity import digest_uri, sha256_bytes

    return digest_uri(sha256_bytes(data))


def _digest_file(path: Path) -> str:
    from arena.core.identity import digest_uri, sha256_file

    return digest_uri(sha256_file(path))


def _parse_attach(values: list[str] | None) -> dict[str, Path]:
    attached: dict[str, Path] = {}
    for raw in values or []:
        if "=" not in raw:
            raise ValueError(
                f"invalid --attach {raw!r}; expected GATE=/path/to/evidence.json"
            )
        gate_id, _, path_text = raw.partition("=")
        gate_id = gate_id.strip().upper()
        if gate_id not in MANDATORY_GATE_IDS:
            raise ValueError(f"unknown gate in --attach: {gate_id}")
        path = Path(path_text).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"attached evidence missing for {gate_id}: {path}")
        attached[gate_id] = path
    return attached


def _reject_simulated_external_floor(gate_id: str, path: Path) -> None:
    """Refuse to treat simulate=/fake live modes as external-floor passes."""
    if gate_id not in EXTERNAL_FLOOR_GATES:
        return
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"attached {gate_id} evidence is not JSON: {path}"
        ) from exc
    if not isinstance(document, dict):
        raise ValueError(f"attached {gate_id} evidence must be a JSON object: {path}")

    mode = str(document.get("mode") or document.get("execution_mode") or "").lower()
    uri = str(document.get("uri") or document.get("store_uri") or "")
    simulated = (
        mode in {"simulate", "simulated", "simulation"}
        or "simulate=" in uri
        or document.get("simulation") is True
        or document.get("simulated") is True
    )
    if simulated:
        raise ValueError(
            f"refusing to attach simulated evidence for {gate_id}: {path}. "
            "Live HF / separately deployed OpenEnv / isolated Gimitest require "
            "real non-simulated qualification JSON."
        )

    if gate_id == "R-05":
        endpoint = str(
            document.get("endpoint")
            or document.get("service")
            or document.get("base_url")
            or ""
        ).lower()
        deployment = str(document.get("deployment") or document.get("kind") or "").lower()
        if (
            "127.0.0.1" in endpoint
            or "localhost" in endpoint
            or deployment in {"loopback", "local-http", "local"}
        ) and document.get("separately_deployed") is not True:
            raise ValueError(
                f"refusing to treat local/loopback OpenEnv as R-05: {path}. "
                "Set separately_deployed=true only for a real operated service, "
                "or leave R-05 missing."
            )


def collect_doctor(out_dir: Path) -> dict[str, Any]:
    from arena.core.support import doctor_report

    report = doctor_report()
    path = out_dir / "doctor.json"
    digest = _write_json(path, report)
    preview_required = [
        item
        for item in report.get("capabilities", [])
        if item.get("required_for_1_0") and item.get("release_status") == "preview"
    ]
    return {
        "ok": bool(report.get("ok")),
        "path": str(path),
        "digest": digest,
        "schema_registry_digest": report.get("schema_registry_digest"),
        "platform_status": report.get("platform_status"),
        "preview_required_for_1_0": [item.get("name") for item in preview_required],
        "note": (
            "Doctor reports local dependency readiness only. It never authenticates "
            "and never promotes preview HF/OpenEnv/Gimitest to stable."
        ),
    }


def collect_schema_registry(out_dir: Path) -> dict[str, Any]:
    from arena.core.support import load_schema_registry

    registry = load_schema_registry()
    path = out_dir / "schema-registry.snapshot.json"
    digest = _write_json(path, registry)
    by_status: dict[str, int] = {}
    for item in registry.get("schemas", []):
        status = str(item.get("status") or "unknown")
        by_status[status] = by_status.get(status, 0) + 1
    return {
        "ok": True,
        "path": str(path),
        "digest": digest,
        "release": registry.get("release"),
        "schema_count": len(registry.get("schemas", [])),
        "by_status": by_status,
    }


def collect_golden_fixture_digests(repo: Path, out_dir: Path) -> dict[str, Any]:
    demo = repo / "examples" / "eval" / "demo"
    fixtures: list[dict[str, Any]] = []
    missing: list[str] = []
    for policy_dir in sorted(demo.glob("*.arena")):
        if not policy_dir.is_dir() or policy_dir.name.startswith("."):
            continue
        digest_path = policy_dir / "DIGEST"
        if not digest_path.is_file():
            missing.append(str(policy_dir.relative_to(repo)))
            continue
        declared = digest_path.read_text(encoding="utf-8").strip()
        fixtures.append(
            {
                "path": str(policy_dir.relative_to(repo)),
                "declared_digest": declared,
                "digest_file_digest": _digest_file(digest_path),
                "milestone_hint": "0.2-eval-demo",
            }
        )
    document = {
        "schema": "arena.golden-fixture-inventory/v1",
        "description": (
            "Checked-in historical demo policy digests. Authentic 0.3/0.5 archive "
            "fixtures still need separate release binding for R-09."
        ),
        "fixtures": fixtures,
        "missing": missing,
    }
    path = out_dir / "golden-fixture-digests.json"
    digest = _write_json(path, document)
    return {
        "ok": bool(fixtures) and not missing,
        "path": str(path),
        "digest": digest,
        "fixture_count": len(fixtures),
        "missing": missing,
        "note": "Local inventory only; not a complete R-09 authentic-fixture report.",
    }


def collect_perf_smoke_baselines(repo: Path, out_dir: Path) -> dict[str, Any]:
    baseline = repo / "tests" / "baselines" / "perf_smoke.json"
    if not baseline.is_file():
        return {
            "ok": False,
            "path": None,
            "error": f"missing perf-smoke baseline: {baseline}",
            "note": "Perf-smoke baselines are not the R-10 gate.",
        }
    document = json.loads(baseline.read_text(encoding="utf-8"))
    path = out_dir / "perf-smoke-baselines.json"
    payload = {
        "schema": "arena.perf-smoke-collector/v1",
        "source": str(baseline.relative_to(repo)),
        "source_digest": _digest_file(baseline),
        "baseline": document,
        "note": (
            "Copied checked-in perf-smoke baselines. This is a catastrophic-regression "
            "smoke check, not R-10 Linux/macOS release performance evidence."
        ),
    }
    digest = _write_json(path, payload)
    return {
        "ok": True,
        "path": str(path),
        "digest": digest,
        "source_digest": payload["source_digest"],
        "baselines_seconds": document.get("baselines_seconds"),
        "note": payload["note"],
    }


def collect_hermetic_capable(repo: Path, out_dir: Path) -> dict[str, Any]:
    """Inventory hermetic-capable tests without claiming the slow/Docker gate passed."""
    candidates = [
        repo / "tests" / "acceptance" / "test_u01_hermetic.py",
        repo / "tests" / "acceptance" / "test_eval_hermetic.py",
        repo / "tests" / "acceptance" / "test_u01_clean_room.py",
    ]
    present = [str(path.relative_to(repo)) for path in candidates if path.is_file()]
    missing = [str(path.relative_to(repo)) for path in candidates if not path.is_file()]
    document = {
        "schema": "arena.hermetic-capable-inventory/v1",
        "present": present,
        "missing": missing,
        "markers": ["slow", "docker"],
        "commands": [
            "pytest -m slow -q",
            "pytest -m docker -q",
        ],
        "ran_slow_hermetic": False,
        "note": (
            "Hermetic-capable inventory only. Exact release-wheel + network-none "
            "results must be attached for R-02; this collector does not invent them."
        ),
    }
    path = out_dir / "hermetic-capable.json"
    digest = _write_json(path, document)
    return {
        "ok": bool(present) and not missing,
        "path": str(path),
        "digest": digest,
        "present": present,
        "missing": missing,
        "ran_slow_hermetic": False,
        "note": document["note"],
    }


def collect_release_truth(repo: Path, out_dir: Path) -> dict[str, Any]:
    script = repo / "scripts" / "check_release_truth.py"
    if not script.is_file():
        return {"ok": False, "error": f"missing {script}"}
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    document = {
        "schema": "arena.release-truth-collector/v1",
        "command": [sys.executable, str(script.relative_to(repo))],
        "exit_code": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "ok": proc.returncode == 0,
    }
    path = out_dir / "release-truth.json"
    digest = _write_json(path, document)
    return {
        "ok": proc.returncode == 0,
        "path": str(path),
        "digest": digest,
        "exit_code": proc.returncode,
    }


def collect_adversarial_inventory(repo: Path, out_dir: Path) -> dict[str, Any]:
    adv_dir = repo / "tests" / "adversarial"
    files = sorted(
        str(path.relative_to(repo))
        for path in adv_dir.glob("test_*.py")
        if path.is_file()
    )
    document = {
        "schema": "arena.adversarial-inventory/v1",
        "tests": files,
        "count": len(files),
        "ran_full_soak": False,
        "note": (
            "Adversarial unit inventory only. Release-scale soak / resource envelope "
            "must be attached separately for R-07."
        ),
    }
    path = out_dir / "adversarial-inventory.json"
    digest = _write_json(path, document)
    return {
        "ok": bool(files),
        "path": str(path),
        "digest": digest,
        "count": len(files),
        "note": document["note"],
    }


def collect_supply_chain_scripts(repo: Path, out_dir: Path) -> dict[str, Any]:
    scripts = {
        "check_secret_scan": repo / "scripts" / "check_secret_scan.py",
        "check_pip_audit": repo / "scripts" / "check_pip_audit.py",
        "check_release_truth": repo / "scripts" / "check_release_truth.py",
    }
    present = {
        name: str(path.relative_to(repo))
        for name, path in scripts.items()
        if path.is_file()
    }
    missing = [name for name, path in scripts.items() if not path.is_file()]
    workflow = repo / ".github" / "workflows" / "release-candidate.yml"
    document = {
        "schema": "arena.supply-chain-inventory/v1",
        "scripts": present,
        "missing_scripts": missing,
        "release_candidate_workflow": (
            str(workflow.relative_to(repo)) if workflow.is_file() else None
        ),
        "workflow_digest": _digest_file(workflow) if workflow.is_file() else None,
        "ran_remote_attestations": False,
        "note": (
            "Local script/workflow inventory only. SBOM/provenance attestations from "
            "GitHub Actions are required for R-08 and are not invented here."
        ),
    }
    path = out_dir / "supply-chain-inventory.json"
    digest = _write_json(path, document)
    return {
        "ok": not missing and workflow.is_file(),
        "path": str(path),
        "digest": digest,
        "note": document["note"],
    }


def collect_recovery_inventory(repo: Path, out_dir: Path) -> dict[str, Any]:
    candidates = [
        repo / "tests" / "adversarial" / "test_process_budget_matrix.py",
        repo / "tests" / "adversarial" / "test_c6_failures.py",
        repo / "tests" / "adversarial" / "test_c8_store.py",
        repo / "tests" / "unit" / "test_external_mirror.py",
    ]
    present = [str(path.relative_to(repo)) for path in candidates if path.is_file()]
    document = {
        "schema": "arena.recovery-inventory/v1",
        "present": present,
        "rollback_rehearsal": False,
        "note": (
            "Local recovery-related test inventory only. Full fault matrix + rollback "
            "rehearsal must be attached for R-12."
        ),
    }
    path = out_dir / "recovery-inventory.json"
    digest = _write_json(path, document)
    return {
        "ok": bool(present),
        "path": str(path),
        "digest": digest,
        "note": document["note"],
    }


def _slot_status(
    *,
    gate_id: str,
    attached: Path | None,
    local_proof: dict[str, Any] | None,
) -> dict[str, Any]:
    spec = gate_by_id(gate_id)
    base = {
        "id": gate_id,
        "title": spec["title"],
        "owner": spec["owner"],
        "evidence_filename": spec["evidence_filename"],
        "how_to_attach": spec["how_to_attach"],
        "never_auto_pass": gate_id in NEVER_AUTO_PASS_GATES,
        "external_floor": gate_id in EXTERNAL_FLOOR_GATES,
    }
    if attached is not None:
        digest = _digest_file(attached)
        return {
            **base,
            "slot": "filled",
            "status": "attached",
            "claim_level": "attached-external",
            "evidence_path": str(attached),
            "evidence_digest": digest,
            "local_proof": local_proof,
            "note": (
                "External/attached evidence recorded with content digest. "
                "Final release still requires `arena release assemble` + verify."
            ),
        }
    if local_proof and local_proof.get("ok"):
        return {
            **base,
            "slot": "local-partial",
            "status": "local-partial",
            "claim_level": "local-proof",
            "evidence_path": local_proof.get("path"),
            "evidence_digest": local_proof.get("digest"),
            "local_proof": local_proof,
            "note": (
                "Local collector proof only — not a release-gate pass. "
                + str(local_proof.get("note") or "")
            ).strip(),
        }
    return {
        **base,
        "slot": "missing",
        "status": "missing",
        "claim_level": "external-required"
        if gate_id in NEVER_AUTO_PASS_GATES
        else "uncollected",
        "evidence_path": None,
        "evidence_digest": None,
        "local_proof": local_proof,
        "note": "No honest local or attached evidence for this gate yet.",
    }


def collect_release_evidence(
    *,
    repo: Path | None = None,
    out_dir: Path | None = None,
    attach: dict[str, Path] | None = None,
    run_doctor: bool = True,
) -> dict[str, Any]:
    repo_root = (repo or ROOT).resolve()
    evidence_root = (out_dir or (repo_root / "evidence")).resolve()
    local_dir = evidence_root / "local"
    local_dir.mkdir(parents=True, exist_ok=True)

    attached = dict(attach or {})
    for gate_id, path in attached.items():
        _reject_simulated_external_floor(gate_id, path)

    local_checks: dict[str, Any] = {
        "host": {
            "os": platform.system().lower(),
            "arch": platform.machine().lower(),
            "python": platform.python_version(),
        },
        "commit": _git_commit(repo_root),
        "collected_at": _utc_now(),
    }

    if run_doctor:
        local_checks["doctor"] = collect_doctor(local_dir)
    local_checks["schema_registry"] = collect_schema_registry(local_dir)
    local_checks["golden_fixture_digests"] = collect_golden_fixture_digests(
        repo_root, local_dir
    )
    local_checks["perf_smoke_baselines"] = collect_perf_smoke_baselines(
        repo_root, local_dir
    )
    local_checks["hermetic_capable"] = collect_hermetic_capable(repo_root, local_dir)
    local_checks["release_truth"] = collect_release_truth(repo_root, local_dir)
    local_checks["adversarial_inventory"] = collect_adversarial_inventory(
        repo_root, local_dir
    )
    local_checks["supply_chain_inventory"] = collect_supply_chain_scripts(
        repo_root, local_dir
    )
    local_checks["recovery_inventory"] = collect_recovery_inventory(
        repo_root, local_dir
    )

    local_by_gate: dict[str, dict[str, Any] | None] = {
        "R-01": None,
        "R-02": local_checks["hermetic_capable"],
        "R-03": None,
        "R-04": {
            "ok": False,
            "path": None,
            "digest": None,
            "note": (
                "Live Hugging Face evidence not attached. file:// local proof does "
                "not satisfy the R-04 stable-store floor."
            ),
        },
        "R-05": None,
        "R-06": None,
        "R-07": local_checks["adversarial_inventory"],
        "R-08": local_checks["supply_chain_inventory"],
        "R-09": {
            "ok": bool(local_checks["schema_registry"].get("ok"))
            and bool(local_checks["golden_fixture_digests"].get("ok")),
            "path": local_checks["golden_fixture_digests"].get("path"),
            "digest": local_checks["golden_fixture_digests"].get("digest"),
            "schema_registry": local_checks["schema_registry"],
            "golden_fixture_digests": local_checks["golden_fixture_digests"],
            "note": local_checks["golden_fixture_digests"].get("note"),
        },
        "R-10": local_checks["perf_smoke_baselines"],
        "R-11": None,
        "R-12": local_checks["recovery_inventory"],
        "R-13": {
            "ok": bool(local_checks["release_truth"].get("ok"))
            and bool((local_checks.get("doctor") or {}).get("schema_registry_digest")),
            "path": local_checks["release_truth"].get("path"),
            "digest": local_checks["release_truth"].get("digest"),
            "doctor": local_checks.get("doctor"),
            "release_truth": local_checks["release_truth"],
            "note": (
                "Local release-truth + doctor snapshot; final-tag recheck still required."
            ),
        },
        "R-14": None,
    }

    gates = [
        _slot_status(
            gate_id=gate_id,
            attached=attached.get(gate_id),
            local_proof=local_by_gate.get(gate_id),
        )
        for gate_id in MANDATORY_GATE_IDS
    ]

    filled = [g["id"] for g in gates if g["slot"] == "filled"]
    local_partial = [g["id"] for g in gates if g["slot"] == "local-partial"]
    missing = [g["id"] for g in gates if g["slot"] == "missing"]

    skeleton = {
        "schema": COLLECTOR_SCHEMA,
        "kind": "skeleton",
        "release_evidence_schema": "arena.release-evidence/v1",
        "note": (
            "Skeleton only. Not a signed arena.release-evidence/v1 index. "
            "Missing external-floor gates (HF/OpenEnv/Gimitest) are intentionally "
            "not marked passed."
        ),
        "commit": local_checks["commit"],
        "collected_at": local_checks["collected_at"],
        "host": local_checks["host"],
        "gates": gates,
        "local_checks": {
            key: value
            for key, value in local_checks.items()
            if key not in {"host", "commit", "collected_at"}
        },
        "summary": {
            "filled": filled,
            "local_partial": local_partial,
            "missing": missing,
            "never_auto_pass": sorted(NEVER_AUTO_PASS_GATES),
            "external_floor_missing": [
                gate_id
                for gate_id in sorted(EXTERNAL_FLOOR_GATES)
                if gate_id in missing
            ],
        },
        "next_steps": [
            "Attach CI / live HF / separate OpenEnv / isolated Gimitest evidence "
            "with --attach GATE=/path.json",
            "See docs/releasing.md and docs/qualifications/README.md",
            "When all R-01…R-14 files exist, run arena release assemble + sign + verify",
            "Do not tag v1.0.0 or publish to PyPI from this collector",
        ],
    }

    index_path = evidence_root / "release-index.json"
    if index_path.exists() or index_path.is_symlink():
        existing = json.loads(index_path.read_text(encoding="utf-8"))
        if existing.get("schema") == "arena.release-evidence/v1":
            raise RuntimeError(
                f"refusing to overwrite signed-ready release evidence: {index_path}"
            )
    skeleton_digest = _write_json(index_path, skeleton)
    skeleton["skeleton_path"] = str(index_path)
    skeleton["skeleton_digest"] = skeleton_digest
    _write_json(index_path, skeleton)
    return skeleton


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect local R-01…R-14 evidence and emit evidence/release-index.json "
            "skeleton. Never invents live HF/OpenEnv/Gimitest passes."
        )
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=None,
        help="Repository root (default: Arena checkout containing this script)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Evidence directory (default: <repo>/evidence)",
    )
    parser.add_argument(
        "--attach",
        action="append",
        default=[],
        help="Attach real gate evidence as GATE=/path/to/file.json (repeatable)",
    )
    parser.add_argument(
        "--skip-doctor",
        action="store_true",
        help="Skip arena doctor (still runs registry/golden/perf/truth checks)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the skeleton document to stdout",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = (args.repo or ROOT).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        attached = _parse_attach(args.attach)
        document = collect_release_evidence(
            repo=repo,
            out_dir=args.out,
            attach=attached,
            run_doctor=not args.skip_doctor,
        )
    except (OSError, ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"collect_release_evidence: {exc}", file=sys.stderr)
        return 2

    summary = document["summary"]
    print(
        "Release evidence skeleton: "
        f"filled={len(summary['filled'])} "
        f"local_partial={len(summary['local_partial'])} "
        f"missing={len(summary['missing'])}"
    )
    if summary["external_floor_missing"]:
        print(
            "External floor still missing (not invented): "
            + ", ".join(summary["external_floor_missing"])
        )
    print(f"Wrote {document['skeleton_path']}")
    if args.json:
        print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    if __package__ is None or __package__ == "":
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        __package__ = "r_gates"
    raise SystemExit(main())
