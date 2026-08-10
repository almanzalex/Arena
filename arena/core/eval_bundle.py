"""Releaseable evaluation bundles locking suite/population/trajectory digests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from arena.core.errors import IntegrityError, SchemaError
from arena.core.identity import canonical_json, digest_uri, sha256_bytes, sha256_file
from arena.core.io import publish_directory
from arena.core.manifests import (
    EVAL_BUNDLE_SCHEMA,
    dump_json,
    dump_yaml,
    load_manifest,
    validate_eval_bundle_manifest,
)


def build_eval_bundle(
    *,
    eval_run_dir: Path | str,
    report: dict[str, Any] | None = None,
    out_dir: Path | str,
    evaluation_digest: str | None = None,
    extra_artifacts: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Copy locked evidence into an atomically replaceable bundle directory."""
    final = Path(out_dir)

    def build(stage: Path) -> dict[str, Any]:
        return _build_eval_bundle_into(
            eval_run_dir=eval_run_dir,
            report=report,
            out_dir=stage,
            evaluation_digest=evaluation_digest,
            extra_artifacts=extra_artifacts,
        )

    return publish_directory(final, build, replace=True, verify=verify_eval_bundle)


def _load_eval_run_record(eval_run_dir: Path) -> dict[str, Any] | None:
    for name in ("eval_run.json", "eval_run.yaml"):
        src = eval_run_dir / name
        if src.exists():
            return load_manifest(src)
    return None


def _refuse_incomplete_bundle(
    *,
    eval_run: dict[str, Any] | None,
    report: dict[str, Any] | None,
) -> None:
    """Incomplete evidence must never publish as a finished eval bundle."""
    if eval_run is not None:
        state = eval_run.get("state", "complete")
        if state != "complete":
            raise SchemaError(
                f"refusing to bundle incomplete evaluation: state={state}; "
                "only complete eval-run records may be published as finished evidence"
            )
        run_digest = eval_run.get("evaluation_digest")
        if report is not None and report.get("evaluation_digest") not in (None, run_digest):
            raise SchemaError(
                "refusing to bundle report whose evaluation_digest does not match "
                f"eval-run suite identity: report={report.get('evaluation_digest')!r} "
                f"eval-run={run_digest!r}"
            )
        if report is not None and report.get("state") not in (None, "complete"):
            raise SchemaError(
                f"refusing to bundle incomplete report: state={report.get('state')}"
            )
    elif report is not None and report.get("state") not in (None, "complete"):
        raise SchemaError(
            f"refusing to bundle incomplete report: state={report.get('state')}"
        )


def _build_eval_bundle_into(
    *,
    eval_run_dir: Path | str,
    report: dict[str, Any] | None,
    out_dir: Path,
    evaluation_digest: str | None,
    extra_artifacts: dict[str, str] | None,
) -> dict[str, Any]:
    """Build a complete bundle inside a private staging directory."""
    eval_run_dir = Path(eval_run_dir)
    eval_run = _load_eval_run_record(eval_run_dir)
    _refuse_incomplete_bundle(eval_run=eval_run, report=report)

    artifacts: dict[str, str] = {}
    # Lock eval_run record
    for name in ("eval_run.json", "eval_run.yaml"):
        src = eval_run_dir / name
        if src.exists():
            dest = out_dir / name
            shutil.copy2(src, dest)
            artifacts[name] = digest_uri(sha256_file(dest))
            if evaluation_digest is None and eval_run is not None:
                evaluation_digest = eval_run.get("evaluation_digest")
            break

    # Lock cell trajectories
    traj_root = out_dir / "trajectories"
    traj_root.mkdir()
    for cell_dir in sorted(p for p in eval_run_dir.iterdir() if p.is_dir()):
        src_traj = cell_dir / "trajectories"
        if not src_traj.is_dir():
            continue
        dest_cell = traj_root / cell_dir.name
        shutil.copytree(src_traj, dest_cell)
        for f in sorted(dest_cell.rglob("*")):
            if f.is_file():
                rel = str(f.relative_to(out_dir))
                artifacts[rel] = digest_uri(sha256_file(f))

    if report is not None:
        dump_json(report, out_dir / "report.json")
        dump_yaml(report, out_dir / "report.yaml")
        artifacts["report.json"] = digest_uri(sha256_file(out_dir / "report.json"))

    for key, digest in (extra_artifacts or {}).items():
        artifacts[key] = digest

    if not evaluation_digest:
        raise SchemaError("evaluation_digest missing; cannot build eval bundle")

    bundle = {
        "schema": EVAL_BUNDLE_SCHEMA,
        "evaluation_digest": evaluation_digest,
        "artifacts": artifacts,
        "reproduce": {
            "mode": "reaggregate_from_locked_rollouts",
            "note": "Recompute metrics from locked trajectories; do not require trainer repos.",
        },
    }
    validate_eval_bundle_manifest(bundle)
    bundle["digest"] = digest_uri(sha256_bytes(canonical_json(bundle)))
    dump_yaml(bundle, out_dir / "bundle.yaml")
    dump_json(bundle, out_dir / "bundle.json")
    return bundle


def verify_eval_bundle(bundle_dir: Path | str) -> dict[str, Any]:
    """Rehash locked artifacts and fail loudly on tamper or missing files.

    Returns a machine-readable verification record. Callers must treat any raised
    :class:`~arena.core.errors.IntegrityError` / :class:`~arena.core.errors.SchemaError`
    as a hard reject — never a soft warning.
    """
    root = Path(bundle_dir)
    manifest_path = root / "bundle.json"
    if not manifest_path.exists():
        manifest_path = root / "bundle.yaml"
    if not manifest_path.exists():
        raise SchemaError(f"eval bundle missing bundle.json/yaml: {root}")
    locked = load_manifest(manifest_path)
    validate_eval_bundle_manifest(locked)
    artifacts = locked.get("artifacts") or {}
    if not isinstance(artifacts, dict) or not artifacts:
        raise SchemaError("eval bundle artifacts map is empty")

    checked: list[str] = []
    for rel, expected in sorted(artifacts.items()):
        path = root / rel
        if not path.is_file():
            raise IntegrityError(
                f"eval bundle missing locked artifact: {rel}",
                code="EVAL_BUNDLE_MISSING_ARTIFACT",
                repair="Restore the complete bundle or re-run arena eval bundle from the source eval-run.",
                context={"path": rel, "expected": expected},
            )
        actual = digest_uri(sha256_file(path))
        if actual != expected:
            raise IntegrityError(
                f"eval bundle artifact integrity check failed: {rel}",
                code="EVAL_BUNDLE_TAMPERED",
                repair="Do not trust this bundle; re-pull or rebuild from known-good sources.",
                context={"path": rel, "expected": expected, "actual": actual},
            )
        checked.append(rel)

    # Manifest self-digest (excluding the digest field itself).
    body = {key: value for key, value in locked.items() if key != "digest"}
    expected_digest = locked.get("digest")
    actual_digest = digest_uri(sha256_bytes(canonical_json(body)))
    if expected_digest is not None and actual_digest != expected_digest:
        raise IntegrityError(
            "eval bundle manifest digest mismatch",
            code="EVAL_BUNDLE_MANIFEST_TAMPERED",
            repair="Do not trust this bundle; the locked index itself was modified.",
            context={"expected": expected_digest, "actual": actual_digest},
        )

    return {
        "ok": True,
        "kind": "eval-bundle",
        "evaluation_digest": locked.get("evaluation_digest"),
        "digest": expected_digest or actual_digest,
        "artifacts_checked": checked,
        "artifact_count": len(checked),
    }
