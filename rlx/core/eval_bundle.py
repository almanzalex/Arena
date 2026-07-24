"""Releaseable evaluation bundles locking suite/population/trajectory digests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from rlx.core.errors import SchemaError
from rlx.core.identity import canonical_json, digest_uri, sha256_bytes, sha256_file
from rlx.core.io import publish_directory
from rlx.core.manifests import (
    EVAL_BUNDLE_SCHEMA,
    dump_json,
    dump_yaml,
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

    return publish_directory(final, build, replace=True)


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

    artifacts: dict[str, str] = {}
    # Lock eval_run record
    for name in ("eval_run.json", "eval_run.yaml"):
        src = eval_run_dir / name
        if src.exists():
            dest = out_dir / name
            shutil.copy2(src, dest)
            artifacts[name] = digest_uri(sha256_file(dest))
            if evaluation_digest is None and name.endswith(".json"):
                data = json.loads(src.read_text(encoding="utf-8"))
                evaluation_digest = data.get("evaluation_digest")
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
