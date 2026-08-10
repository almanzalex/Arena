"""Compare evaluation reports/bundles for claim comparability.

Labs must not juxtapose metrics when suite identity, policy digests, or seed
protocols differ. This module extracts those bindings from reports, eval runs,
and release bundles, then fails loud on any mismatch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from arena.core.errors import CompatibilityError, SchemaError
from arena.core.identity import canonical_json, digest_uri, sha256_bytes
from arena.core.manifests import (
    EVAL_BUNDLE_SCHEMA,
    EVAL_REPORT_SCHEMA,
    EVAL_REPORT_V1_SCHEMA,
    EVAL_RUN_SCHEMA,
    EVAL_RUN_V1_SCHEMA,
    load_manifest,
)

SEED_PROTOCOL_SCHEMA = "arena.seed-protocol/v1"
CLAIM_BINDINGS_SCHEMA = "arena.eval-claim-bindings/v1"
COMPARE_RESULT_SCHEMA = "arena.eval-compare/v1"

_REPORT_SCHEMAS = {EVAL_REPORT_SCHEMA, EVAL_REPORT_V1_SCHEMA}
_RUN_SCHEMAS = {EVAL_RUN_SCHEMA, EVAL_RUN_V1_SCHEMA}


def load_eval_claim(path: Path | str) -> dict[str, Any]:
    """Load a report, eval-run, or bundle path into a comparable claim view."""
    root = Path(path)
    if not root.exists():
        raise SchemaError(f"eval claim path does not exist: {root}")

    report: dict[str, Any] | None = None
    eval_run: dict[str, Any] | None = None
    bundle: dict[str, Any] | None = None
    suite: dict[str, Any] | None = None
    source_kind: str

    if root.is_dir():
        bundle = _maybe_load(root / "bundle.json") or _maybe_load(root / "bundle.yaml")
        report = _maybe_load(root / "report.json") or _maybe_load(root / "report.yaml")
        eval_run = _maybe_load(root / "eval_run.json") or _maybe_load(root / "eval_run.yaml")
        suite = _maybe_load(root / "suite.yaml") or _maybe_load(root / "suite.json")
        if bundle is not None:
            source_kind = "bundle"
        elif eval_run is not None:
            source_kind = "eval_run_dir"
        elif report is not None:
            source_kind = "report_dir"
        else:
            raise SchemaError(
                f"directory is not an eval report, run, or bundle: {root} "
                "(expected report.json, eval_run.json, and/or bundle.json)"
            )
    else:
        data = load_manifest(root)
        schema = data.get("schema")
        if schema in _REPORT_SCHEMAS:
            report = data
            source_kind = "report"
            sibling = root.parent
            eval_run = _maybe_load(sibling / "eval_run.json") or _maybe_load(
                sibling / "eval_run.yaml"
            )
            suite = _maybe_load(sibling / "suite.yaml") or _maybe_load(sibling / "suite.json")
        elif schema in _RUN_SCHEMAS:
            eval_run = data
            source_kind = "eval_run"
            sibling = root.parent
            report = _maybe_load(sibling / "report.json") or _maybe_load(sibling / "report.yaml")
            suite = _maybe_load(sibling / "suite.yaml") or _maybe_load(sibling / "suite.json")
        elif schema == EVAL_BUNDLE_SCHEMA:
            bundle = data
            source_kind = "bundle_manifest"
            sibling = root.parent
            report = _maybe_load(sibling / "report.json") or _maybe_load(sibling / "report.yaml")
            eval_run = _maybe_load(sibling / "eval_run.json") or _maybe_load(
                sibling / "eval_run.yaml"
            )
            suite = _maybe_load(sibling / "suite.yaml") or _maybe_load(sibling / "suite.json")
        else:
            raise SchemaError(
                f"unsupported eval claim schema {schema!r} at {root}; "
                "expected eval-report, eval-run, or eval-bundle"
            )

    if bundle is not None and eval_run is None and "evaluation_digest" in bundle:
        # Bundle identity alone is usable for suite digest, but prefer run data.
        pass

    return {
        "path": str(root),
        "source_kind": source_kind,
        "report": report,
        "eval_run": eval_run,
        "bundle": bundle,
        "suite": suite,
    }


def extract_claim_bindings(claim: dict[str, Any]) -> dict[str, Any]:
    """Project suite digest, policy digests, and seed protocol for comparison."""
    report = claim.get("report") or {}
    eval_run = claim.get("eval_run") or {}
    bundle = claim.get("bundle") or {}
    suite = claim.get("suite") or {}

    suite_digest = (
        eval_run.get("evaluation_digest")
        or report.get("evaluation_digest")
        or bundle.get("evaluation_digest")
    )
    if not suite_digest:
        raise SchemaError(
            f"eval claim at {claim.get('path')!r} is missing evaluation_digest "
            "(suite digest); cannot compare incomparable or incomplete claims"
        )

    intent_digest = eval_run.get("evaluation_intent_digest") or report.get(
        "evaluation_intent_digest"
    )
    policy_digests = _policy_digests(eval_run=eval_run, report=report)
    seed_protocol = _seed_protocol(eval_run=eval_run, suite=suite)
    seed_protocol_digest = (
        digest_uri(sha256_bytes(canonical_json(seed_protocol))) if seed_protocol else None
    )

    return {
        "schema": CLAIM_BINDINGS_SCHEMA,
        "path": claim.get("path"),
        "source_kind": claim.get("source_kind"),
        "suite_digest": suite_digest,
        "evaluation_intent_digest": intent_digest,
        "policy_digests": policy_digests,
        "seed_protocol": seed_protocol,
        "seed_protocol_digest": seed_protocol_digest,
    }


def compare_eval_claims(
    left_path: Path | str,
    right_path: Path | str,
    *,
    raise_on_mismatch: bool = True,
) -> dict[str, Any]:
    """Compare two eval claims and optionally raise on incomparable bindings."""
    left_claim = load_eval_claim(left_path)
    right_claim = load_eval_claim(right_path)
    left = extract_claim_bindings(left_claim)
    right = extract_claim_bindings(right_claim)

    mismatches: list[dict[str, Any]] = []

    if left["suite_digest"] != right["suite_digest"]:
        mismatches.append(
            {
                "code": "SUITE_DIGEST_MISMATCH",
                "field": "suite_digest",
                "left": left["suite_digest"],
                "right": right["suite_digest"],
                "message": (
                    "evaluation suite digests differ; metrics are not comparable "
                    "across different locked suites"
                ),
            }
        )

    if left["policy_digests"] != right["policy_digests"]:
        mismatches.append(
            {
                "code": "POLICY_DIGEST_MISMATCH",
                "field": "policy_digests",
                "left": left["policy_digests"],
                "right": right["policy_digests"],
                "message": (
                    "policy digests differ; refusing to compare claims over different "
                    "policy sets"
                ),
            }
        )

    left_seed = left["seed_protocol_digest"]
    right_seed = right["seed_protocol_digest"]
    if left_seed != right_seed:
        mismatches.append(
            {
                "code": "SEED_PROTOCOL_MISMATCH",
                "field": "seed_protocol",
                "left": left_seed,
                "right": right_seed,
                "left_protocol": left.get("seed_protocol"),
                "right_protocol": right.get("seed_protocol"),
                "message": (
                    "seed protocols differ; seeded evaluation claims are not "
                    "comparable without identical seed schedules"
                ),
            }
        )

    # Intent digest is advisory when present on both sides; mismatch is loud.
    left_intent = left.get("evaluation_intent_digest")
    right_intent = right.get("evaluation_intent_digest")
    if left_intent and right_intent and left_intent != right_intent:
        mismatches.append(
            {
                "code": "EVALUATION_INTENT_MISMATCH",
                "field": "evaluation_intent_digest",
                "left": left_intent,
                "right": right_intent,
                "message": (
                    "evaluation intent digests differ; semantic suite projections "
                    "are not comparable"
                ),
            }
        )

    comparable = not mismatches
    result = {
        "schema": COMPARE_RESULT_SCHEMA,
        "ok": comparable,
        "comparable": comparable,
        "left": {
            "path": left["path"],
            "source_kind": left["source_kind"],
            "suite_digest": left["suite_digest"],
            "evaluation_intent_digest": left.get("evaluation_intent_digest"),
            "policy_digests": left["policy_digests"],
            "seed_protocol_digest": left_seed,
        },
        "right": {
            "path": right["path"],
            "source_kind": right["source_kind"],
            "suite_digest": right["suite_digest"],
            "evaluation_intent_digest": right.get("evaluation_intent_digest"),
            "policy_digests": right["policy_digests"],
            "seed_protocol_digest": right_seed,
        },
        "mismatches": mismatches,
    }

    if mismatches and raise_on_mismatch:
        codes = ", ".join(item["code"] for item in mismatches)
        details = "; ".join(item["message"] for item in mismatches)
        raise CompatibilityError(
            f"incomparable evaluation claims ({codes}): {details}",
            code="EVAL_CLAIMS_INCOMPARABLE",
            cause=codes,
            repair=(
                "Compare only reports/bundles that share the same evaluation_digest, "
                "policy digest set, and seed protocol. Re-run under one locked suite "
                "or pick matched evidence."
            ),
            context=result,
        )
    return result


def _maybe_load(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = load_manifest(path)
    if not isinstance(data, dict):
        raise SchemaError(f"expected mapping in {path}")
    return data


def _policy_digests(
    *,
    eval_run: dict[str, Any],
    report: dict[str, Any],
) -> list[str]:
    digests: set[str] = set()
    for cell in eval_run.get("cells") or []:
        lineage = cell.get("lineage") or {}
        for digest in lineage.get("policy_digests") or []:
            digests.add(str(digest))
        assignments = cell.get("assignments") or {}
        for digest in assignments.values():
            digests.add(str(digest))
        for key in ("candidate_policy", "opponent_policy"):
            value = cell.get(key)
            if value:
                digests.add(str(value))

    metrics = report.get("metrics") or {}
    payoff = metrics.get("payoff_matrix") or {}
    for key in ("rows", "cols"):
        for digest in payoff.get(key) or []:
            digests.add(str(digest))

    return sorted(digests)


def _seed_protocol(
    *,
    eval_run: dict[str, Any],
    suite: dict[str, Any],
) -> dict[str, Any] | None:
    cells = eval_run.get("cells") or []
    ledger = eval_run.get("sampling_ledger") or []
    suite_seeds = suite.get("seeds")
    action_mode = suite.get("action_mode") or eval_run.get("action_mode")

    cell_seeds = [
        {
            "cell_id": cell.get("cell_id"),
            "seeds": list(cell.get("seeds") or []),
        }
        for cell in sorted(cells, key=lambda item: str(item.get("cell_id") or ""))
    ]
    sampling = [
        {
            "role": entry.get("role"),
            "sampler": entry.get("sampler"),
            "seed": entry.get("seed"),
            "stream": entry.get("stream"),
            "index": entry.get("index"),
        }
        for entry in ledger
    ]
    sampling.sort(
        key=lambda item: (
            str(item.get("role") or ""),
            str(item.get("stream") or ""),
            int(item.get("index") or 0),
        )
    )

    if not cell_seeds and suite_seeds is None and not sampling:
        return None

    protocol: dict[str, Any] = {
        "schema": SEED_PROTOCOL_SCHEMA,
        "cells": cell_seeds,
        "sampling": sampling,
    }
    if suite_seeds is not None:
        protocol["suite_seeds"] = suite_seeds
    if action_mode is not None:
        protocol["action_mode"] = action_mode
    return protocol
