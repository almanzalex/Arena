"""Comparability gates for eval reports and bundles."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arena.cli.main import main
from arena.core.errors import CompatibilityError
from arena.core.eval_compare import compare_eval_claims, extract_claim_bindings, load_eval_claim
from arena.core.identity import canonical_json, digest_uri, sha256_bytes

POLICY_A = "sha256:" + ("a" * 64)
POLICY_B = "sha256:" + ("b" * 64)
SUITE = "sha256:" + ("1" * 64)
INTENT = "sha256:" + ("2" * 64)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _claim_bundle(
    root: Path,
    *,
    suite_digest: str = SUITE,
    policy_digests: list[str] | None = None,
    seeds: list[int] | None = None,
) -> Path:
    policies = policy_digests or [POLICY_A, POLICY_B]
    seed_list = seeds if seeds is not None else [0, 1]
    cells = []
    for index, policy in enumerate(policies):
        cells.append(
            {
                "cell_id": f"cell-{index}",
                "assignments": {"player_0": policies[0], "player_1": policy},
                "candidate_policy": policies[0],
                "opponent_policy": policy,
                "seeds": list(seed_list),
                "lineage": {"policy_digests": sorted({policies[0], policy})},
                "sampling": {
                    "player_0": {
                        "index": 0,
                        "policy": policies[0],
                        "sampler": "enumerated_crossplay",
                        "seed": 0,
                        "stream": "eval:player_0:0",
                    },
                    "player_1": {
                        "index": index,
                        "policy": policy,
                        "sampler": "enumerated_crossplay",
                        "seed": 0,
                        "stream": f"eval:player_1:{index}",
                    },
                },
                "failures": 0,
                "evidence_refs": [],
            }
        )
    eval_run = {
        "schema": "arena.eval-run/v1",
        "run_id": "eval-test",
        "evaluation_digest": suite_digest,
        "evaluation_intent_digest": INTENT,
        "state": "complete",
        "denominators": {"attempted": len(cells), "completed": len(cells), "failed": 0},
        "sampling_ledger": [
            {
                "role": "player_0",
                "sampler": "enumerated_crossplay",
                "seed": 0,
                "stream": "eval:player_0:0",
                "index": 0,
                "policy": policies[0],
            },
            *[
                {
                    "role": "player_1",
                    "sampler": "enumerated_crossplay",
                    "seed": 0,
                    "stream": f"eval:player_1:{index}",
                    "index": index,
                    "policy": policy,
                }
                for index, policy in enumerate(policies)
            ],
        ],
        "cells": cells,
    }
    report = {
        "schema": "arena.eval-report/v1",
        "evaluation_digest": suite_digest,
        "evaluation_intent_digest": INTENT,
        "eval_run_digest": digest_uri(sha256_bytes(canonical_json(cells))),
        "state": "complete",
        "denominators": eval_run["denominators"],
        "metrics": {
            "payoff_matrix": {
                "kind": "payoff_matrix",
                "rows": list(policies),
                "cols": list(policies),
                "matrix": [[0.0 for _ in policies] for _ in policies],
            }
        },
    }
    bundle = {
        "schema": "arena.eval-bundle/v0alpha1",
        "evaluation_digest": suite_digest,
        "artifacts": {
            "eval_run.json": digest_uri(sha256_bytes(canonical_json(eval_run))),
            "report.json": digest_uri(sha256_bytes(canonical_json(report))),
        },
        "reproduce": {"mode": "reaggregate_from_locked_rollouts"},
    }
    _write_json(root / "eval_run.json", eval_run)
    _write_json(root / "report.json", report)
    _write_json(root / "bundle.json", bundle)
    _write_json(
        root / "suite.json",
        {
            "schema": "arena.evaluation/v0alpha1",
            "name": "compare-fixture",
            "seeds": {"start": seed_list[0], "count": len(seed_list)},
            "action_mode": "deterministic",
        },
    )
    return root


def test_compare_equal_bundles(tmp_path: Path) -> None:
    left = _claim_bundle(tmp_path / "left")
    right = _claim_bundle(tmp_path / "right")
    result = compare_eval_claims(left, right)
    assert result["ok"] is True
    assert result["comparable"] is True
    assert result["mismatches"] == []
    assert result["left"]["suite_digest"] == result["right"]["suite_digest"]
    assert result["left"]["policy_digests"] == result["right"]["policy_digests"]
    assert result["left"]["seed_protocol_digest"] == result["right"]["seed_protocol_digest"]
    assert result["left"]["policy_digests"] == sorted([POLICY_A, POLICY_B])


def test_compare_policy_digest_mismatch(tmp_path: Path) -> None:
    left = _claim_bundle(tmp_path / "left")
    right = _claim_bundle(
        tmp_path / "right",
        policy_digests=[POLICY_A, "sha256:" + ("d" * 64)],
    )
    with pytest.raises(CompatibilityError, match="POLICY_DIGEST_MISMATCH") as excinfo:
        compare_eval_claims(left, right)
    assert excinfo.value.code == "EVAL_CLAIMS_INCOMPARABLE"
    assert any(
        item["code"] == "POLICY_DIGEST_MISMATCH" for item in excinfo.value.context["mismatches"]
    )


def test_compare_suite_digest_mismatch(tmp_path: Path) -> None:
    left = _claim_bundle(tmp_path / "left")
    right = _claim_bundle(tmp_path / "right", suite_digest="sha256:" + ("9" * 64))
    with pytest.raises(CompatibilityError, match="SUITE_DIGEST_MISMATCH") as excinfo:
        compare_eval_claims(left, right)
    assert any(
        item["code"] == "SUITE_DIGEST_MISMATCH"
        for item in (excinfo.value.context or {})["mismatches"]
    )


def test_compare_seed_protocol_mismatch(tmp_path: Path) -> None:
    left = _claim_bundle(tmp_path / "left")
    right = _claim_bundle(tmp_path / "right", seeds=[0, 99])
    with pytest.raises(CompatibilityError, match="SEED_PROTOCOL_MISMATCH"):
        compare_eval_claims(left, right)


def test_cli_eval_compare_equal_and_mismatch(tmp_path: Path, capsys) -> None:
    left = _claim_bundle(tmp_path / "left")
    right = _claim_bundle(tmp_path / "right")
    assert main(["eval", "compare", str(left), str(right), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True
    assert out["data"]["comparable"] is True

    mismatched = _claim_bundle(tmp_path / "mismatch", suite_digest="sha256:" + ("c" * 64))
    code = main(["eval", "compare", str(left), str(mismatched), "--json"])
    assert code == 3
    err = json.loads(capsys.readouterr().out)
    assert err["ok"] is False
    assert err["code"] == "EVAL_CLAIMS_INCOMPARABLE"


def test_bindings_from_report_include_policy_digests(tmp_path: Path) -> None:
    bundle = _claim_bundle(tmp_path / "bundle")
    claim = load_eval_claim(bundle / "report.json")
    bindings = extract_claim_bindings(claim)
    assert bindings["suite_digest"] == SUITE
    assert bindings["policy_digests"] == sorted([POLICY_A, POLICY_B])
    assert bindings["seed_protocol_digest"]  # sibling eval_run.json supplies seeds
