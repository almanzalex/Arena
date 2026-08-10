from __future__ import annotations

import json
import os
import sys
import time
from importlib.metadata import version
from pathlib import Path

import pytest

from arena.core.attestation import generate_signing_keypair
from arena.core.errors import (
    ConformanceError,
    ExternalUnavailableError,
    IncompleteExecutionError,
    SchemaError,
)
from arena.core.identity import canonical_json, parse_digest
from arena.core.manifests import (
    evaluation_binding_digest,
    evaluation_intent_digest,
    load_manifest,
)
from arena.core.release import (
    assemble_release_evidence,
    sign_qualification_ledger,
    sign_release_evidence,
    verify_release_evidence,
)
from arena.core.spaces import gymnasium_space_to_dict
from arena.core.supervisor import run_supervised
from arena.core.support import (
    _probe_isolated_python,
    doctor_report,
    load_schema_registry,
)


def test_strict_identity_and_manifest_parser_reject_ambiguity(tmp_path: Path) -> None:
    with pytest.raises(SchemaError, match="64 lowercase"):
        parse_digest("sha256:ABC")
    with pytest.raises(SchemaError, match="finite"):
        canonical_json({"value": float("nan")})

    duplicate_json = tmp_path / "duplicate.json"
    duplicate_json.write_text('{"a": 1, "a": 2}', encoding="utf-8")
    with pytest.raises(SchemaError, match="duplicate"):
        load_manifest(duplicate_json)

    alias_yaml = tmp_path / "alias.yaml"
    alias_yaml.write_text("base: &base {a: 1}\ncopy: *base\n", encoding="utf-8")
    with pytest.raises(SchemaError, match="aliases"):
        load_manifest(alias_yaml)

    numeric_key = tmp_path / "numeric-key.yaml"
    numeric_key.write_text("1: value\n", encoding="utf-8")
    with pytest.raises(SchemaError, match="keys must be strings"):
        load_manifest(numeric_key)

    too_large = tmp_path / "large.json"
    too_large.write_text('{"x":"' + ("a" * 128) + '"}', encoding="utf-8")
    with pytest.raises(SchemaError, match="byte limit"):
        load_manifest(too_large, max_bytes=32)


def test_unbounded_box_uses_explicit_json_safe_sentinels() -> None:
    gymnasium = pytest.importorskip("gymnasium")
    import numpy as np

    descriptor = gymnasium_space_to_dict(
        gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2,),
            dtype=np.float32,
        )
    )
    assert descriptor["low"] == "-inf"
    assert descriptor["high"] == "inf"
    assert b'"-inf"' in canonical_json(descriptor)


def test_evaluation_intent_excludes_operational_binding() -> None:
    base = {
        "schema": "arena.evaluation/v0alpha1",
        "task": {
            "adapter": "openenv",
            "env": "openenv://arena/competitive_rps_v0",
            "interaction": "parallel",
            "packaging": {
                "kind": "openenv",
                "base_url": "https://one.invalid",
                "connect_timeout_seconds": 1,
            },
        },
        "task_intent_digest": "sha256:" + ("1" * 64),
        "assignments": {
            "player_0": "sha256:" + ("2" * 64),
            "player_1": "sha256:" + ("3" * 64),
        },
        "seeds": [0, 1],
        "action_mode": "deterministic",
        "metrics": ["mean_return"],
        "provider": "gimitest",
        "provider_config": {
            "semantic": {},
            "test_class": "gimitest.gtest:GTest",
            "isolation": {
                "mode": "subprocess",
                "python": "/env-a/bin/python",
                "timeout_seconds": 30,
            },
        },
        "failure_policy": {
            "missingness": "fail",
            "max_failed_episodes": 0,
            "timeout_seconds": 60,
        },
    }
    moved = json.loads(json.dumps(base))
    moved["task"]["packaging"]["base_url"] = "https://two.invalid"
    moved["provider_config"]["isolation"]["python"] = "/env-b/bin/python"
    moved["provider_config"]["isolation"]["timeout_seconds"] = 90
    moved["failure_policy"]["timeout_seconds"] = 120

    assert evaluation_intent_digest(base) == evaluation_intent_digest(moved)
    assert evaluation_binding_digest(base, workers=1) != evaluation_binding_digest(
        moved, workers=8
    )

    changed = json.loads(json.dumps(base))
    changed["seeds"] = [0, 2]
    assert evaluation_intent_digest(base) != evaluation_intent_digest(changed)


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_hard_budget_eval_process_is_complete_and_schedule_stable(
    tmp_path: Path,
) -> None:
    from arena.conformance.fixtures import build_fixed_action_rps_policy
    from arena.runtime.evaluation import run_evaluation

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
    suite = {
        "schema": "arena.evaluation/v0alpha1",
        "name": "hard-budget",
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
        "seeds": [0, 1],
        "action_mode": "deterministic",
        "metrics": ["mean_return"],
        "budgets": {"executor": "process", "timeout_seconds": 30},
    }
    result = run_evaluation(
        suite,
        policy_index={},
        out_dir=tmp_path / "run",
    )
    assert result["state"] == "complete"
    assert result["denominators"] == {
        "attempted": 2,
        "completed": 2,
        "failed": 0,
    }
    assert (tmp_path / "run" / "eval_run.json").is_file()


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_hard_budget_timeout_publishes_failed_ledger_not_fake_success(
    tmp_path: Path,
) -> None:
    from arena.conformance.fixtures import build_fixed_action_rps_policy
    from arena.runtime.evaluation import build_eval_report, run_evaluation

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
    result = run_evaluation(
        suite,
        policy_index={},
        out_dir=tmp_path / "failed-run",
    )
    assert result["state"] == "failed"
    assert result["denominators"] == {
        "attempted": 1,
        "completed": 0,
        "failed": 1,
    }
    with pytest.raises(IncompleteExecutionError, match="incomplete evaluation"):
        build_eval_report(result)


def test_schema_registry_is_unique_and_records_legacy_freeze() -> None:
    registry = load_schema_registry()
    ids = [item["id"] for item in registry["schemas"]]
    assert len(ids) == len(set(ids))
    legacy = {item["id"]: item for item in registry["schemas"]}
    assert legacy["arena.policy/v0alpha1"]["status"] == "legacy-frozen"
    assert legacy["arena.eval-run/v1"]["status"] == "stable"


def test_doctor_does_not_authenticate(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "canary-secret")
    report = doctor_report("hf")
    capability = report["capabilities"][0]
    assert capability["authentication_attempted"] is False
    assert "canary-secret" not in json.dumps(report)


def test_doctor_probes_isolated_gimitest_interpreter() -> None:
    probe = _probe_isolated_python(
        Path(sys.executable),
        distributions=["arena", "gimitest", "torch", "pettingzoo"],
        release=version("arena"),
    )
    assert probe["status"] == "ready"
    assert probe["versions"]["gimitest"] == "1.0"
    mismatch = _probe_isolated_python(
        Path(sys.executable),
        distributions=["arena"],
        release="0.0-does-not-match",
    )
    assert mismatch["status"] == "incompatible"


def test_doctor_rejects_non_executable_isolated_worker(
    monkeypatch, tmp_path: Path
) -> None:
    fake = tmp_path / "python"
    fake.write_text("not an interpreter", encoding="utf-8")
    monkeypatch.setenv("ARENA_GIMITEST_PYTHON", str(fake))
    capability = doctor_report("gimitest")["capabilities"][0]
    assert capability["local_status"] == "locally-unqualified"
    assert capability["isolated_probe"]["status"] == "unavailable"


def test_cli_json_grammar_help_and_secret_redaction(capsys) -> None:
    from arena.cli.main import main

    assert main(["doctor", "--capability", "core", "--json"]) == 0
    success = json.loads(capsys.readouterr().out)
    assert success["schema"] == "arena.cli-result/v1"
    assert success["command"] == "doctor"

    assert main(["eval", "run", "--help", "--json"]) == 0
    help_result = json.loads(capsys.readouterr().out)
    assert "usage:" in help_result["data"]["help"]

    secret = "canary-do-not-print"
    code = main(
        [
            "pull",
            f"hf://datasets/lab/repo?token={secret}#not-a-digest",
            "--json",
        ]
    )
    output = capsys.readouterr().out
    assert code == 5
    assert secret not in output
    failure = json.loads(output)
    assert failure["schema"] == "arena.diagnostic/v1"
    assert failure["command"] == "pull"


def test_supervisor_kills_process_group_on_timeout(tmp_path: Path) -> None:
    marker = tmp_path / "grandchild-survived"
    child = (
        "import pathlib,time,sys;"
        "time.sleep(0.7);"
        "pathlib.Path(sys.argv[1]).write_text('bad')"
    )
    parent = (
        "import subprocess,sys,time;"
        f"subprocess.Popen([sys.executable,'-c',{child!r},sys.argv[1]]);"
        "time.sleep(30)"
    )
    with pytest.raises(ExternalUnavailableError, match="wall time"):
        run_supervised(
            [sys.executable, "-c", parent, str(marker)],
            timeout_seconds=0.15,
        )
    time.sleep(0.8)
    assert not marker.exists()


def test_supervisor_enforces_output_budget() -> None:
    with pytest.raises(ExternalUnavailableError, match="stdout bytes"):
        run_supervised(
            [sys.executable, "-c", "import os; os.write(1, b'x' * 2000000)"],
            timeout_seconds=5,
            max_stdout_bytes=1024,
        )


def test_signed_release_and_current_ledger_are_content_bound(tmp_path: Path) -> None:
    pytest.importorskip("cryptography")
    artifact = tmp_path / "arena-1.0.0-py3-none-any.whl"
    artifact.write_bytes(b"release artifact")
    gate_files = {}
    for index in range(1, 15):
        gate = tmp_path / f"R-{index:02d}.json"
        gate.write_text(json.dumps({"ok": True, "gate": index}), encoding="utf-8")
        gate_files[f"R-{index:02d}"] = gate
    evidence_path = tmp_path / "evidence.json"
    assemble_release_evidence(
        release="1.0.0",
        tag="v1.0.0",
        commit="a" * 40,
        gates=gate_files,
        artifacts=[artifact],
        out=evidence_path,
    )
    assert "eval_bundles" not in json.loads(evidence_path.read_text(encoding="utf-8"))
    keys = generate_signing_keypair(
        private_key=tmp_path / "private.pem",
        public_key=tmp_path / "public.pem",
    )
    signature = tmp_path / "evidence.sig.json"
    sign_release_evidence(
        evidence_path,
        private_key=keys["private_key"],
        out=signature,
    )
    verified = verify_release_evidence(
        evidence_path,
        signature=signature,
        public_key=keys["public_key"],
    )
    assert verified["ok"] is True
    assert verified["local_artifacts_checked"] == [str(artifact)]
    assert verified["local_eval_bundles_checked"] == []
    assert verified["local_gate_evidence_checked"] == [
        str(gate_files[f"R-{index:02d}"]) for index in range(1, 15)
    ]

    ledger = {
        "schema": "arena.qualification-ledger/v1",
        "release": "1.0.0",
        "records": [{"capability": "hf", "status": "pass"}],
    }
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_bytes(canonical_json(ledger) + b"\n")
    ledger_signature = tmp_path / "ledger.sig.json"
    sign_qualification_ledger(
        ledger_path,
        private_key=keys["private_key"],
        out=ledger_signature,
    )
    current = verify_release_evidence(
        evidence_path,
        signature=signature,
        public_key=keys["public_key"],
        current_ledger=ledger_path,
        current_ledger_signature=ledger_signature,
        current_ledger_key=keys["public_key"],
    )
    assert current["mode"] == "current"

    tampered = {**ledger, "records": [{"capability": "hf", "status": "failed"}]}
    ledger_path.write_bytes(canonical_json(tampered) + b"\n")
    with pytest.raises(ConformanceError, match="subject digest"):
        verify_release_evidence(
            evidence_path,
            signature=signature,
            public_key=keys["public_key"],
            current_ledger=ledger_path,
            current_ledger_signature=ledger_signature,
            current_ledger_key=keys["public_key"],
        )

    ledger_path.write_bytes(canonical_json(ledger) + b"\n")
    gate_files["R-08"].write_text('{"ok": false}', encoding="utf-8")
    with pytest.raises(ConformanceError, match="gate evidence digest mismatch"):
        verify_release_evidence(
            evidence_path,
            signature=signature,
            public_key=keys["public_key"],
        )



def test_release_evidence_binds_eval_bundle_digests_when_present(tmp_path: Path) -> None:
    pytest.importorskip("cryptography")
    from arena.core.identity import digest_uri, sha256_bytes
    from arena.core.manifests import dump_json

    artifact = tmp_path / "arena-1.0.0-py3-none-any.whl"
    artifact.write_bytes(b"release artifact")
    gate_files = {}
    for index in range(1, 15):
        gate = tmp_path / f"R-{index:02d}.json"
        gate.write_text(json.dumps({"ok": True, "gate": index}), encoding="utf-8")
        gate_files[f"R-{index:02d}"] = gate

    evaluation_digest = digest_uri(sha256_bytes(b"locked-evaluation"))
    bundle_dir = tmp_path / "eval-bundle"
    bundle_dir.mkdir()
    bundle_manifest = {
        "schema": "arena.eval-bundle/v0alpha1",
        "evaluation_digest": evaluation_digest,
        "artifacts": {"eval_run.json": digest_uri(sha256_bytes(b"eval-run"))},
        "reproduce": {
            "mode": "reaggregate_from_locked_rollouts",
            "note": "test fixture",
        },
    }
    bundle_manifest["digest"] = digest_uri(sha256_bytes(canonical_json(bundle_manifest)))
    dump_json(bundle_manifest, bundle_dir / "bundle.json")

    evidence_path = tmp_path / "evidence.json"
    document = assemble_release_evidence(
        release="1.0.0",
        tag="v1.0.0",
        commit="b" * 40,
        gates=gate_files,
        artifacts=[artifact],
        out=evidence_path,
        eval_bundles=[bundle_dir],
    )
    assert len(document["eval_bundles"]) == 1
    assert document["eval_bundles"][0]["evaluation_digest"] == evaluation_digest
    assert document["eval_bundles"][0]["path"].endswith("bundle.json")

    keys = generate_signing_keypair(
        private_key=tmp_path / "private.pem",
        public_key=tmp_path / "public.pem",
    )
    signature = tmp_path / "evidence.sig.json"
    sign_release_evidence(
        evidence_path,
        private_key=keys["private_key"],
        out=signature,
    )
    verified = verify_release_evidence(
        evidence_path,
        signature=signature,
        public_key=keys["public_key"],
    )
    assert verified["ok"] is True
    assert verified["local_eval_bundles_checked"] == [document["eval_bundles"][0]["path"]]

    bundle_path = Path(document["eval_bundles"][0]["path"])
    mutated = dict(bundle_manifest)
    mutated["evaluation_digest"] = digest_uri(sha256_bytes(b"tampered-evaluation"))
    dump_json(mutated, bundle_path)
    with pytest.raises(ConformanceError, match="eval bundle digest mismatch"):
        verify_release_evidence(
            evidence_path,
            signature=signature,
            public_key=keys["public_key"],
        )

    # Restore file digest, then only mutate the evaluation_digest claim in evidence.
    dump_json(bundle_manifest, bundle_path)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["eval_bundles"][0]["evaluation_digest"] = digest_uri(
        sha256_bytes(b"wrong-claim")
    )
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ConformanceError, match="subject digest"):
        verify_release_evidence(
            evidence_path,
            signature=signature,
            public_key=keys["public_key"],
        )


def test_qualification_ledger_rejects_ambiguous_health(tmp_path: Path) -> None:
    pytest.importorskip("cryptography")
    keys = generate_signing_keypair(
        private_key=tmp_path / "private.pem",
        public_key=tmp_path / "public.pem",
    )
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_bytes(
        canonical_json(
            {
                "schema": "arena.qualification-ledger/v1",
                "release": "1.0.0",
                "records": [
                    {"capability": "hf", "status": "pass"},
                    {"capability": "hf", "status": "stale"},
                ],
            }
        )
        + b"\n"
    )
    with pytest.raises(SchemaError, match="duplicate"):
        sign_qualification_ledger(
            ledger_path,
            private_key=keys["private_key"],
            out=tmp_path / "ledger.sig.json",
        )


@pytest.mark.skipif(os.name != "posix", reason="process-group contract is POSIX-stable")
def test_supervisor_success_keeps_streams_separate() -> None:
    result = run_supervised(
        [
            sys.executable,
            "-c",
            "import sys; print('out'); print('err', file=sys.stderr)",
        ],
        timeout_seconds=5,
    )
    assert result.returncode == 0
    assert result.stdout == "out\n"
    assert result.stderr == "err\n"
