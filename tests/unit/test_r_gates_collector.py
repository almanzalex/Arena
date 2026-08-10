"""Tests for the R-gates release-evidence collector (stream D)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from r_gates.collect_release_evidence import (  # noqa: E402
    COLLECTOR_SCHEMA,
    collect_release_evidence,
    main,
)
from r_gates.gates import EXTERNAL_FLOOR_GATES, MANDATORY_GATE_IDS  # noqa: E402


def test_collector_emits_skeleton_with_all_gates(tmp_path: Path) -> None:
    out = tmp_path / "evidence"
    document = collect_release_evidence(repo=REPO, out_dir=out)

    assert document["schema"] == COLLECTOR_SCHEMA
    assert document["kind"] == "skeleton"
    assert document["schema"] != "arena.release-evidence/v1"
    assert [gate["id"] for gate in document["gates"]] == list(MANDATORY_GATE_IDS)

    index = out / "release-index.json"
    assert index.is_file()
    on_disk = json.loads(index.read_text(encoding="utf-8"))
    assert on_disk["schema"] == COLLECTOR_SCHEMA

    slots = {gate["id"]: gate["slot"] for gate in document["gates"]}
    for gate_id in EXTERNAL_FLOOR_GATES:
        assert slots[gate_id] == "missing", gate_id
        assert document["gates"][int(gate_id[2:]) - 1]["status"] != "pass"

    assert slots["R-09"] == "local-partial"
    assert slots["R-10"] == "local-partial"
    assert slots["R-13"] == "local-partial"
    assert slots["R-01"] == "missing"
    assert slots["R-11"] == "missing"
    assert slots["R-14"] == "missing"

    assert (out / "local" / "schema-registry.snapshot.json").is_file()
    assert (out / "local" / "golden-fixture-digests.json").is_file()
    assert (out / "local" / "perf-smoke-baselines.json").is_file()
    assert (out / "local" / "hermetic-capable.json").is_file()
    assert (out / "local" / "doctor.json").is_file()

    summary = document["summary"]
    assert "R-04" in summary["external_floor_missing"]
    assert "R-05" in summary["external_floor_missing"]
    assert "R-06" in summary["external_floor_missing"]


def test_collector_never_marks_external_floor_pass_without_attach(
    tmp_path: Path,
) -> None:
    document = collect_release_evidence(repo=REPO, out_dir=tmp_path / "evidence")
    for gate in document["gates"]:
        if gate["id"] in EXTERNAL_FLOOR_GATES:
            assert gate["slot"] == "missing"
            assert gate["status"] == "missing"
            assert gate["evidence_path"] is None


def test_attach_live_hf_fills_r04_only(tmp_path: Path) -> None:
    live = tmp_path / "hf-live.json"
    live.write_text(
        json.dumps(
            {
                "schema": "arena.store-qualification/v1",
                "capability": "hf",
                "mode": "live",
                "simulation": False,
                "uri": "hf://models/example/arena?revision=abc123",
            }
        ),
        encoding="utf-8",
    )
    document = collect_release_evidence(
        repo=REPO,
        out_dir=tmp_path / "evidence",
        attach={"R-04": live},
    )
    by_id = {gate["id"]: gate for gate in document["gates"]}
    assert by_id["R-04"]["slot"] == "filled"
    assert by_id["R-04"]["status"] == "attached"
    assert by_id["R-04"]["evidence_digest"].startswith("sha256:")
    assert by_id["R-05"]["slot"] == "missing"
    assert by_id["R-06"]["slot"] == "missing"


def test_rejects_simulated_hf_attach(tmp_path: Path) -> None:
    simulated = tmp_path / "hf-sim.json"
    simulated.write_text(
        json.dumps(
            {
                "schema": "arena.store-qualification/v1",
                "mode": "simulation",
                "uri": "hf://models/x/y?simulate=/tmp/fake",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="simulated evidence"):
        collect_release_evidence(
            repo=REPO,
            out_dir=tmp_path / "evidence",
            attach={"R-04": simulated},
        )


def test_rejects_loopback_openenv_without_separate_flag(tmp_path: Path) -> None:
    loopback = tmp_path / "openenv-local.json"
    loopback.write_text(
        json.dumps(
            {
                "schema": "arena.adapter-qualification/v1",
                "capability": "openenv",
                "endpoint": "http://127.0.0.1:8000",
                "deployment": "loopback",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="loopback OpenEnv"):
        collect_release_evidence(
            repo=REPO,
            out_dir=tmp_path / "evidence",
            attach={"R-05": loopback},
        )


def test_accepts_separately_deployed_openenv(tmp_path: Path) -> None:
    remote = tmp_path / "openenv-separate.json"
    remote.write_text(
        json.dumps(
            {
                "schema": "arena.adapter-qualification/v1",
                "capability": "openenv",
                "separately_deployed": True,
                "deployment": "operated-service",
                "endpoint": "https://openenv.example.invalid/task",
                "mode": "live",
            }
        ),
        encoding="utf-8",
    )
    document = collect_release_evidence(
        repo=REPO,
        out_dir=tmp_path / "evidence",
        attach={"R-05": remote},
    )
    by_id = {gate["id"]: gate for gate in document["gates"]}
    assert by_id["R-05"]["slot"] == "filled"
    assert by_id["R-04"]["slot"] == "missing"


def test_refuses_overwrite_of_signed_release_evidence(tmp_path: Path) -> None:
    out = tmp_path / "evidence"
    out.mkdir()
    index = out / "release-index.json"
    index.write_text(
        json.dumps(
            {
                "schema": "arena.release-evidence/v1",
                "release": "1.0.0",
                "tag": "v1.0.0",
                "commit": "a" * 40,
                "gates": [],
                "artifacts": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        collect_release_evidence(repo=REPO, out_dir=out)


def test_cli_main_writes_skeleton(tmp_path: Path, capsys) -> None:
    out = tmp_path / "evidence"
    code = main(["--repo", str(REPO), "--out", str(out)])
    assert code == 0
    captured = capsys.readouterr()
    assert "External floor still missing" in captured.out
    assert (out / "release-index.json").is_file()
    document = json.loads((out / "release-index.json").read_text(encoding="utf-8"))
    assert document["summary"]["external_floor_missing"] == ["R-04", "R-05", "R-06"]


def test_templates_cover_all_gates() -> None:
    templates = REPO / "evidence" / "templates"
    for gate_id in MANDATORY_GATE_IDS:
        matches = list(templates.glob(f"{gate_id}-*.json"))
        assert matches, f"missing template for {gate_id}"
        payload = json.loads(matches[0].read_text(encoding="utf-8"))
        assert payload.get("gate") == gate_id
