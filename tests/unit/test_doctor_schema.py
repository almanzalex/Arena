from __future__ import annotations

import json

from arena.core.support import (
    DOCTOR_CAPABILITY_KEYS,
    DOCTOR_SCHEMA,
    DOCTOR_SUMMARY_KEYS,
    DOCTOR_TOP_LEVEL_KEYS,
    doctor_report,
    format_doctor_human,
    load_support_matrix,
)

REQUIRED_PREVIEW = ("openenv", "hf", "gimitest")


def test_support_matrix_keeps_required_integrations_preview() -> None:
    matrix = load_support_matrix()
    expected_evidence = {
        "openenv": "docs/qualifications/openenv/R-05-openenv-separate-service.json",
        "gimitest": "docs/qualifications/gimitest/R-06-gimitest.json",
        "hf": "none-attached",
    }
    for name in REQUIRED_PREVIEW:
        assert matrix["capabilities"][name]["status"] == "preview"
        assert matrix["capabilities"][name].get("evidence") == expected_evidence[name]
        assert matrix["capabilities"][name].get("stable_requires")
        assert matrix["capabilities"][name].get("qualify")


def test_doctor_report_schema_stability() -> None:
    report = doctor_report()
    assert report["schema"] == DOCTOR_SCHEMA
    assert tuple(report) == DOCTOR_TOP_LEVEL_KEYS
    assert tuple(report["summary"]) == DOCTOR_SUMMARY_KEYS
    assert set(report["platform"]) == {"os", "arch", "python"}
    for capability in report["capabilities"]:
        assert tuple(capability) == DOCTOR_CAPABILITY_KEYS
        assert capability["authentication_attempted"] is False
        assert capability["usable_today"] in {"stable", "preview", "no"}
        assert capability["local_status"] in {"ready", "locally-unqualified"}
        assert capability["release_status"] in {"stable", "preview", "unsupported"}


def test_doctor_does_not_promote_preview_to_stable_when_deps_ready() -> None:
    report = doctor_report()
    by_name = {item["name"]: item for item in report["capabilities"]}
    for name in REQUIRED_PREVIEW:
        item = by_name[name]
        assert item["release_status"] == "preview"
        assert item["usable_today"] != "stable"
        if item["local_status"] == "ready":
            assert item["usable_today"] == "preview"
            assert name in report["summary"]["usable_today_preview"]
        else:
            assert item["usable_today"] == "no"
            assert name in report["summary"]["locally_unqualified"]
        assert any(
            gap["name"] == name
            for gap in report["summary"]["preview_missing_for_stable"]
        )


def test_doctor_human_output_separates_preview_and_stable() -> None:
    text = format_doctor_human(doctor_report())
    assert "Usable today (stable claim):" in text
    assert "preview only" in text
    assert "What is missing to promote preview" in text
    assert "Platform:" in text
    assert "how to qualify:" in text
    assert "Doctor never authenticates." in text


def test_doctor_cli_json_preserves_schema(capsys) -> None:
    from arena.cli.main import main

    assert main(["doctor", "--json"]) == 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["schema"] == "arena.cli-result/v1"
    assert envelope["command"] == "doctor"
    data = envelope["data"]
    assert data["schema"] == DOCTOR_SCHEMA
    assert tuple(data) == DOCTOR_TOP_LEVEL_KEYS
    for name in REQUIRED_PREVIEW:
        row = next(item for item in data["capabilities"] if item["name"] == name)
        assert row["release_status"] == "preview"
        assert row["usable_today"] != "stable"
