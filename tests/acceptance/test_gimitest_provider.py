from __future__ import annotations

import shutil
from pathlib import Path

import pytest

pytest.importorskip("gimitest")
pytest.importorskip("pettingzoo")
pytest.importorskip("torch")

from rlx.cli.main import main
from rlx.conformance.qualification import qualify_evaluation_fixture
from rlx.core.identity import canonical_json, digest_uri, sha256_bytes
from rlx.runtime.evaluation import build_eval_report, run_evaluation


@pytest.mark.acceptance
@pytest.mark.requires_gimitest
def test_i01_gimitest_provider_records_complete_lineage(tmp_path: Path) -> None:
    rock = Path("examples/eval/demo/rock.rlx").resolve()
    paper = Path("examples/eval/demo/paper.rlx").resolve()
    provider_config = {
        "suite": "base-hooks",
        "test_class": "gimitest.gtest:GTest",
        "parameters": {"purpose": "RLX provider qualification"},
    }
    suite = {
        "schema": "rlx.evaluation/v0alpha1",
        "name": "gimitest-lineage",
        "provider": "gimitest",
        "provider_config": provider_config,
        "interaction": "parallel",
        "task": {
            "adapter": "pettingzoo-parallel",
            "env": "rlx/competitive_rps_v0",
            "interaction": "parallel",
            "config": {"max_cycles": 1},
        },
        "assignments": {"player_0": str(rock), "player_1": str(paper)},
        "seeds": [0],
        "action_mode": "deterministic",
        "metrics": ["mean_return"],
    }
    result = run_evaluation(suite, policy_index={}, out_dir=tmp_path / "run")
    expected_config_digest = digest_uri(sha256_bytes(canonical_json(provider_config)))
    assert result["provider"]["kind"] == "gimitest"
    assert result["provider"]["config_digest"] == expected_config_digest
    assert result["task_digest"].startswith("sha256:")
    assert len(result["cells"]) == 1
    lineage = result["cells"][0]["lineage"]
    assert lineage["task_digest"] == result["task_digest"]
    assert lineage["provider"] == result["provider"]
    assert len(lineage["policy_digests"]) == 2
    assert result["cells"][0]["failures"] == 0

    report = build_eval_report(result)
    assert report["provider"] == result["provider"]
    assert report["task_digest"] == result["task_digest"]


def test_external_gimitest_class_requires_explicit_trust() -> None:
    from rlx.adapters.eval_gimitest import _resolve_test_class
    from rlx.core.errors import SchemaError

    with pytest.raises(SchemaError, match="execute Python"):
        _resolve_test_class("tests.some_lab:Scenario", allow_external=False)


@pytest.mark.acceptance
@pytest.mark.requires_gimitest
def test_gimitest_cli_and_qualification_fixture(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    shutil.copytree("examples/eval/demo", fixture / "demo")
    shutil.copy2("examples/eval/robustness.yaml", fixture / "robustness.yaml")

    assert main(
        [
            "eval",
            "run",
            str(fixture / "robustness.yaml"),
            "--out",
            str(tmp_path / "cli-run"),
            "--provider",
            "gimitest",
            "--json",
        ]
    ) == 0
    qualification = qualify_evaluation_fixture(
        fixture / "robustness.yaml",
        report_path=tmp_path / "gimitest-qualification.json",
    )
    assert qualification["ok"] is True
    assert qualification["checks"]["provider_lineage"]["ok"] is True
