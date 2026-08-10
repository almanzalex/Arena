from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

pytest.importorskip("gimitest")
pytest.importorskip("pettingzoo")
pytest.importorskip("torch")

from arena.cli.main import main
from arena.conformance.qualification import qualify_evaluation_fixture
from arena.core.identity import canonical_json, digest_uri, sha256_bytes
from arena.runtime.evaluation import build_eval_report, run_evaluation


@pytest.mark.acceptance
@pytest.mark.requires_gimitest
def test_i01_gimitest_provider_records_complete_lineage(tmp_path: Path) -> None:
    rock = Path("examples/eval/demo/rock.arena").resolve()
    paper = Path("examples/eval/demo/paper.arena").resolve()
    provider_config = {
        "semantic": {},
        "suite": "base-hooks",
        "test_class": "gimitest.gtest:GTest",
        "parameters": {"purpose": "Arena provider qualification"},
    }
    suite = {
        "schema": "arena.evaluation/v0alpha1",
        "name": "gimitest-lineage",
        "provider": "gimitest",
        "provider_config": provider_config,
        "interaction": "parallel",
        "task": {
            "adapter": "pettingzoo-parallel",
            "env": "arena/competitive_rps_v0",
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

    native = run_evaluation(
        suite,
        policy_index={},
        out_dir=tmp_path / "native-run",
        provider="native",
    )
    assert result["evaluation_intent_digest"] == native["evaluation_intent_digest"]
    assert result["execution_binding_digest"] != native["execution_binding_digest"]
    assert result["semantic_result_digest"] == native["semantic_result_digest"]

    report = build_eval_report(result)
    assert report["provider"] == result["provider"]
    assert report["task_digest"] == result["task_digest"]


def test_external_gimitest_class_requires_explicit_trust() -> None:
    from arena.adapters.eval_gimitest import _resolve_test_class
    from arena.core.errors import SchemaError

    with pytest.raises(SchemaError, match="execute Python"):
        _resolve_test_class("tests.some_lab:Scenario", allow_external=False)


@pytest.mark.acceptance
@pytest.mark.requires_gimitest
def test_gimitest_non_noop_scenario_changes_intent_and_result_digests(
    tmp_path: Path,
) -> None:
    rock = str(Path("examples/eval/demo/rock.arena").resolve())
    paper = str(Path("examples/eval/demo/paper.arena").resolve())
    task = {
        "adapter": "pettingzoo-parallel",
        "env": "arena/competitive_rps_v0",
        "interaction": "parallel",
        "config": {"max_cycles": 1},
    }
    base = {
        "schema": "arena.evaluation/v0alpha1",
        "interaction": "parallel",
        "task": task,
        "assignments": {"player_0": rock, "player_1": paper},
        "seeds": [0],
        "action_mode": "deterministic",
        "metrics": ["mean_return"],
    }
    native = run_evaluation(
        {**base, "name": "gimitest-native-baseline", "provider": "native"},
        policy_index={},
        out_dir=tmp_path / "native",
    )
    transformed = run_evaluation(
        {
            **base,
            "name": "gimitest-non-noop",
            "provider": "gimitest",
            "provider_config": {
                "test_class": (
                    "arena.adapters.eval_gimitest.scenarios:RewardTransformScenario"
                ),
                "parameters": {"reward_scale": -1.0},
            },
        },
        policy_index={},
        out_dir=tmp_path / "transformed",
    )
    returns = transformed["cell_results"][0]["episodes"][0]["returns"]
    assert returns == {"player_0": 1.0, "player_1": -1.0}
    assert (
        transformed["evaluation_intent_digest"] != native["evaluation_intent_digest"]
    )
    assert transformed["semantic_result_digest"] != native["semantic_result_digest"]


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


@pytest.mark.acceptance
@pytest.mark.requires_gimitest
def test_gimitest_can_run_across_isolated_python_worker_boundary(
    tmp_path: Path,
) -> None:
    provider_config = {
        "suite": "base-hooks",
        "test_class": "gimitest.gtest:GTest",
        "parameters": {"purpose": "subprocess provider qualification"},
            "isolation": {
                "mode": "subprocess",
                "python": str(Path(sys.executable)),
                "timeout_seconds": 60,
            },
    }
    suite = {
        "schema": "arena.evaluation/v0alpha1",
        "name": "gimitest-subprocess",
        "provider": "gimitest",
        "provider_config": provider_config,
        "interaction": "parallel",
        "task": {
            "adapter": "pettingzoo-parallel",
            "env": "arena/competitive_rps_v0",
            "interaction": "parallel",
            "config": {"max_cycles": 1},
        },
        "assignments": {
            "player_0": str(Path("examples/eval/demo/rock.arena").resolve()),
            "player_1": str(Path("examples/eval/demo/paper.arena").resolve()),
        },
        "seeds": [0],
        "action_mode": "deterministic",
        "metrics": ["mean_return"],
    }
    result = run_evaluation(
        suite,
        policy_index={},
        out_dir=tmp_path / "subprocess-run",
    )
    assert result["provider"]["kind"] == "gimitest"
    assert result["provider"]["config_digest"] == digest_uri(
        sha256_bytes(canonical_json(provider_config))
    )
    assert result["cells"][0]["failures"] == 0
