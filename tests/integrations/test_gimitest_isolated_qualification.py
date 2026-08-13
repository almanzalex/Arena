"""Gimitest isolated-worker doctor/qualification contracts (no matrix stable flip)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from arena.adapters.eval_gimitest import ISOLATED_PYTHON_ENV, resolve_isolation
from arena.core.errors import SchemaError
from arena.core.support import capability_report, doctor_report


def test_gimitest_doctor_locally_unqualified_without_env(monkeypatch) -> None:
    monkeypatch.delenv(ISOLATED_PYTHON_ENV, raising=False)
    report = capability_report("gimitest")
    assert report["local_status"] == "locally-unqualified"
    assert report["usable_today"] == "no"
    assert report["release_status"] == "preview"
    assert report["isolated_probe"]["status"] == "unavailable"
    assert ISOLATED_PYTHON_ENV in (report.get("repair") or "")
    assert doctor_report("gimitest")["ok"] is False


def test_gimitest_doctor_ready_with_configured_worker(monkeypatch) -> None:
    """Same-machine proof: pointing the env at a ready interpreter qualifies locally.

    This does not claim release-stable. Matrix may point at local R-06 JSON while
    status stays preview until claimed-platform release CI repeats the proof.
    """
    # Use abspath, not resolve(): resolve() follows the venv symlink to the base
    # interpreter and falsely reports the worker as missing packages.
    monkeypatch.setenv(ISOLATED_PYTHON_ENV, os.path.abspath(sys.executable))
    report = capability_report("gimitest")
    assert report["isolated_probe"]["status"] == "ready"
    assert report["local_status"] == "ready"
    assert report["usable_today"] == "preview"
    assert report["release_status"] == "preview"
    assert report["evidence"] == "docs/qualifications/gimitest/R-06-gimitest.json"
    full = doctor_report("gimitest")
    assert full["ok"] is True
    assert "gimitest" not in full["summary"]["locally_unqualified"]


def test_resolve_isolation_fills_python_from_env(monkeypatch, tmp_path: Path) -> None:
    worker = tmp_path / "python"
    worker.write_text("#!/bin/sh\n", encoding="utf-8")
    worker.chmod(0o755)
    monkeypatch.setenv(ISOLATED_PYTHON_ENV, str(worker))
    isolation = resolve_isolation({"isolation": {"mode": "subprocess"}})
    assert isolation["python"] == str(worker)


def test_resolve_isolation_requires_env_or_python(monkeypatch) -> None:
    monkeypatch.delenv(ISOLATED_PYTHON_ENV, raising=False)
    with pytest.raises(SchemaError, match=ISOLATED_PYTHON_ENV):
        resolve_isolation({"isolation": {"mode": "subprocess"}})


@pytest.mark.requires_gimitest
def test_gimitest_subprocess_uses_env_python_when_omitted(
    tmp_path: Path, monkeypatch
) -> None:
    pytest.importorskip("gimitest")
    pytest.importorskip("pettingzoo")
    pytest.importorskip("torch")

    from arena.runtime.evaluation import run_evaluation

    monkeypatch.setenv(ISOLATED_PYTHON_ENV, os.path.abspath(sys.executable))
    suite = {
        "schema": "arena.evaluation/v0alpha1",
        "name": "gimitest-env-isolation",
        "provider": "gimitest",
        "provider_config": {
            "semantic": {},
            "test_class": "gimitest.gtest:GTest",
            "parameters": {"purpose": "env-driven isolation"},
            "isolation": {"mode": "subprocess", "timeout_seconds": 60},
        },
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
    result = run_evaluation(suite, policy_index={}, out_dir=tmp_path / "run")
    assert result["state"] == "complete"
    worker = (result.get("provider") or {}).get("worker") or {}
    assert worker.get("protocol") == "arena.eval-provider/v1"
    assert os.path.abspath(worker["python"]["executable"]) == os.path.abspath(
        os.environ[ISOLATED_PYTHON_ENV]
    )
