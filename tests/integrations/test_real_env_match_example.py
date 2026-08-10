"""End-to-end: Gymnasium CartPole + OpenSpiel through Match API example."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "examples/integrations/run_real_env_match.py"


@pytest.mark.requires_pettingzoo
@pytest.mark.requires_openspiel
def test_run_real_env_match_example_cartpole_and_openspiel(tmp_path: Path) -> None:
    pytest.importorskip("gymnasium")
    pytest.importorskip("pettingzoo")
    pytest.importorskip("pyspiel")
    pytest.importorskip("torch")

    out = tmp_path / "smoke"
    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLE),
            "--out",
            str(out),
        ],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    summary = json.loads((out / "env-smoke-summary.json").read_text(encoding="utf-8"))
    assert summary["ok"] is True
    assert summary["matches"]["cartpole"]["outcome"]["episodes_completed"] == 2
    assert summary["matches"]["cartpole"]["outcome"]["failure_count"] == 0
    assert (
        summary["matches"]["openspiel_tic_tac_toe"]["outcome"]["episodes_completed"]
        == 3
    )
    # Optional cloud / isolated capabilities must not be reported as authenticated.
    for name, probe in summary["optional_capabilities"].items():
        assert probe.get("authentication_attempted") in (False, None)
        if name == "gimitest":
            # Without ARENA_GIMITEST_PYTHON the example must not claim success.
            assert probe["ok"] is False or probe.get("local_status") == "ready"


@pytest.mark.requires_pettingzoo
def test_run_real_env_match_requires_gimitest_when_asked(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("gymnasium")
    pytest.importorskip("pettingzoo")
    pytest.importorskip("torch")

    monkeypatch.delenv("ARENA_GIMITEST_PYTHON", raising=False)
    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLE),
            "--out",
            str(tmp_path / "need-gimi"),
            "--skip-openspiel",
            "--require",
            "gimitest",
        ],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "gimitest" in (completed.stderr + completed.stdout).lower()
