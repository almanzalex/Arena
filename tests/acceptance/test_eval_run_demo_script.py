"""Hermetic coverage for examples/eval/run_demo.sh (does not mutate demo/)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("pettingzoo")

REPO = Path(__file__).resolve().parents[2]
DEMO_SCRIPT = REPO / "examples" / "eval" / "run_demo.sh"
DEMO_SRC = REPO / "examples" / "eval" / "demo"


@pytest.mark.acceptance
@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_eval_run_demo_sh_is_hermetic(tmp_path: Path) -> None:
    assert DEMO_SCRIPT.is_file()
    assert (DEMO_SRC / "evaluation.yaml").is_file()

    work = tmp_path / "demo-work"
    work.mkdir()
    # Snapshot checked-in demo paths so we can prove the script did not dirty them.
    before = {
        p.relative_to(DEMO_SRC): p.stat().st_mtime_ns
        for p in DEMO_SRC.rglob("*")
        if p.is_file()
    }

    env = os.environ.copy()
    venv_bin = Path(sys.executable).resolve().parent
    env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"
    env["ARENA_PYTHON"] = sys.executable
    env["ARENA_BIN"] = f"{sys.executable} -m arena"
    env["ARENA_EVAL_DEMO_WORK"] = str(work)
    env["ARENA_EVAL_DEMO_KEEP"] = "1"
    # Avoid inheriting a parent .arena / trainer path surprises.
    env.pop("PYTHONPATH", None)
    env["PYTHONPATH"] = str(REPO)

    proc = subprocess.run(
        ["bash", str(DEMO_SCRIPT)],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "nontransitivity_warning" in proc.stdout or "digests bound" in proc.stdout
    assert (work / "eval-run" / "report.json").is_file()
    assert (work / "bundle" / "bundle.json").is_file()
    assert (work / "slice").is_dir()

    report = (work / "eval-run" / "report.json").read_text(encoding="utf-8")
    assert "sampling_ledger_digest" in report
    assert "execution_binding_digest" in report
    assert "nontransitivity_warning" in report

    after = {
        p.relative_to(DEMO_SRC): p.stat().st_mtime_ns
        for p in DEMO_SRC.rglob("*")
        if p.is_file()
    }
    assert after == before, "run_demo.sh must not mutate examples/eval/demo"


@pytest.mark.acceptance
@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_eval_matrix_cli_from_two_policies(tmp_path: Path, monkeypatch, capsys) -> None:
    import json

    from arena.cli.main import main
    from arena.conformance.fixtures import build_fixed_action_rps_policy
    from arena.core.store import LocalStore

    monkeypatch.chdir(tmp_path)
    LocalStore(tmp_path).init()
    a = build_fixed_action_rps_policy(
        tmp_path / "a.arena", role=["player_0", "player_1"], action=0, name="a"
    )
    b = build_fixed_action_rps_policy(
        tmp_path / "b.arena", role=["player_0", "player_1"], action=1, name="b"
    )
    out = tmp_path / "matrix"
    code = main(
        [
            "eval",
            "matrix",
            "--policy",
            str(a),
            "--policy",
            str(b),
            "--env",
            "arena/competitive_rps_v0",
            "--config",
            '{"max_cycles": 1}',
            "--out",
            str(out),
            "--json",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err or captured.out
    summary = json.loads(captured.out)
    assert summary["cells"] == 4
    assert summary["population_digest"].startswith("sha256:")
    assert summary["sampling_ledger_digest"].startswith("sha256:")
    assert summary["execution_binding_digest"].startswith("sha256:")
    assert (out / "report.json").exists()
