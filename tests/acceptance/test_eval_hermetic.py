"""Hermetic + doc-driven eval clean-room gate (0.2 / U-02 companion)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from _eval_fixtures import build_cyclic_rps_eval_fixture

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_CLEAN_ROOM_DOC = REPO_ROOT / "docs" / "eval-clean-room.md"

# Sibling module fixtures (hermetic_wheel / hermetic_wheelhouse).
pytest_plugins = ["test_u01_hermetic"]

EXPECTED_EVAL_COMMANDS: tuple[str, ...] = (
    "arena init",
    "arena population create ./population.yaml --ref populations/opp",
    "arena eval validate ./evaluation.yaml --population ./population.yaml",
    (
        "arena eval run ./evaluation.yaml "
        "--policy rock=./rock.arena --policy paper=./paper.arena --policy scissors=./scissors.arena "
        "--population ./population.yaml --out ./eval-run"
    ),
    "arena eval report ./eval-run --out ./eval-run",
    "arena data select ./eval-run --out ./slice --outcome loss --role player_0",
    "arena eval bundle ./eval-run --out ./bundle --report ./eval-run/report.json",
)


def _parse_eval_cleanroom_commands(path: Path) -> list[str]:
    import re

    text = path.read_text(encoding="utf-8")
    fence = re.compile(r"^```(?P<info>[^\n]*)\n(?P<body>.*?)^```", re.DOTALL | re.MULTILINE)
    bodies = []
    for m in fence.finditer(text):
        info = m.group("info").strip().split()
        if "arena-eval-clean-room" in info:
            bodies.append(m.group("body"))
    assert bodies, f"missing arena-eval-clean-room block in {path}"
    assert len(bodies) == 1
    flat: list[str] = []
    buf = ""
    for line in bodies[0].splitlines():
        raw = line.rstrip()
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        if raw.endswith("\\"):
            buf += raw[:-1].rstrip() + " "
            continue
        buf += raw.strip()
        flat.append(" ".join(buf.split()))
        buf = ""
    if buf.strip():
        flat.append(" ".join(buf.split()))
    return flat


@pytest.mark.acceptance
def test_eval_cleanroom_doc_matches_cli() -> None:
    commands = _parse_eval_cleanroom_commands(EVAL_CLEAN_ROOM_DOC)
    assert commands == list(EXPECTED_EVAL_COMMANDS)


@pytest.mark.slow
@pytest.mark.acceptance
@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_eval_hermetic_venv(
    hermetic_wheel: Path,
    hermetic_wheelhouse: Path,
    tmp_path: Path,
) -> None:
    """Fresh wheel install: population → eval → report → select → bundle, offline."""
    from test_u01_hermetic import _NET_GUARD_SITECUSTOMIZE

    fx = build_cyclic_rps_eval_fixture(tmp_path / "author")
    venv_dir = tmp_path / "venv"
    sandbox = tmp_path / "sandbox"
    fake_home = tmp_path / "home"
    for d in (fake_home / ".cache", fake_home / ".config", fake_home / ".local" / "share"):
        d.mkdir(parents=True, exist_ok=True)

    r = subprocess.run(
        [sys.executable, "-m", "venv", "--clear", str(venv_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    vbin = venv_dir / ("Scripts" if os.name == "nt" else "bin")
    vpython = vbin / ("python.exe" if os.name == "nt" else "python")

    r = subprocess.run(
        [
            str(vpython),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(hermetic_wheelhouse),
            f"{hermetic_wheel}[torch,pettingzoo]",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, f"wheel install failed:\n{r.stdout}\n{r.stderr}"

    site = subprocess.run(
        [str(vpython), "-c", "import site; print(site.getsitepackages()[0])"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    (Path(site) / "sitecustomize.py").write_text(_NET_GUARD_SITECUSTOMIZE, encoding="utf-8")

    sandbox.mkdir()
    for name in ("rock.arena", "paper.arena", "scissors.arena"):
        shutil.copytree(fx["root"] / name, sandbox / name)
    shutil.copy2(fx["population"], sandbox / "population.yaml")
    shutil.copy2(fx["evaluation"], sandbox / "evaluation.yaml")
    shutil.copy2(EVAL_CLEAN_ROOM_DOC, sandbox / "eval-clean-room.md")

    base_env = {
        "PATH": f"{vbin}{os.pathsep}/usr/bin:/bin",
        "HOME": str(fake_home),
        "XDG_CACHE_HOME": str(fake_home / ".cache"),
        "XDG_CONFIG_HOME": str(fake_home / ".config"),
        "XDG_DATA_HOME": str(fake_home / ".local" / "share"),
        "TMPDIR": str(tmp_path / "tmp"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "OMP_NUM_THREADS": "1",
    }
    (tmp_path / "tmp").mkdir(exist_ok=True)
    run_env = {**base_env, "ARENA_CLEANROOM_NO_NET": "1", "PIP_NO_INDEX": "1"}

    got = subprocess.run(
        [str(vpython), "-c", "import arena; print(arena.__file__); print(arena.__version__)"],
        cwd=sandbox,
        env=base_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert got.returncode == 0, got.stderr
    assert "site-packages" in got.stdout
    assert str(REPO_ROOT) not in got.stdout

    # Build policy CLI args — docs use friendly names; CLI indexes digests via Policy.load.
    commands = _parse_eval_cleanroom_commands(sandbox / "eval-clean-room.md")
    assert commands == list(EXPECTED_EVAL_COMMANDS)

    for cmd in commands:
        argv = cmd.split()
        assert argv[0] == "arena"
        proc = subprocess.run(
            [str(vpython), "-m", "arena", *argv[1:]],
            cwd=sandbox,
            env=run_env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"cmd failed: {cmd}\n{proc.stdout}\n{proc.stderr}"
        if cmd.startswith("arena eval run"):
            # Prefer --out path; also accept store run_dir if printed.
            assert (sandbox / "eval-run").exists() or "run_dir" in proc.stdout, proc.stdout

    assert (sandbox / "eval-run" / "eval_run.json").exists(), list((sandbox / ".arena" / "runs").glob("*") if (sandbox / ".arena" / "runs").exists() else [])
    assert (sandbox / "eval-run" / "report.json").exists()
    assert (sandbox / "slice" / "dataset.json").exists()
    assert (sandbox / "bundle" / "bundle.json").exists()
    report = (sandbox / "eval-run" / "report.json").read_text(encoding="utf-8")
    assert "nontransitivity_warning" in report
