"""Smoke test for scripts/rehearse_release_sign.sh loud-fail behavior."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "rehearse_release_sign.sh"


def test_rehearse_release_sign_lists_missing_gates(tmp_path: Path) -> None:
    assert SCRIPT.is_file()
    key_dir = tmp_path / "keys"
    key_dir.mkdir()
    # Point evidence at an empty dir so every gate is missing.
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    env = {
        **os.environ,
        "EVIDENCE_DIR": str(evidence_dir),
        "KEY_DIR": str(key_dir),
    }
    # KEY_DIR under tmp_path is not /tmp or evidence/local — script must reject
    # unsafe dirs. Use /tmp child instead.
    safe_keys = Path("/tmp") / f"arena-rehearse-test-{os.getpid()}"
    safe_keys.mkdir(parents=True, exist_ok=True)
    env["KEY_DIR"] = str(safe_keys)
    try:
        proc = subprocess.run(
            ["bash", str(SCRIPT)],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        for child in safe_keys.iterdir():
            child.unlink()
        safe_keys.rmdir()

    assert proc.returncode == 2, proc.stdout + proc.stderr
    combined = proc.stdout + proc.stderr
    assert "refusing to invent gate passes" in combined
    assert "R-01=" in combined
    assert "R-14=" in combined
    assert "no release-index was written" in combined


def test_rehearse_release_sign_rejects_unsafe_key_dir(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "KEY_DIR": str(tmp_path / "unsafe-keys"),
        "EVIDENCE_DIR": str(tmp_path / "evidence"),
    }
    (tmp_path / "evidence").mkdir()
    (tmp_path / "unsafe-keys").mkdir()
    proc = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "KEY_DIR must be under /tmp" in (proc.stdout + proc.stderr)
