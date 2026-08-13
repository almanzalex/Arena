"""Smoke test for scripts/rehearse_release_sign.sh loud-fail behavior."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "rehearse_release_sign.sh"


def test_rehearse_release_sign_lists_missing_gates(tmp_path: Path) -> None:
    assert SCRIPT.is_file()
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    # Prefer an explicit /tmp child so the safety check accepts KEY_DIR even when
    # pytest's tmp_path is outside /tmp (common on macOS).
    safe_keys = Path("/tmp") / f"arena-rehearse-test-{os.getpid()}"
    safe_keys.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "EVIDENCE_DIR": str(evidence_dir),
        "KEY_DIR": str(safe_keys),
    }
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
        shutil.rmtree(safe_keys, ignore_errors=True)

    assert proc.returncode == 2, proc.stdout + proc.stderr
    combined = proc.stdout + proc.stderr
    assert "refusing to invent gate passes" in combined
    assert "R-01=" in combined
    assert "R-14=" in combined
    assert "no release-index was written" in combined


def test_rehearse_release_sign_rejects_unsafe_key_dir(tmp_path: Path) -> None:
    # Must be outside /tmp and outside evidence/local — pytest tmp under /tmp on
    # Linux CI would otherwise be accepted by the script's safety check.
    unsafe = REPO / f".arena-rehearse-unsafe-{os.getpid()}"
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    unsafe.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "KEY_DIR": str(unsafe),
        "EVIDENCE_DIR": str(evidence_dir),
    }
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
        shutil.rmtree(unsafe, ignore_errors=True)

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "KEY_DIR must be under /tmp" in (proc.stdout + proc.stderr)
