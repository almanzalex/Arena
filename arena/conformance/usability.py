"""Literate clean-room usability harness.

Unlike a command-list parser, this evaluates the reader-facing guide as a
handoff artifact: required context must be present, the only runnable commands
are discovered from the prose, and every attempt is retained with timing and
output for release review. It cannot substitute for human comprehension.
"""

from __future__ import annotations

import re
import shlex
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from arena.core.manifests import dump_json

_FENCE = re.compile(r"^```(?P<info>[^\n]*)\n(?P<body>.*?)^```", re.MULTILINE | re.DOTALL)
_REQUIRED_GUIDE_CONCEPTS = (
    "Fresh virtualenv",
    "What you should receive",
    "Success criteria",
    "Troubleshooting",
    "Remaining human step",
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def read_clean_room_guide(path: Path | str) -> list[str]:
    """Read the guide like a newcomer and derive the authored execution plan."""
    text = Path(path).read_text(encoding="utf-8")
    missing = [item for item in _REQUIRED_GUIDE_CONCEPTS if item not in text]
    if missing:
        raise AssertionError(f"clean-room guide omits reader-critical sections: {missing}")
    blocks = [
        match.group("body")
        for match in _FENCE.finditer(text)
        if "arena-clean-room" in match.group("info").split()
    ]
    if len(blocks) != 1:
        raise AssertionError("guide must contain exactly one arena-clean-room command block")
    commands = [
        line.strip()
        for line in blocks[0].splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not commands or any(not command.startswith("arena ") for command in commands):
        raise AssertionError("guide must give a non-empty, arena-only received-artifact flow")
    return commands


def run_blind_reader(
    guide: Path | str,
    *,
    cwd: Path | str,
    env: dict[str, str],
    transcript_path: Path | str,
) -> dict[str, Any]:
    """Execute a guide-derived plan and preserve a literate reader transcript."""
    started = _now()
    attempts: list[dict[str, Any]] = []
    success = True
    for command in read_clean_room_guide(guide):
        t0 = time.monotonic()
        proc = subprocess.run(
            shlex.split(command),
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        attempts.append(
            {
                "command": command,
                "exit_code": proc.returncode,
                "elapsed_seconds": round(time.monotonic() - t0, 6),
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        )
        if proc.returncode:
            success = False
            break
    report = {
        "schema": "arena.blind-reader-usability/v1",
        "started_at": started,
        "finished_at": _now(),
        "success_from_docs_and_received_artifacts_only": success,
        "attempts": attempts,
        "friction_score": 1 if success else 5,
        "automated_limit": (
            "This harness verifies authored instructions and isolated execution. "
            "A real newcomer must still score prose clarity, ambiguity, and intervention."
        ),
    }
    dump_json(report, transcript_path)
    return report
