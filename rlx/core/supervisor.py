"""Hard-budget subprocess supervision for optional execution boundaries."""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from rlx.core.errors import ExternalUnavailableError, redact


@dataclass(frozen=True)
class SupervisedResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float


def _terminate_tree(process: subprocess.Popen[bytes], *, grace_seconds: float) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    else:  # pragma: no cover - Windows is outside the 1.0 support matrix.
        process.terminate()
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
    else:  # pragma: no cover
        process.kill()
    process.wait(timeout=max(grace_seconds, 1.0))


def run_supervised(
    command: Sequence[str],
    *,
    timeout_seconds: float,
    cwd: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    max_stdout_bytes: int = 1_048_576,
    max_stderr_bytes: int = 1_048_576,
    grace_seconds: float = 2.0,
) -> SupervisedResult:
    """Run an argv-only command with wall-time, output, and process-tree budgets."""
    argv = tuple(str(part) for part in command)
    if not argv:
        raise ValueError("supervised command must not be empty")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if min(max_stdout_bytes, max_stderr_bytes) < 1:
        raise ValueError("output byte limits must be positive")
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="rlx-supervisor-") as raw:
        stdout_path = Path(raw) / "stdout"
        stderr_path = Path(raw) / "stderr"
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            try:
                process = subprocess.Popen(
                    argv,
                    cwd=cwd,
                    env=dict(env) if env is not None else None,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )
            except OSError as exc:
                raise ExternalUnavailableError(
                    f"could not start external process: {redact(str(exc))}",
                    code="EXTERNAL_START_FAILED",
                    cause=type(exc).__name__,
                    context={"executable": argv[0]},
                ) from exc
            reason: str | None = None
            while process.poll() is None:
                elapsed = time.monotonic() - started
                stdout.flush()
                stderr.flush()
                if elapsed > timeout_seconds:
                    reason = "timeout"
                    break
                if stdout_path.stat().st_size > max_stdout_bytes:
                    reason = "stdout_limit"
                    break
                if stderr_path.stat().st_size > max_stderr_bytes:
                    reason = "stderr_limit"
                    break
                time.sleep(0.025)
            if reason is not None:
                _terminate_tree(process, grace_seconds=grace_seconds)
            else:
                process.wait()
            stdout.flush()
            stderr.flush()
            if reason is None and stdout_path.stat().st_size > max_stdout_bytes:
                reason = "stdout_limit"
            if reason is None and stderr_path.stat().st_size > max_stderr_bytes:
                reason = "stderr_limit"
        duration = time.monotonic() - started
        stdout_bytes = stdout_path.read_bytes()[:max_stdout_bytes]
        stderr_bytes = stderr_path.read_bytes()[:max_stderr_bytes]
        stdout_text = stdout_bytes.decode("utf-8", errors="replace")
        stderr_text = stderr_bytes.decode("utf-8", errors="replace")
        if reason is not None:
            limit = (
                f"{timeout_seconds}s wall time"
                if reason == "timeout"
                else (
                    f"{max_stdout_bytes} stdout bytes"
                    if reason == "stdout_limit"
                    else f"{max_stderr_bytes} stderr bytes"
                )
            )
            raise ExternalUnavailableError(
                f"external process exceeded {limit}",
                code=f"EXTERNAL_{reason.upper()}",
                cause=reason.replace("_", " "),
                context={
                    "executable": argv[0],
                    "duration_seconds": round(duration, 6),
                    "stderr_tail": redact(stderr_text[-2000:]),
                },
            )
        return SupervisedResult(
            command=argv,
            returncode=int(process.returncode),
            stdout=stdout_text,
            stderr=stderr_text,
            duration_seconds=duration,
        )
