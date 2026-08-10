"""OpenEnv / Gimitest local qualification and fail-loud (no fake live credentials)."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

import pytest

from arena.core.support import capability_report, doctor_report


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_gimitest_unconfigured_worker_is_locally_unqualified(monkeypatch) -> None:
    monkeypatch.delenv("ARENA_GIMITEST_PYTHON", raising=False)
    report = capability_report("gimitest")
    assert report["local_status"] == "locally-unqualified"
    assert report["isolated_probe"]["status"] == "unavailable"
    assert report["credentials_required"] is False
    assert report["authentication_attempted"] is False
    assert "ARENA_GIMITEST_PYTHON" in (report.get("repair") or "")
    full = doctor_report("gimitest")
    assert full["ok"] is False


def test_credential_capabilities_never_pretend_authenticated(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "canary-not-for-doctor")
    monkeypatch.setenv("WANDB_API_KEY", "canary-wandb")
    for name in ("hf", "wandb", "mlflow", "oci"):
        report = capability_report(name)
        assert report["credentials_required"] is True
        assert report["authentication_attempted"] is False
        payload = json.dumps(report)
        assert "canary-not-for-doctor" not in payload
        assert "canary-wandb" not in payload


def test_openenv_doctor_does_not_require_cloud_credentials() -> None:
    report = capability_report("openenv")
    assert report["credentials_required"] is False
    assert report["authentication_attempted"] is False


@pytest.mark.requires_openenv
def test_openenv_unreachable_endpoint_fails_loud_not_success(tmp_path: Path) -> None:
    pytest.importorskip("openenv")
    pytest.importorskip("pettingzoo")

    from arena.adapters.task_openenv.adapter import PILOT_CONTRACT
    from arena.conformance.fixtures import build_fixed_action_rps_policy
    from arena.core.sdk import Policy
    from arena.runtime.match import run_match

    port = _free_port()
    left = build_fixed_action_rps_policy(
        tmp_path / "left.arena", role=["player_0", "player_1"], action=0
    )
    right = build_fixed_action_rps_policy(
        tmp_path / "right.arena", role=["player_0", "player_1"], action=1
    )
    result = run_match(
        task_spec={
            "adapter": "openenv",
            "env": f"openenv://127.0.0.1:{port}/arena/competitive_rps_v0",
            "interaction": "parallel",
            "packaging": {"kind": "openenv"},
            "contract": PILOT_CONTRACT,
        },
        assignments={"player_0": Policy.load(left), "player_1": Policy.load(right)},
        seeds=[0],
        out_dir=tmp_path / "unreachable",
    )
    assert result["outcome"]["episodes_completed"] == 0
    assert result["outcome"]["failure_count"] == 1
    assert result["failures"][0]["kind"] == "disconnect"


@pytest.mark.requires_openenv
def test_openenv_loopback_server_qualification(tmp_path: Path) -> None:
    pytest.importorskip("openenv")
    pytest.importorskip("pettingzoo")

    from arena.cli.main import main
    from arena.conformance.qualification import qualify_task_fixture
    from arena.core.manifests import load_manifest
    from arena.core.tasks import verify_task_equivalence

    port = _free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "arena.adapters.task_openenv.server",
            "--port",
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 30
        while True:
            if proc.poll() is not None:
                output = proc.stdout.read() if proc.stdout else ""
                pytest.fail(f"OpenEnv loopback exited before ready: {output}")
            try:
                with urlopen(f"http://127.0.0.1:{port}/health", timeout=0.5) as response:  # noqa: S310
                    if response.status == 200:
                        break
            except Exception:
                if time.monotonic() >= deadline:
                    pytest.fail("OpenEnv loopback did not become ready in 30 seconds")
                time.sleep(0.1)

        imported_path = tmp_path / "openenv-rps.yaml"
        assert (
            main(
                [
                    "task",
                    "import",
                    f"openenv://127.0.0.1:{port}/arena/competitive_rps_v0",
                    "--name",
                    "task:rps-openenv-smoke@0.3",
                    "--out",
                    str(imported_path),
                    "--source-revision",
                    "openenv-0.4.1",
                ]
            )
            == 0
        )
        imported = load_manifest(imported_path)
        suite = load_manifest("examples/tasks/rps-equivalence.yaml")
        native = load_manifest("examples/tasks/native-rps.yaml")
        equivalence = verify_task_equivalence(native, imported, suite)
        assert equivalence["ok"] is True
        qualification = qualify_task_fixture(
            imported_path,
            peer=Path("examples/tasks/native-rps.yaml"),
            trace_suite=Path("examples/tasks/rps-equivalence.yaml"),
            report_path=tmp_path / "openenv-loopback-qualification.json",
        )
        assert qualification["ok"] is True
        assert qualification["adapter"] == "openenv"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_gimitest_provider_fails_loud_when_package_missing(monkeypatch) -> None:
    """If gimitest cannot be imported, decoration must raise — never silent pass."""
    import sys

    # Drop cached modules so the next import attempts a fresh load.
    for key in list(sys.modules):
        if key == "gimitest" or key.startswith("gimitest."):
            monkeypatch.delitem(sys.modules, key, raising=False)

    real_import = __import__

    def blocker(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        if name == "gimitest" or name.startswith("gimitest."):
            raise ImportError("simulated missing gimitest")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", blocker)

    from arena.adapters.eval_gimitest import decorate_env
    from arena.core.errors import ArenaError

    class _Dummy:
        pass

    with pytest.raises(ArenaError, match="Gimitest provider support is incomplete"):
        decorate_env(_Dummy(), {"test_class": "gimitest.gtest:GTest"})
