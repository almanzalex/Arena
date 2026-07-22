from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

import pytest

pytest.importorskip("openenv")
pytest.importorskip("pettingzoo")

from rlx.cli.main import main
from rlx.conformance.qualification import qualify_task_fixture
from rlx.core.manifests import load_manifest
from rlx.core.tasks import verify_task_equivalence
from rlx.runtime.evaluation import run_evaluation


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.mark.acceptance
@pytest.mark.requires_openenv
def test_t01_t02_real_openenv_transport_equivalence(tmp_path: Path) -> None:
    port = _free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "rlx.adapters.task_openenv.server",
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
                pytest.fail(f"OpenEnv pilot exited before ready: {output}")
            try:
                with urlopen(f"http://127.0.0.1:{port}/health", timeout=0.5) as response:  # noqa: S310
                    if response.status == 200:
                        break
            except Exception:
                if time.monotonic() >= deadline:
                    pytest.fail("OpenEnv pilot did not become ready in 30 seconds")
                time.sleep(0.1)

        imported_path = tmp_path / "openenv-rps.yaml"
        assert main(
            [
                "task",
                "import",
                f"openenv://127.0.0.1:{port}/rlx/competitive_rps_v0",
                "--name",
                "task:rps-openenv@0.3",
                "--out",
                str(imported_path),
                "--source-revision",
                "openenv-0.4.1",
            ]
        ) == 0
        imported = load_manifest(imported_path)
        suite = load_manifest("examples/tasks/rps-equivalence.yaml")
        native = load_manifest("examples/tasks/native-rps.yaml")
        result = verify_task_equivalence(native, imported, suite)
        assert result["ok"] is True
        assert result["diffs"] == []
        assert imported["packaging"]["schema_digest"].startswith("sha256:")
        assert main(
            [
                "task",
                "verify-equivalence",
                "examples/tasks/native-rps.yaml",
                str(imported_path),
                "--trace-suite",
                "examples/tasks/rps-equivalence.yaml",
            ]
        ) == 0
        qualification = qualify_task_fixture(
            imported_path,
            peer=Path("examples/tasks/native-rps.yaml"),
            trace_suite=Path("examples/tasks/rps-equivalence.yaml"),
            report_path=tmp_path / "qualification.json",
        )
        assert qualification["ok"] is True
        assert qualification["adapter"] == "openenv"

        policies = {
            "player_0": str(Path("examples/eval/demo/rock.rlx").resolve()),
            "player_1": str(Path("examples/eval/demo/paper.rlx").resolve()),
        }

        def evaluate(task, out):
            return run_evaluation(
                {
                    "schema": "rlx.evaluation/v0alpha1",
                    "name": "same-policy-native-openenv",
                    "provider": "native",
                    "interaction": "parallel",
                    "task": task,
                    "assignments": policies,
                    "seeds": [0],
                    "action_mode": "deterministic",
                    "metrics": ["mean_return"],
                },
                policy_index={},
                out_dir=out,
            )

        native_eval = evaluate(native, tmp_path / "native-eval")
        openenv_eval = evaluate(imported, tmp_path / "openenv-eval")
        assert native_eval["cells"][0]["assignments"] == openenv_eval["cells"][0]["assignments"]
        assert (
            native_eval["cell_results"][0]["episodes"][0]["returns"]
            == openenv_eval["cell_results"][0]["episodes"][0]["returns"]
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
