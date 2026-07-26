"""Execute the 1.0 native↔OpenEnv↔Gimitest value proof on loopback."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

from arena.core.io import publish_directory
from arena.core.manifests import dump_json, load_manifest
from arena.core.tasks import import_openenv_task, verify_task_equivalence
from arena.runtime.evaluation import run_evaluation


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="./arena-1.0-boundaries")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    destination = Path(args.out).resolve()
    result: dict[str, object] = {}

    def build(stage: Path) -> None:
        port = _free_port()
        server_log = stage / "openenv-server.log"
        with server_log.open("w", encoding="utf-8") as log:
            server = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "arena.adapters.task_openenv.server",
                    "--port",
                    str(port),
                ],
                cwd=root,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        try:
            deadline = time.monotonic() + 30
            while True:
                if server.poll() is not None:
                    raise RuntimeError(
                        "OpenEnv server exited before ready: "
                        + server_log.read_text(encoding="utf-8")[-2000:]
                    )
                try:
                    with urlopen(  # noqa: S310 - fixed loopback qualification server.
                        f"http://127.0.0.1:{port}/health",
                        timeout=0.5,
                    ) as response:
                        if response.status == 200:
                            break
                except Exception:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("OpenEnv server was not ready in 30 seconds")
                    time.sleep(0.05)

            openenv_task_path = stage / "openenv-rps.yaml"
            openenv_task = import_openenv_task(
                f"openenv://127.0.0.1:{port}/arena/competitive_rps_v0",
                name="task:rps-openenv@1.0",
                out=openenv_task_path,
                source_revision="openenv-0.4.1",
            )
            native_task = load_manifest(root / "examples/tasks/native-rps.yaml")
            trace_suite = load_manifest(root / "examples/tasks/rps-equivalence.yaml")
            equivalence = verify_task_equivalence(
                native_task,
                openenv_task,
                trace_suite,
            )
            shared_task_intent = equivalence["shared_task_intent_digest"]
            policies = {
                "player_0": str(
                    (root / "examples/eval/demo/rock.arena").resolve()
                ),
                "player_1": str(
                    (root / "examples/eval/demo/paper.arena").resolve()
                ),
            }

            def suite(task: dict, *, provider: str = "native", config: dict | None = None):
                return {
                    "schema": "arena.evaluation/v0alpha1",
                    "name": "arena-1.0-boundary-proof",
                    "provider": provider,
                    "provider_config": config or {},
                    "interaction": "parallel",
                    "task": task,
                    "task_intent_digest": shared_task_intent,
                    "assignments": policies,
                    "seeds": [0, 1],
                    "action_mode": "deterministic",
                    "metrics": ["mean_return"],
                    "budgets": {
                        "executor": "process",
                        "timeout_seconds": 30,
                    },
                }

            isolated_python = os.environ.get("ARENA_GIMITEST_PYTHON")
            isolation = (
                {
                    "mode": "subprocess",
                    "python": str(Path(isolated_python)),
                    "timeout_seconds": 60,
                }
                if isolated_python
                else None
            )
            native = run_evaluation(
                suite(native_task),
                policy_index={},
                out_dir=stage / "native-run",
            )
            external = run_evaluation(
                suite(openenv_task),
                policy_index={},
                out_dir=stage / "openenv-run",
            )
            gimitest = run_evaluation(
                suite(
                    native_task,
                    provider="gimitest",
                    config={
                        "semantic": {},
                        "test_class": "gimitest.gtest:GTest",
                        "parameters": {"purpose": "provider-boundary-equivalence"},
                        **({"isolation": isolation} if isolation else {}),
                    },
                ),
                policy_index={},
                out_dir=stage / "gimitest-run",
            )
            transformed = run_evaluation(
                suite(
                    native_task,
                    provider="gimitest",
                    config={
                        "semantic": {
                            "test_class": (
                                "arena.adapters.eval_gimitest.scenarios:"
                                "RewardTransformScenario"
                            ),
                            "parameters": {"reward_scale": -1.0},
                        },
                        "test_class": (
                            "arena.adapters.eval_gimitest.scenarios:"
                            "RewardTransformScenario"
                        ),
                        "parameters": {"reward_scale": -1.0},
                        **({"isolation": isolation} if isolation else {}),
                    },
                ),
                policy_index={},
                out_dir=stage / "gimitest-non-noop-run",
            )
            assert native["state"] == external["state"] == gimitest["state"] == "complete"
            assert (
                native["evaluation_intent_digest"]
                == external["evaluation_intent_digest"]
                == gimitest["evaluation_intent_digest"]
            )
            assert (
                native["semantic_result_digest"]
                == external["semantic_result_digest"]
                == gimitest["semantic_result_digest"]
            )
            assert len(
                {
                    native["execution_binding_digest"],
                    external["execution_binding_digest"],
                    gimitest["execution_binding_digest"],
                }
            ) == 3
            assert transformed["semantic_result_digest"] != native["semantic_result_digest"]
            assert transformed["evaluation_intent_digest"] != native["evaluation_intent_digest"]
            result.update(
                {
                    "schema": "arena.local-boundary-proof/v1",
                    "ok": True,
                    "shared_task_intent_digest": shared_task_intent,
                    "evaluation_intent_digest": native[
                        "evaluation_intent_digest"
                    ],
                    "semantic_result_digest": native["semantic_result_digest"],
                    "bindings": {
                        "native": native["execution_binding_digest"],
                        "openenv": external["execution_binding_digest"],
                        "gimitest": gimitest["execution_binding_digest"],
                    },
                    "non_noop_gimitest_result_digest": transformed[
                        "semantic_result_digest"
                    ],
                    "non_noop_gimitest_intent_digest": transformed[
                        "evaluation_intent_digest"
                    ],
                    "gimitest_isolated_python": isolated_python,
                    "denominators": native["denominators"],
                }
            )
            dump_json(result, stage / "result.json")
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait(timeout=5)

    publish_directory(destination, build)
    print(json.dumps({**result, "out": str(destination)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
