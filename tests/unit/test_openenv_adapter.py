from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("pettingzoo")

from rlx.adapters.task_openenv.adapter import (
    PILOT_CONTRACT,
    PILOT_ENV,
    OpenEnvParallelEnv,
)
from rlx.adapters.task_pettingzoo.pilot_env import CompetitiveRPSParallel
from rlx.conformance.fixtures import build_fixed_action_rps_policy
from rlx.core.errors import TaskRuntimeError
from rlx.core.sdk import Policy
from rlx.core.tasks import verify_task_equivalence
from rlx.runtime.match import run_match


class _LoopbackClient:
    def __init__(self) -> None:
        self.env = CompetitiveRPSParallel(max_cycles=1)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.env.close()

    def reset(self, seed=None):
        observations, infos = self.env.reset(seed=seed)
        return SimpleNamespace(observation={"observations": observations, "infos": infos})

    def step(self, payload):
        observations, rewards, terms, truncs, infos = self.env.step(payload["actions"])
        return SimpleNamespace(
            observation={
                "observations": observations,
                "rewards": rewards,
                "terminations": terms,
                "truncations": truncs,
                "infos": infos,
            }
        )


def _openenv_spec(factory=lambda spec: _LoopbackClient()):
    return {
        "adapter": "openenv",
        "env": PILOT_ENV,
        "interaction": "parallel",
        "packaging": {"kind": "openenv", "_client_factory": factory},
        "contract": PILOT_CONTRACT,
    }


def test_t01_native_openenv_trace_equivalence() -> None:
    native = {
        "adapter": "pettingzoo-parallel",
        "env": "rlx/competitive_rps_v0",
        "interaction": "parallel",
        "config": {"max_cycles": 1},
    }
    suite = {
        "schema": "rlx.trace-suite/v1",
        "interaction": "parallel",
        "episodes": [
            {"seed": 0, "actions": [{"player_0": 0, "player_1": 1}]},
            {"seed": 9, "actions": [{"player_0": 2, "player_1": 1}]},
        ],
        "tolerances": {"default": 0.0},
    }
    result = verify_task_equivalence(native, _openenv_spec(), suite)
    assert result["ok"] is True
    assert result["episodes"] == 2


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (TimeoutError("late"), "timeout"),
        (ConnectionError("closed"), "disconnect"),
        (RuntimeError("container execution_error"), "container_crash"),
        (ValueError("invalid response"), "protocol_error"),
    ],
)
def test_t03_openenv_failure_semantics(error: Exception, kind: str) -> None:
    class BrokenClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def reset(self, seed=None):
            raise error

    env = OpenEnvParallelEnv(_openenv_spec(lambda spec: BrokenClient()))
    with pytest.raises(TaskRuntimeError) as exc:
        env.reset(seed=0)
    assert exc.value.kind == kind
    assert exc.value.details["operation"] == "reset"


def test_t03_transport_failure_is_recorded_on_run(tmp_path) -> None:
    class TimedOutClient:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def reset(self, seed=None):
            raise TimeoutError("remote deadline")

    left = build_fixed_action_rps_policy(
        tmp_path / "left.rlx", role=["player_0", "player_1"], action=0
    )
    right = build_fixed_action_rps_policy(
        tmp_path / "right.rlx", role=["player_0", "player_1"], action=1
    )
    result = run_match(
        task_spec=_openenv_spec(lambda spec: TimedOutClient()),
        assignments={"player_0": Policy.load(left), "player_1": Policy.load(right)},
        seeds=[0],
        out_dir=tmp_path / "run",
    )
    assert result["outcome"]["episodes_completed"] == 0
    assert result["outcome"]["failure_count"] == 1
    assert result["failures"][0]["kind"] == "timeout"
    assert result["failures"][0]["details"]["operation"] == "reset"


def test_t03_connect_failure_is_recorded_as_disconnect(tmp_path) -> None:
    def cannot_connect(spec):
        del spec
        raise ConnectionError("connection refused")

    left = build_fixed_action_rps_policy(
        tmp_path / "left.rlx", role=["player_0", "player_1"], action=0
    )
    right = build_fixed_action_rps_policy(
        tmp_path / "right.rlx", role=["player_0", "player_1"], action=1
    )
    result = run_match(
        task_spec=_openenv_spec(cannot_connect),
        assignments={"player_0": Policy.load(left), "player_1": Policy.load(right)},
        seeds=[0],
        out_dir=tmp_path / "run",
    )
    assert result["outcome"]["failure_count"] == 1
    assert result["failures"][0]["kind"] == "disconnect"
    assert result["failures"][0]["details"]["operation"] == "connect"
