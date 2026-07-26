from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("pettingzoo")

from arena.adapters.task_openenv.adapter import (
    PILOT_CONTRACT,
    PILOT_ENV,
    OpenEnvParallelEnv,
    _verify_schema_pin,
)
from arena.adapters.task_pettingzoo.pilot_env import CompetitiveRPSParallel
from arena.conformance.fixtures import build_fixed_action_rps_policy
from arena.core.errors import SchemaError, TaskRuntimeError
from arena.core.sdk import Policy
from arena.core.tasks import verify_task_equivalence
from arena.runtime.match import run_match


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
        "env": "arena/competitive_rps_v0",
        "interaction": "parallel",
        "config": {"max_cycles": 1},
    }
    suite = {
        "schema": "arena.trace-suite/v1",
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
    "base_url",
    [
        "file:///tmp/openenv",
        "http://user:secret@127.0.0.1:8000",
        "http://127.0.0.1:8000?token=secret",
        "http://127.0.0.1:8000#schema",
    ],
)
def test_schema_pin_rejects_unsafe_endpoint_urls(base_url: str) -> None:
    with pytest.raises(SchemaError, match=r"http\(s\) URL"):
        _verify_schema_pin(base_url, "sha256:" + ("0" * 64), 0.1)


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
        tmp_path / "left.arena", role=["player_0", "player_1"], action=0
    )
    right = build_fixed_action_rps_policy(
        tmp_path / "right.arena", role=["player_0", "player_1"], action=1
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
        tmp_path / "left.arena", role=["player_0", "player_1"], action=0
    )
    right = build_fixed_action_rps_policy(
        tmp_path / "right.arena", role=["player_0", "player_1"], action=1
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


def test_openenv_schema_pin_is_rechecked_in_same_process(monkeypatch) -> None:
    from arena.core.identity import canonical_json, digest_uri, sha256_bytes

    expected_schema = {"type": "object", "version": 1}
    changed_schema = {"type": "object", "version": 2}
    expected = digest_uri(sha256_bytes(canonical_json(expected_schema)))
    payloads = [expected_schema, changed_schema]

    class Response:
        status = 200

        def __init__(self, payload):
            import json

            self.payload = json.dumps(payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return self.payload

    monkeypatch.setattr(
        "arena.adapters.task_openenv.adapter.urlopen",
        lambda *args, **kwargs: Response(payloads.pop(0)),
    )
    _verify_schema_pin("http://task.invalid", expected, 1)
    with pytest.raises(TaskRuntimeError, match="schema changed"):
        _verify_schema_pin("http://task.invalid", expected, 1)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rewards", {"player_0": float("nan"), "player_1": 0.0}),
        (
            "terminations",
            {"player_0": 1, "player_1": False},
        ),
    ],
)
def test_openenv_step_rejects_nonfinite_and_ambiguous_types(field, value) -> None:
    class InvalidClient(_LoopbackClient):
        def step(self, payload):
            result = super().step(payload)
            result.observation[field] = value
            return result

    env = OpenEnvParallelEnv(_openenv_spec(lambda spec: InvalidClient()))
    try:
        env.reset(seed=0)
        with pytest.raises(TaskRuntimeError) as exc:
            env.step({"player_0": 0, "player_1": 1})
        assert exc.value.kind == "protocol_error"
    finally:
        env.close()


def test_openenv_close_failure_is_retained_as_diagnostic() -> None:
    class BadClose(_LoopbackClient):
        def __exit__(self, *args):
            raise ConnectionError("late close failure")

    env = OpenEnvParallelEnv(_openenv_spec(lambda spec: BadClose()))
    with pytest.warns(ResourceWarning, match="close failed"):
        env.close()
    assert env.cleanup_diagnostics[0]["code"] == "OPENENV_CLOSE_FAILED"
