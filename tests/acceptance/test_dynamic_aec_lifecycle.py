from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("pettingzoo")

from arena.conformance.fixtures import build_fixed_action_rps_policy
from arena.conformance.qualification import qualify_task_fixture
from arena.core.errors import CompatibilityError
from arena.core.sdk import Match, Policy, Task
from arena.runtime.evaluation import run_evaluation


def _policy(tmp_path: Path) -> Policy:
    bundle = build_fixed_action_rps_policy(
        tmp_path / "dynamic.arena",
        role=["agent_0", "agent_1", "agent_2"],
        action=0,
        name="dynamic-fixed-action",
    )
    return Policy.load(bundle)


def _task(policy: Policy) -> dict:
    return {
        "adapter": "pettingzoo-parallel",
        "env": "arena/dynamic_lineup_aec_v0",
        "interaction": "dynamic_aec",
        "lifecycle": {"birth_eligibility": {"agent_2": [policy.digest]}},
    }


def _role_policy(tmp_path: Path) -> Policy:
    bundle = build_fixed_action_rps_policy(
        tmp_path / "role-dynamic.arena",
        role=["contestant"],
        action=0,
        name="dynamic-role-policy",
    )
    return Policy.load(bundle)


@pytest.mark.acceptance
def test_dynamic_birth_removal_match_and_eval(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    assignments = {agent: policy for agent in ("agent_0", "agent_1", "agent_2")}
    task = _task(policy)

    result = Match(task=Task.load(task), assignments=assignments).run(
        seeds=[11], out=tmp_path / "match"
    )
    assert result["outcome"] == {
        "episodes_requested": 1,
        "episodes_completed": 1,
        "failure_count": 0,
    }
    assert result["episodes"][0]["lifecycle_events"] == 4

    episode = json.loads(
        (tmp_path / "match" / "trajectories" / "episode_0000.json").read_text()
    )
    assert episode["interaction"] == "dynamic_aec"
    assert episode["initial_agents"] == ["agent_0", "agent_1"]
    assert episode["steps"][1]["join_events"] == ["agent_2"]
    assert episode["steps"][1]["leave_events"] == ["agent_0"]
    assert episode["agent_segments"]["agent_0"]["left_step"] == 1
    assert episode["agent_segments"]["agent_2"]["origin"] == "birth"
    assert episode["agent_segments"]["agent_2"]["left_step"] == 3
    assert all("agents_alive" in step for step in episode["steps"])

    suite = {
        "schema": "arena.evaluation/v0alpha1",
        "name": "dynamic-lineup-eval",
        "provider": "native",
        "interaction": "dynamic_aec",
        "task": task,
        "assignments": {agent: str(policy.root) for agent in assignments},
        "seeds": [11],
        "action_mode": "deterministic",
        "metrics": ["mean_return"],
    }
    evaluated = run_evaluation(suite, policy_index={}, out_dir=tmp_path / "eval")
    assert evaluated["cells"][0]["failures"] == 0

    qualification = qualify_task_fixture(
        "examples/tasks/dynamic-lineup.yaml",
        peer=None,
        trace_suite="examples/tasks/dynamic-lineup-trace.yaml",
        report_path=tmp_path / "dynamic-qualification.json",
    )
    assert qualification["ok"] is True
    assert qualification["adapter"] == "pettingzoo-parallel"
    assert qualification["checks"]["immutable_contract"]["agents"] == [
        "agent_0",
        "agent_1",
    ]


def test_dynamic_birth_requires_digest_eligibility(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    assignments = {agent: policy for agent in ("agent_0", "agent_1", "agent_2")}
    task = _task(policy)
    task["lifecycle"]["birth_eligibility"]["agent_2"] = ["sha256:" + "0" * 64]
    with pytest.raises(CompatibilityError, match="birth_eligibility.agent_2"):
        Match(task=Task.load(task), assignments=assignments).run(
            seeds=[0], out=tmp_path / "refused"
        )
    assert not (tmp_path / "refused").exists()


@pytest.mark.acceptance
def test_role_resolver_records_rejoin_as_a_new_agent_segment(
    tmp_path: Path,
) -> None:
    policy = _role_policy(tmp_path)
    task = {
        "adapter": "pettingzoo-parallel",
        "env": "arena/dynamic_reentry_aec_v0",
        "interaction": "dynamic_aec",
        "lifecycle": {
            "resolver": {
                "kind": "role",
                "agent_roles": {
                    "agent_0": "contestant",
                    "agent_1": "contestant",
                    "agent_2": "contestant",
                },
                "join_eligibility": {"contestant": [policy.digest]},
            }
        },
    }
    result = Match(
        task=Task.load(task),
        assignments={"contestant": policy},
    ).run(seeds=[5], out=tmp_path / "reentry")
    assert result["outcome"] == {
        "episodes_requested": 1,
        "episodes_completed": 1,
        "failure_count": 0,
    }
    assert result["lifecycle_resolver"] == "role"
    assert set(result["resolved_agents"]) == {"agent_0", "agent_1", "agent_2"}
    episode = json.loads(
        (tmp_path / "reentry" / "trajectories" / "episode_0000.json").read_text()
    )
    assert [segment["origin"] for segment in episode["agent_segment_history"]["agent_0"]] == [
        "reset",
        "rejoin",
    ]
    assert episode["agent_segment_history"]["agent_0"][0]["left_step"] == 0
    assert episode["agent_segment_history"]["agent_0"][1]["joined_step"] == 1
    assert episode["agent_segment_history"]["agent_2"][0]["origin"] == "birth"
    details = episode["steps"][1]["join_event_details"]
    assert [(item["agent"], item["origin"]) for item in details] == [
        ("agent_0", "rejoin"),
        ("agent_2", "birth"),
    ]
    assert all(item["assignment_key"] == "contestant" for item in details)
