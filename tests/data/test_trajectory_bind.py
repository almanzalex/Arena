"""Prove dataset provenance bind / unbind / fail-loud mismatch behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arena.core.errors import ConformanceError, SchemaError
from arena.core.identity import digest_uri, sha256_bytes
from arena.core.manifests import DATASET_SCHEMA
from arena.dataset import (
    PROVENANCE_BINDING_SCHEMA,
    bind_dataset_provenance,
    select_bound_episodes,
    unbind_dataset_provenance,
    verify_dataset_provenance,
)

POLICY_A = digest_uri("a" * 64)
POLICY_B = digest_uri("b" * 64)
TASK = {
    "adapter": "pettingzoo-parallel",
    "env": "arena/competitive_rps_v0",
    "version": "test+pettingzoo",
}


def _write_episode(
    path: Path,
    *,
    policy_0: str,
    policy_1: str,
    seed: int = 0,
    task: dict | None = None,
) -> Path:
    payload = {
        "schema": "arena.trajectory/v0alpha1",
        "seed": seed,
        "episode_index": 0,
        "status": "completed",
        "action_mode": "deterministic",
        "task": dict(task or TASK),
        "agents": ["player_0", "player_1"],
        "role_map": {"player_0": "player_0", "player_1": "player_1"},
        "policies": {"player_0": policy_0, "player_1": policy_1},
        "returns": {"player_0": 1.0, "player_1": -1.0},
        "steps": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _dataset_from_episodes(episodes: list[Path], *, name: str = "slice") -> dict:
    entries = []
    for ep in episodes:
        raw = ep.read_bytes()
        entries.append(
            {
                "path": str(ep.resolve()),
                "digest": digest_uri(sha256_bytes(raw)),
                "seed": 0,
                "source_run": str(ep.parent.parent.resolve()),
            }
        )
    return {
        "schema": DATASET_SCHEMA,
        "name": name,
        "source_runs": [str(episodes[0].parent.parent.resolve())],
        "episodes": entries,
        "query": {},
        "lineage": {"note": "synthetic test slice"},
    }


def test_bind_unbind_roundtrip_changes_digest(tmp_path: Path) -> None:
    ep = _write_episode(
        tmp_path / "run" / "trajectories" / "episode_0000.json",
        policy_0=POLICY_A,
        policy_1=POLICY_B,
    )
    dataset = _dataset_from_episodes([ep])
    bound = bind_dataset_provenance(
        dataset,
        policy_digest=POLICY_A,
        task=TASK,
        role="player_0",
    )
    assert bound["provenance"]["schema"] == PROVENANCE_BINDING_SCHEMA
    assert bound["provenance"]["policy_digest"] == POLICY_A
    assert bound["provenance"]["task"]["env"] == TASK["env"]
    assert bound["digest"] != dataset.get("digest")
    assert "provenance" not in dataset

    verify_dataset_provenance(bound, expect_policy=POLICY_A, expect_task=TASK)

    unbound = unbind_dataset_provenance(bound)
    assert "provenance" not in unbound
    assert unbound["digest"] != bound["digest"]
    with pytest.raises(SchemaError, match="not provenance-bound"):
        verify_dataset_provenance(unbound)


def test_bind_fails_loud_on_policy_digest_mismatch(tmp_path: Path) -> None:
    ep = _write_episode(
        tmp_path / "run" / "trajectories" / "episode_0000.json",
        policy_0=POLICY_A,
        policy_1=POLICY_B,
    )
    dataset = _dataset_from_episodes([ep])
    with pytest.raises(ConformanceError, match="policy_digest_mismatch") as exc_info:
        bind_dataset_provenance(dataset, policy_digest=digest_uri("c" * 64), task=TASK)
    err = exc_info.value
    assert err.code == "DATASET_PROVENANCE_MISMATCH"
    assert err.context["mismatches"][0]["kind"] == "policy_digest_mismatch"


def test_verify_fails_loud_when_expect_policy_disagrees(tmp_path: Path) -> None:
    ep = _write_episode(
        tmp_path / "run" / "trajectories" / "episode_0000.json",
        policy_0=POLICY_A,
        policy_1=POLICY_B,
    )
    bound = bind_dataset_provenance(
        _dataset_from_episodes([ep]),
        policy_digest=POLICY_A,
        task=TASK,
    )
    with pytest.raises(ConformanceError, match="policy digest mismatch") as exc_info:
        verify_dataset_provenance(bound, expect_policy=POLICY_B)
    assert exc_info.value.code == "DATASET_POLICY_DIGEST_MISMATCH"


def test_select_bound_episodes_binds_and_rejects_mismatch(tmp_path: Path) -> None:
    run = tmp_path / "eval-run"
    _write_episode(
        run / "trajectories" / "episode_0000.json",
        policy_0=POLICY_A,
        policy_1=POLICY_B,
        seed=1,
    )
    # Sibling episode with a different primary policy — must not be selectable
    # under POLICY_A binding without failing verification if forced in.
    other = tmp_path / "other-run"
    _write_episode(
        other / "trajectories" / "episode_0000.json",
        policy_0=POLICY_B,
        policy_1=POLICY_A,
        seed=2,
    )

    selected = select_bound_episodes(
        source_runs=[run],
        query={"policy": POLICY_A, "task": TASK["env"]},
        name="bound-a",
        out_dir=tmp_path / "out-a",
        task=TASK,
        role="player_0",
    )
    assert len(selected["episodes"]) == 1
    assert selected["provenance"]["policy_digest"] == POLICY_A
    assert (tmp_path / "out-a" / "dataset.yaml").is_file()
    verify_dataset_provenance(selected, expect_policy=POLICY_A)

    with pytest.raises(ConformanceError, match="no episodes matched"):
        select_bound_episodes(
            source_runs=[other],
            policy_digest=POLICY_A,
            role="player_0",
            task=TASK,
        )

    with pytest.raises(ConformanceError, match="policy digest mismatch"):
        select_bound_episodes(
            source_runs=[run],
            query={"policy": POLICY_A},
            policy_digest=POLICY_B,
            require_policy=True,
            allow_empty=True,
        )


def test_select_bound_requires_policy_by_default(tmp_path: Path) -> None:
    run = tmp_path / "eval-run"
    _write_episode(
        run / "trajectories" / "episode_0000.json",
        policy_0=POLICY_A,
        policy_1=POLICY_B,
    )
    with pytest.raises(SchemaError, match="requires policy_digest"):
        select_bound_episodes(source_runs=[run], query={"task": TASK["env"]})
