"""Spike coverage for streaming reads and sharded materialize (RFC 012)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arena.core.dataset import materialize_dataset, select_episodes
from arena.core.errors import ConformanceError, SchemaError
from arena.core.identity import digest_uri, sha256_bytes
from arena.core.manifests import load_manifest
from arena.dataset import (
    SHARD_METHOD,
    iter_verified_episodes,
    materialize_dataset_sharded,
    shard_id_for_index,
)


def _episode_payload(*, seed: int) -> dict:
    return {
        "schema": "arena.trajectory/v0alpha1",
        "seed": seed,
        "task": {"env": "arena/competitive_rps_v0"},
        "agents": ["player_0", "player_1"],
        "role_map": {"player_0": "player_0", "player_1": "player_1"},
        "policies": {},
        "returns": {"player_0": 1.0, "player_1": -1.0},
        "steps": [
            {
                "observations": {"player_0": observation, "player_1": 0},
                "actions": {"player_0": 1, "player_1": 0},
                "rewards": {"player_0": 1.0, "player_1": -1.0},
                "terminations": {"player_0": False, "player_1": False},
                "truncations": {"player_0": False, "player_1": False},
            }
            for observation in (0, 1, 2, 3)
        ],
    }


def _source_run(tmp_path: Path, n: int = 6) -> Path:
    run = tmp_path / "source-run"
    trajectories = run / "trajectories"
    trajectories.mkdir(parents=True)
    for index in range(n):
        episode = _episode_payload(seed=index)
        (trajectories / f"episode_{index:04d}.json").write_text(
            json.dumps(episode, sort_keys=True),
            encoding="utf-8",
        )
    return run


def _select(tmp_path: Path, n: int = 6) -> Path:
    source = _source_run(tmp_path, n=n)
    selected_dir = tmp_path / "selected"
    select_episodes(
        source_runs=[source],
        query={},
        name="stream-spike",
        out_dir=selected_dir,
    )
    return selected_dir / "dataset.yaml"


def test_shard_id_index_mod_is_deterministic() -> None:
    assert [shard_id_for_index(i, 3) for i in range(6)] == [0, 1, 2, 0, 1, 2]
    with pytest.raises(SchemaError):
        shard_id_for_index(0, 0)


def test_stream_read_verifies_digests_without_copy(tmp_path: Path) -> None:
    manifest = _select(tmp_path, n=4)
    selected = load_manifest(manifest)
    streamed = list(iter_verified_episodes(manifest))
    assert len(streamed) == 4
    for index, entry, episode in streamed:
        assert entry["digest"] == selected["episodes"][index]["digest"]
        assert episode["seed"] == selected["episodes"][index]["seed"]
        # No portable tree required for stream-read of absolute select paths.
        assert Path(entry["path"]).is_absolute()


def test_stream_read_fail_loud_on_mutation(tmp_path: Path) -> None:
    manifest = _select(tmp_path, n=2)
    selected = load_manifest(manifest)
    victim = Path(selected["episodes"][0]["path"])
    payload = json.loads(victim.read_text(encoding="utf-8"))
    payload["seed"] = 999
    victim.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with pytest.raises(ConformanceError, match="digest mismatch"):
        list(iter_verified_episodes(manifest))


def test_stream_read_split_filter_on_sharded_tree(tmp_path: Path) -> None:
    manifest = _select(tmp_path, n=6)
    portable = materialize_dataset_sharded(
        manifest,
        out_dir=tmp_path / "sharded",
        shard_count=3,
        splits={"train": 0.5, "validation": 0.5},
        split_seed=7,
    )
    train = list(
        iter_verified_episodes(
            tmp_path / "sharded" / "dataset.yaml",
            split="train",
        )
    )
    assert train
    assert all(entry.get("split") == "train" for _, entry, _ in train)
    assert sum(1 for _ in train) == portable["splits"]["counts"]["train"]


def test_sharded_materialize_determinism_across_two_runs(tmp_path: Path) -> None:
    manifest = _select(tmp_path, n=6)
    first = materialize_dataset_sharded(
        manifest,
        out_dir=tmp_path / "portable-a",
        shard_count=3,
        splits={"train": 0.7, "validation": 0.3},
        split_seed=11,
    )
    second = materialize_dataset_sharded(
        manifest,
        out_dir=tmp_path / "portable-b",
        shard_count=3,
        splits={"train": 0.7, "validation": 0.3},
        split_seed=11,
    )
    assert first["digest"] == second["digest"]
    assert first["lineage"]["shard_method"] == SHARD_METHOD
    assert first["lineage"]["shard_count"] == 3
    assert first["lineage"]["sharded"] is True
    assert [e["shard"] for e in first["episodes"]] == [0, 1, 2, 0, 1, 2]
    assert [e["path"] for e in first["episodes"]] == [e["path"] for e in second["episodes"]]
    assert [e["digest"] for e in first["episodes"]] == [e["digest"] for e in second["episodes"]]
    assert [e["split"] for e in first["episodes"]] == [e["split"] for e in second["episodes"]]

    for name in ("portable-a", "portable-b"):
        on_disk = load_manifest(tmp_path / name / "dataset.yaml")
        assert on_disk["digest"] == first["digest"]
        for index, entry in enumerate(on_disk["episodes"]):
            path = tmp_path / name / entry["path"]
            assert path.is_file()
            assert entry["digest"] == digest_uri(sha256_bytes(path.read_bytes()))
            assert entry["shard"] == index % 3
            assert str(entry["path"]).startswith(f"episodes/shard_{index % 3:04d}/")


def test_sharded_and_flat_materialize_digests_differ_by_design(tmp_path: Path) -> None:
    """Explicit non-goal: sharded layout must not claim flat content identity."""
    manifest = _select(tmp_path, n=4)
    flat = materialize_dataset(manifest, out_dir=tmp_path / "flat")
    sharded = materialize_dataset_sharded(
        manifest,
        out_dir=tmp_path / "sharded",
        shard_count=2,
    )
    assert flat["digest"] != sharded["digest"]
    assert flat["lineage"].get("sharded") is None
    assert sharded["lineage"]["sharded"] is True
    # Episode *content* digests still match by index.
    assert [e["digest"] for e in flat["episodes"]] == [
        e["digest"] for e in sharded["episodes"]
    ]


def test_flat_atomic_materialize_unchanged_by_shard_spike(tmp_path: Path) -> None:
    """Regression guard: core flat materialize stays non-sharded and complete."""
    manifest = _select(tmp_path, n=3)
    portable = materialize_dataset(manifest, out_dir=tmp_path / "portable")
    assert portable["lineage"]["materialized"] is True
    assert "sharded" not in portable["lineage"]
    assert portable["episodes"][0]["path"] == "episodes/episode_000000.json"
    assert (tmp_path / "portable" / "dataset.yaml").is_file()
    assert (tmp_path / "portable" / "dataset.json").is_file()
    assert (tmp_path / "portable" / "episodes" / "episode_000000.json").is_file()
    assert not (tmp_path / "portable" / "episodes" / "shard_0000").exists()
