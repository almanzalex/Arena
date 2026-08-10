"""PR-scale soak: materialize determinism + no partial final artifacts."""

from __future__ import annotations

import errno
import json
import shutil
from pathlib import Path

import pytest

from arena.core.dataset import (
    _looks_like_complete_dataset,
    materialize_dataset,
    select_episodes,
)
from arena.core.manifests import load_manifest

SOAK_N = 80
SPLIT_SEED = 42
SPLIT_SPEC = {"train": 0.7, "validation": 0.2, "test": 0.1}


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


def _source_run_with_n(tmp_path: Path, n: int) -> Path:
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


def _sibling_staging_dirs(parent: Path, *, final_name: str) -> list[Path]:
    return [
        path
        for path in parent.iterdir()
        if path.is_dir()
        and path.name != final_name
        and (
            path.name.startswith(f".{final_name}.")
            or path.name.startswith(".arena-dataset-")
        )
    ]


def _assert_no_valid_partial_final(out_dir: Path) -> None:
    assert not _looks_like_complete_dataset(out_dir), (
        f"valid-looking partial final dataset left at {out_dir}"
    )
    if out_dir.exists():
        # An empty placeholder is acceptable; a half-written tree is not.
        if out_dir.is_dir():
            leftovers = list(out_dir.rglob("*"))
            assert leftovers == [], f"partial contents under final out_dir: {leftovers}"
        else:
            pytest.fail(f"final path exists as non-directory: {out_dir}")
    staging = _sibling_staging_dirs(out_dir.parent, final_name=out_dir.name)
    assert staging == [], f"leftover staging directories after failure: {staging}"


@pytest.mark.acceptance
def test_materialize_soak_determinism_across_two_runs(tmp_path: Path) -> None:
    source_run = _source_run_with_n(tmp_path, SOAK_N)
    selected_dir = tmp_path / "selected"
    selected = select_episodes(
        source_runs=[source_run],
        query={},
        name="soak-slice",
        out_dir=selected_dir,
    )
    assert len(selected["episodes"]) == SOAK_N

    first = materialize_dataset(
        selected_dir / "dataset.yaml",
        out_dir=tmp_path / "portable-a",
        splits=SPLIT_SPEC,
        split_seed=SPLIT_SEED,
    )
    second = materialize_dataset(
        selected_dir / "dataset.yaml",
        out_dir=tmp_path / "portable-b",
        splits=SPLIT_SPEC,
        split_seed=SPLIT_SEED,
    )

    assert first["digest"] == second["digest"]
    assert first["splits"] == second["splits"]
    assert [entry["split"] for entry in first["episodes"]] == [
        entry["split"] for entry in second["episodes"]
    ]
    assert [entry["digest"] for entry in first["episodes"]] == [
        entry["digest"] for entry in second["episodes"]
    ]
    assert sum(first["splits"]["counts"].values()) == SOAK_N
    assert first["splits"]["method"] == "sha256_bucket/v1"
    assert first["splits"]["seed"] == SPLIT_SEED

    # On-disk manifests match returned digests (publication surface is complete).
    for name in ("portable-a", "portable-b"):
        on_disk = load_manifest(tmp_path / name / "dataset.yaml")
        assert on_disk["digest"] == first["digest"]
        assert _looks_like_complete_dataset(tmp_path / name)


@pytest.mark.acceptance
def test_materialize_soak_enospace_leaves_no_partial_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_run = _source_run_with_n(tmp_path, SOAK_N)
    selected_dir = tmp_path / "selected"
    select_episodes(source_runs=[source_run], query={}, out_dir=selected_dir)

    out_dir = tmp_path / "portable-enospace"
    real_copy2 = shutil.copy2
    calls = {"n": 0}

    def boom_copy2(src, dst, *args, **kwargs):  # noqa: ANN001
        calls["n"] += 1
        if calls["n"] >= max(3, SOAK_N // 4):
            raise OSError(errno.ENOSPC, "No space left on device (simulated)")
        return real_copy2(src, dst, *args, **kwargs)

    monkeypatch.setattr(shutil, "copy2", boom_copy2)
    with pytest.raises(OSError) as excinfo:
        materialize_dataset(
            selected_dir / "dataset.yaml",
            out_dir=out_dir,
            splits=SPLIT_SPEC,
            split_seed=SPLIT_SEED,
        )
    assert excinfo.value.errno == errno.ENOSPC
    _assert_no_valid_partial_final(out_dir)

    # Recovery: a subsequent materialize into the same path still succeeds.
    monkeypatch.setattr(shutil, "copy2", real_copy2)
    recovered = materialize_dataset(
        selected_dir / "dataset.yaml",
        out_dir=out_dir,
        splits=SPLIT_SPEC,
        split_seed=SPLIT_SEED,
    )
    assert _looks_like_complete_dataset(out_dir)
    assert recovered["lineage"]["episode_count"] == SOAK_N


@pytest.mark.acceptance
def test_materialize_soak_keyboard_interrupt_leaves_no_partial_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_run = _source_run_with_n(tmp_path, SOAK_N)
    selected_dir = tmp_path / "selected"
    select_episodes(source_runs=[source_run], query={}, out_dir=selected_dir)

    out_dir = tmp_path / "portable-interrupt"
    real_copy2 = shutil.copy2
    calls = {"n": 0}

    def interrupt_copy2(src, dst, *args, **kwargs):  # noqa: ANN001
        calls["n"] += 1
        if calls["n"] >= max(2, SOAK_N // 5):
            raise KeyboardInterrupt()
        return real_copy2(src, dst, *args, **kwargs)

    monkeypatch.setattr(shutil, "copy2", interrupt_copy2)
    with pytest.raises(KeyboardInterrupt):
        materialize_dataset(
            selected_dir / "dataset.yaml",
            out_dir=out_dir,
            splits=SPLIT_SPEC,
            split_seed=SPLIT_SEED,
        )
    _assert_no_valid_partial_final(out_dir)
