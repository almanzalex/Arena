"""Seed protocol hardening: native Discrete seeded matches must be deterministic.

Closes the RL gap where non-deterministic eval pretends to be science: same seeds
must replay bit-identically for native Discrete envs under the RFC 001 contract
(episode seed + role salt + step index).

Expected nondeterminism (documented, not asserted here): GPU/CUDA/MPS, external
task services, ``rng=None`` stochastic fallback, and envs that ignore reset seeds.
See ``docs/seed-determinism.md`` and ``arena.runtime.seed_protocol``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from arena.core.manifests import expand_seeds
from arena.runtime.seed_protocol import policy_rng, policy_rng_seed, role_salt

pytest.importorskip("torch")
pytest.importorskip("pettingzoo")

from arena.conformance.fixtures import build_rps_policy  # noqa: E402
from arena.core.sdk import Match, Policy, Task  # noqa: E402
from arena.runtime.aec_match import run_aec_match  # noqa: E402
from arena.runtime.match import run_match  # noqa: E402

_PARALLEL = "arena/competitive_rps_v0"
_AEC = "arena/competitive_rps_aec_v0"


def _actions(run_dir: Path, i: int) -> list[dict[str, Any]]:
    ep = json.loads((run_dir / "trajectories" / f"episode_{i:04d}.json").read_text())
    return [step["actions"] for step in ep["steps"]]


def test_role_salt_differs_across_roles() -> None:
    assert role_salt("player_0") != role_salt("player_1")
    assert role_salt("player_0") == role_salt("player_0")


def test_policy_rng_seed_includes_role_salt() -> None:
    seed, step = 7, 3
    s0 = policy_rng_seed(seed, "player_0", step)
    s1 = policy_rng_seed(seed, "player_1", step)
    assert s0 == seed + role_salt("player_0") + step
    assert s1 == seed + role_salt("player_1") + step
    assert s0 != s1


def test_policy_rng_streams_diverge_by_role() -> None:
    """Same episode seed + step must not give co-agents identical draws."""
    a = policy_rng(42, "player_0", 0).integers(0, 10_000, size=8)
    b = policy_rng(42, "player_1", 0).integers(0, 10_000, size=8)
    assert not np.array_equal(a, b)
    a2 = policy_rng(42, "player_0", 0).integers(0, 10_000, size=8)
    assert np.array_equal(a, a2)


def test_expand_seeds_list_and_range() -> None:
    assert expand_seeds([3, 1, 4]) == [3, 1, 4]
    assert expand_seeds({"list": [9, 8]}) == [9, 8]
    assert expand_seeds({"start": 2, "count": 3}) == [2, 3, 4]
    assert expand_seeds({"count": 2}) == [0, 1]


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_parallel_seeded_stochastic_byte_identical(tmp_path: Path) -> None:
    """Native Parallel Discrete: same seeds → identical action streams (stochastic)."""
    p0 = build_rps_policy(tmp_path / "p0", role="player_0", seed=10)
    p1 = build_rps_policy(tmp_path / "p1", role="player_1", seed=20)
    assignments = {"player_0": Policy.load(p0), "player_1": Policy.load(p1)}
    task_spec = {
        "adapter": "pettingzoo-parallel",
        "env": _PARALLEL,
        "config": {"max_cycles": 8},
    }
    seeds = [0, 1, 2, 3, 4]
    r1 = run_match(
        task_spec=task_spec,
        assignments=assignments,
        seeds=seeds,
        action_mode="stochastic",
        record=True,
        out_dir=tmp_path / "run1",
    )
    r2 = run_match(
        task_spec=task_spec,
        assignments=assignments,
        seeds=seeds,
        action_mode="stochastic",
        record=True,
        out_dir=tmp_path / "run2",
    )
    assert r1["outcome"]["episodes_completed"] == len(seeds)
    assert r2["outcome"]["episodes_completed"] == len(seeds)
    for i in range(len(seeds)):
        assert _actions(tmp_path / "run1", i) == _actions(tmp_path / "run2", i)

    # Distinct seeds must diverge (non-vacuous reproducibility).
    r3 = run_match(
        task_spec=task_spec,
        assignments=assignments,
        seeds=[100, 101, 102, 103, 104],
        action_mode="stochastic",
        record=True,
        out_dir=tmp_path / "run3",
    )
    assert r3["outcome"]["episodes_completed"] == len(seeds)
    assert any(
        _actions(tmp_path / "run1", i) != _actions(tmp_path / "run3", i) for i in range(len(seeds))
    )


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_aec_seeded_stochastic_byte_identical(tmp_path: Path) -> None:
    """Native AEC Discrete: role salt is threaded; same seeds replay identically."""
    p0 = build_rps_policy(tmp_path / "p0", role="player_0", seed=11)
    p1 = build_rps_policy(tmp_path / "p1", role="player_1", seed=22)
    assignments = {"player_0": Policy.load(p0), "player_1": Policy.load(p1)}
    task_spec = {
        "adapter": "pettingzoo-parallel",
        "env": _AEC,
        "interaction": "aec",
        "config": {"max_cycles": 8},
    }
    seeds = [0, 1, 2, 3]
    r1 = run_aec_match(
        task_spec=task_spec,
        assignments=assignments,
        seeds=seeds,
        action_mode="stochastic",
        record=True,
        out_dir=tmp_path / "aec1",
    )
    r2 = run_aec_match(
        task_spec=task_spec,
        assignments=assignments,
        seeds=seeds,
        action_mode="stochastic",
        record=True,
        out_dir=tmp_path / "aec2",
    )
    assert r1["outcome"]["episodes_completed"] == len(seeds)
    assert r2["outcome"]["episodes_completed"] == len(seeds)
    for i in range(len(seeds)):
        assert _actions(tmp_path / "aec1", i) == _actions(tmp_path / "aec2", i)


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_sdk_match_seeded_parallel_deterministic(tmp_path: Path) -> None:
    """SDK Match facade preserves seed threading for native Parallel Discrete."""
    p0 = build_rps_policy(tmp_path / "p0", role="player_0", seed=1)
    p1 = build_rps_policy(tmp_path / "p1", role="player_1", seed=2)
    match = Match(
        task=Task.load(
            {
                "adapter": "pettingzoo-parallel",
                "env": _PARALLEL,
                "config": {"max_cycles": 4},
            }
        ),
        assignments={"player_0": Policy.load(p0), "player_1": Policy.load(p1)},
        action_mode="deterministic",
        failure_policy={"timeout_seconds": 30, "retain_incomplete": True, "retry": 0},
    )
    seeds = [5, 6, 7]
    match.run(seeds=seeds, record=True, out=tmp_path / "m1")
    match.run(seeds=seeds, record=True, out=tmp_path / "m2")
    for i in range(len(seeds)):
        assert _actions(tmp_path / "m1", i) == _actions(tmp_path / "m2", i)


def test_seed_protocol_module_documents_nondeterminism() -> None:
    import arena.runtime.seed_protocol as mod

    text = (mod.__doc__ or "").lower()
    assert "gpu" in text or "cuda" in text
    assert "external" in text
    assert "rng=none" in text or "rng is none" in text or "unseeded" in text
