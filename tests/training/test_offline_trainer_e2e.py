"""CPU-only offline trainer hardening: real RPS rollouts → train → resume → eval."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("pettingzoo")

# Keep the suite on CPU even when a GPU is present.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from arena.adapters.policy_custom_torch import load_runtime, verify_bundle_self
from arena.conformance.fixtures import build_fixed_action_rps_policy, build_rps_policy
from arena.core.dataset import materialize_dataset, select_episodes
from arena.core.manifests import dump_yaml, load_manifest
from arena.core.sdk import Match, Policy, Task
from arena.runtime.evaluation import build_eval_report, run_evaluation
from arena.runtime.training import run_training_recipe

_PILOT = "arena/competitive_rps_v0"
_MAX_CYCLES = 4
_ROLLOUT_SEEDS = [0, 1, 2, 3, 4, 5]


def _teacher_run(tmp_path: Path) -> Path:
    """Record a tiny parallel RPS match between fixture teachers."""
    p0 = build_rps_policy(tmp_path / "teacher0", role="player_0", seed=11)
    p1 = build_rps_policy(tmp_path / "teacher1", role="player_1", seed=23)
    match = Match(
        task=Task.load(
            {
                "adapter": "pettingzoo-parallel",
                "env": _PILOT,
                "config": {"max_cycles": _MAX_CYCLES},
            }
        ),
        assignments={
            "player_0": Policy.load(p0),
            "player_1": Policy.load(p1),
        },
        action_mode="deterministic",
        failure_policy={"timeout_seconds": 30, "retain_incomplete": True, "retry": 0},
    )
    out = tmp_path / "teacher-run"
    result = match.run(seeds=list(_ROLLOUT_SEEDS), record=True, out=out)
    assert result["outcome"]["episodes_completed"] == len(_ROLLOUT_SEEDS)
    assert (out / "trajectories").is_dir()
    return out


def _portable_dataset(tmp_path: Path, source_run: Path) -> Path:
    selected_dir = tmp_path / "selected"
    selected = select_episodes(
        source_runs=[source_run],
        query={"role": "player_0"},
        name="rps-teacher-player0",
        out_dir=selected_dir,
    )
    assert len(selected["episodes"]) == len(_ROLLOUT_SEEDS)
    portable_dir = tmp_path / "portable-dataset"
    portable = materialize_dataset(selected_dir / "dataset.yaml", out_dir=portable_dir)
    assert portable["lineage"]["materialized"] is True
    return portable_dir / "dataset.yaml"


def _recipe(
    path: Path,
    dataset: Path,
    *,
    algorithm: str = "behavior_cloning",
    algorithm_config: dict[str, Any] | None = None,
    epochs: int = 16,
    seed: int = 17,
) -> Path:
    dump_yaml(
        {
            "schema": "arena.train/v1",
            "name": "offline-trainer-e2e",
            "algorithm": algorithm,
            "algorithm_config": algorithm_config or {},
            "dataset": str(dataset),
            "role": "player_0",
            "roles": ["player_0", "player_1"],
            "seed": seed,
            "epochs": epochs,
            "batch_size": 8,
            "learning_rate": 0.05,
            "observation": {"type": "Discrete", "n": 4, "dtype": "int64"},
            "action": {
                "type": "Discrete",
                "n": 3,
                "dtype": "int64",
                "masks": "none",
            },
            "architecture": {
                "type": "mlp_categorical",
                "observation_dim": 4,
                "hidden_dims": [],
                "action_n": 3,
            },
            "preprocessing": {"id": "normalize_v0", "mean": 0.0, "std": 1.0},
        },
        path,
    )
    return path


def _eval_vs_rock(tmp_path: Path, policy_bundle: Path) -> dict[str, Any]:
    candidate = Policy.load(policy_bundle)
    rock = build_fixed_action_rps_policy(
        tmp_path / "rock.arena",
        role="player_1",
        action=0,
        name="rock",
    )
    rock_pol = Policy.load(rock)
    suite = {
        "schema": "arena.evaluation/v0alpha1",
        "name": "trained-vs-rock",
        "interaction": "parallel",
        "task": {
            "adapter": "pettingzoo-parallel",
            "env": _PILOT,
            "config": {"max_cycles": 1},
        },
        "assignments": {
            "player_0": {"kind": "policy", "policy": candidate.digest},
            "player_1": {"kind": "policy", "policy": rock_pol.digest},
        },
        "seeds": {"start": 0, "count": 2},
        "action_mode": "deterministic",
        "metrics": ["mean_return"],
        "budgets": {"timeout_seconds": 30},
    }
    result = run_evaluation(
        suite,
        policy_index={
            candidate.digest: Path(policy_bundle),
            rock_pol.digest: Path(rock),
        },
        out_dir=tmp_path / "eval-vs-rock",
        workers=1,
        record=False,
    )
    assert result.get("state", "complete") == "complete"
    report = build_eval_report(result)
    assert "mean_return" in report["metrics"]
    mean = report["metrics"]["mean_return"]
    assert "mean" in mean or "n" in mean or isinstance(mean, dict)
    lineage_digests: set[str] = set()
    for cell in result.get("cells") or []:
        lineage_digests.update((cell.get("lineage") or {}).get("policy_digests") or [])
    assert candidate.digest in lineage_digests
    return report


@pytest.mark.slow
@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_offline_trainer_env_rollout_train_verify_eval(tmp_path: Path) -> None:
    """Real env rollouts → select → materialize → BC train → verify → short eval."""
    source_run = _teacher_run(tmp_path)
    dataset = _portable_dataset(tmp_path, source_run)
    recipe = _recipe(tmp_path / "recipe.yaml", dataset, epochs=16)
    result = run_training_recipe(recipe, out_dir=tmp_path / "train-run")

    assert result["schema"] == "arena.train-run/v1"
    assert result["algorithm"] == "behavior_cloning"
    assert result["examples"] == len(_ROLLOUT_SEEDS) * _MAX_CYCLES
    assert result["loss"]["final"] <= result["loss"]["initial"]

    bundle = tmp_path / "train-run" / "policy.arena"
    policy = Policy.load(bundle)
    assert result["output_policy"]["digest"] == policy.digest
    assert verify_bundle_self(bundle)["verify_mode"] == "source-conformance"
    runtime = load_runtime(bundle)
    for obs in range(4):
        action = runtime.act(obs)
        assert action in {0, 1, 2}

    _eval_vs_rock(tmp_path, bundle)


@pytest.mark.slow
@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_offline_trainer_resume_matches_uninterrupted(tmp_path: Path) -> None:
    """Resumed training matches an uninterrupted run (digest + loss epochs)."""
    source_run = _teacher_run(tmp_path)
    dataset = _portable_dataset(tmp_path, source_run)

    first = run_training_recipe(
        _recipe(tmp_path / "first.yaml", dataset, epochs=8),
        out_dir=tmp_path / "first",
    )
    assert first["checkpoint"]["epochs_completed"] == 8

    full = run_training_recipe(
        _recipe(tmp_path / "full.yaml", dataset, epochs=16),
        out_dir=tmp_path / "full",
    )

    resumed_recipe = _recipe(tmp_path / "resumed.yaml", dataset, epochs=16)
    resumed_data = load_manifest(resumed_recipe)
    resumed_data["resume_from"] = str(tmp_path / "first")
    dump_yaml(resumed_data, resumed_recipe)
    resumed = run_training_recipe(resumed_recipe, out_dir=tmp_path / "resumed")

    assert resumed["start_epoch"] == first["checkpoint"]["epochs_completed"]
    assert resumed["output_policy"]["digest"] == full["output_policy"]["digest"]
    assert resumed["loss"]["epochs"] == full["loss"]["epochs"]


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_return_weighted_from_env_rollouts_is_seeded(tmp_path: Path) -> None:
    """Return-weighted regression stays distinct and reproducible on env data."""
    source_run = _teacher_run(tmp_path)
    dataset = _portable_dataset(tmp_path, source_run)
    recipe = _recipe(
        tmp_path / "weighted.yaml",
        dataset,
        algorithm="return_weighted_regression",
        algorithm_config={"temperature": 0.5, "max_weight": 20.0},
        epochs=8,
        seed=31,
    )
    first = run_training_recipe(recipe, out_dir=tmp_path / "weighted-a")
    second = run_training_recipe(recipe, out_dir=tmp_path / "weighted-b")
    assert first["algorithm"] == "return_weighted_regression"
    assert first["sample_weights"]["max"] >= first["sample_weights"]["min"]
    assert first["output_policy"]["digest"] == second["output_policy"]["digest"]
    assert verify_bundle_self(tmp_path / "weighted-a" / "policy.arena")["verify_mode"] == (
        "source-conformance"
    )
