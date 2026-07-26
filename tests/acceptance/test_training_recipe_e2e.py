from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from arena.adapters.policy_custom_torch import load_runtime, verify_bundle_self
from arena.cli.main import main
from arena.core.dataset import materialize_dataset, select_episodes
from arena.core.errors import ConformanceError, SchemaError
from arena.core.manifests import dump_yaml, load_manifest
from arena.core.sdk import Policy
from arena.runtime.training import run_training_recipe


def _source_run(tmp_path: Path) -> Path:
    run = tmp_path / "source-run"
    trajectories = run / "trajectories"
    trajectories.mkdir(parents=True)
    episode = {
        "schema": "arena.trajectory/v0alpha1",
        "seed": 7,
        "task": {"env": "arena/competitive_rps_v0"},
        "agents": ["player_0", "player_1"],
        "role_map": {"player_0": "player_0", "player_1": "player_1"},
        "policies": {},
        "returns": {"player_0": 4.0, "player_1": -4.0},
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
    (trajectories / "episode_0000.json").write_text(json.dumps(episode), encoding="utf-8")
    return run


def _recipe(
    path: Path,
    dataset: Path,
    *,
    algorithm: str = "behavior_cloning",
    algorithm_config: dict | None = None,
    epochs: int = 60,
) -> Path:
    dump_yaml(
        {
            "schema": "arena.train/v1",
            "name": "imitate-paper",
            "algorithm": algorithm,
            "algorithm_config": algorithm_config or {},
            "dataset": str(dataset),
            "role": "player_0",
            "roles": ["player_0", "player_1"],
            "seed": 17,
            "epochs": epochs,
            "batch_size": 4,
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


@pytest.mark.acceptance
def test_select_materialize_train_verify_and_reuse(tmp_path: Path) -> None:
    source_run = _source_run(tmp_path)
    selected_dir = tmp_path / "selected"
    selected = select_episodes(
        source_runs=[source_run], query={"role": "player_0"}, out_dir=selected_dir
    )
    assert len(selected["episodes"]) == 1

    portable_dir = tmp_path / "portable-dataset"
    portable = materialize_dataset(selected_dir / "dataset.yaml", out_dir=portable_dir)
    assert portable["lineage"]["materialized"] is True
    assert portable["episodes"][0]["path"].startswith("episodes/")

    # Removing the producer-side source does not affect the materialized dataset.
    (source_run / "trajectories" / "episode_0000.json").unlink()
    recipe = _recipe(tmp_path / "recipe.yaml", portable_dir / "dataset.yaml")
    result = run_training_recipe(recipe, out_dir=tmp_path / "train-run")
    assert result["schema"] == "arena.train-run/v1"
    assert result["examples"] == 4
    assert result["loss"]["final"] < result["loss"]["initial"]

    bundle = tmp_path / "train-run" / "policy.arena"
    policy = Policy.load(bundle)
    assert result["output_policy"]["digest"] == policy.digest
    assert verify_bundle_self(bundle)["verify_mode"] == "source-conformance"
    runtime = load_runtime(bundle)
    assert [runtime.act(obs) for obs in range(4)] == [1, 1, 1, 1]


def test_training_refuses_materialized_episode_mutation(tmp_path: Path) -> None:
    source_run = _source_run(tmp_path)
    selected_dir = tmp_path / "selected"
    select_episodes(source_runs=[source_run], query={}, out_dir=selected_dir)
    portable_dir = tmp_path / "portable"
    materialize_dataset(selected_dir / "dataset.yaml", out_dir=portable_dir)
    episode = portable_dir / "episodes" / "episode_000000.json"
    episode.write_text("{}", encoding="utf-8")
    recipe = _recipe(tmp_path / "recipe.yaml", portable_dir / "dataset.yaml")
    with pytest.raises(ConformanceError, match="mutation detected"):
        run_training_recipe(recipe, out_dir=tmp_path / "refused")
    assert not (tmp_path / "refused").exists()


def test_training_cli(tmp_path: Path) -> None:
    source_run = _source_run(tmp_path)
    selected_dir = tmp_path / "selected"
    select_episodes(source_runs=[source_run], query={}, out_dir=selected_dir)
    portable_dir = tmp_path / "portable"
    assert main(
        [
            "data",
            "materialize",
            str(selected_dir / "dataset.yaml"),
            "--out",
            str(portable_dir),
            "--json",
        ]
    ) == 0
    recipe = _recipe(tmp_path / "recipe.yaml", portable_dir / "dataset.yaml")
    assert main(["train", str(recipe), "--out", str(tmp_path / "cli-train"), "--json"]) == 0
    assert (tmp_path / "cli-train" / "train.json").exists()


def test_return_weighted_trainer_is_distinct_seeded_registry_case(
    tmp_path: Path,
) -> None:
    source_run = tmp_path / "source-run"
    trajectories = source_run / "trajectories"
    trajectories.mkdir(parents=True)
    for index, (action, episode_return) in enumerate(((0, -5.0), (2, 5.0))):
        episode = {
            "schema": "arena.trajectory/v0alpha1",
            "seed": index,
            "task": {"env": "arena/competitive_rps_v0"},
            "agents": ["player_0", "player_1"],
            "role_map": {"player_0": "player_0", "player_1": "player_1"},
            "policies": {},
            "returns": {"player_0": episode_return, "player_1": -episode_return},
            "steps": [
                {
                    "observations": {"player_0": 0},
                    "actions": {"player_0": action},
                    "rewards": {"player_0": episode_return},
                    "terminations": {"player_0": True},
                    "truncations": {"player_0": False},
                }
                for _ in range(4)
            ],
        }
        (trajectories / f"episode_{index:04d}.json").write_text(
            json.dumps(episode),
            encoding="utf-8",
        )
    selected_dir = tmp_path / "selected"
    select_episodes(source_runs=[source_run], query={}, out_dir=selected_dir)
    portable_dir = tmp_path / "portable"
    materialize_dataset(selected_dir / "dataset.yaml", out_dir=portable_dir)
    recipe = _recipe(
        tmp_path / "weighted.yaml",
        portable_dir / "dataset.yaml",
        algorithm="return_weighted_regression",
        algorithm_config={"temperature": 0.25, "max_weight": 20.0},
    )
    first = run_training_recipe(recipe, out_dir=tmp_path / "weighted-a")
    second = run_training_recipe(recipe, out_dir=tmp_path / "weighted-b")
    assert first["algorithm"] == "return_weighted_regression"
    assert first["sample_weights"]["max"] > first["sample_weights"]["min"]
    assert first["output_policy"]["digest"] == second["output_policy"]["digest"]
    assert load_runtime(tmp_path / "weighted-a" / "policy.arena").act(0) == 2


def test_materialized_dataset_splits_are_portable_and_seeded(
    tmp_path: Path,
) -> None:
    source_run = _source_run(tmp_path)
    original_path = source_run / "trajectories" / "episode_0000.json"
    original = json.loads(original_path.read_text())
    for index in range(1, 12):
        episode = {**original, "seed": index}
        (source_run / "trajectories" / f"episode_{index:04d}.json").write_text(
            json.dumps(episode),
            encoding="utf-8",
        )
    selected_dir = tmp_path / "selected"
    select_episodes(source_runs=[source_run], query={}, out_dir=selected_dir)
    split_spec = {"train": 0.7, "validation": 0.2, "test": 0.1}
    first = materialize_dataset(
        selected_dir / "dataset.yaml",
        out_dir=tmp_path / "portable-a",
        splits=split_spec,
        split_seed=31,
    )
    second = materialize_dataset(
        selected_dir / "dataset.yaml",
        out_dir=tmp_path / "portable-b",
        splits=split_spec,
        split_seed=31,
    )
    assert first["digest"] == second["digest"]
    assert [entry["split"] for entry in first["episodes"]] == [
        entry["split"] for entry in second["episodes"]
    ]
    assert sum(first["splits"]["counts"].values()) == 12

    chosen_split = first["episodes"][0]["split"]
    recipe = _recipe(
        tmp_path / "split-recipe.yaml",
        tmp_path / "portable-a" / "dataset.yaml",
    )
    recipe_data = load_manifest(recipe)
    recipe_data["dataset_split"] = chosen_split
    dump_yaml(recipe_data, recipe)
    result = run_training_recipe(recipe, out_dir=tmp_path / "split-training")
    assert result["dataset_split"] == chosen_split
    assert result["examples"] >= 4


def test_training_checkpoint_resume_matches_uninterrupted_policy(
    tmp_path: Path,
) -> None:
    source_run = _source_run(tmp_path)
    selected_dir = tmp_path / "selected"
    select_episodes(source_runs=[source_run], query={}, out_dir=selected_dir)
    portable_dir = tmp_path / "portable"
    materialize_dataset(selected_dir / "dataset.yaml", out_dir=portable_dir)
    dataset = portable_dir / "dataset.yaml"

    first_recipe = _recipe(tmp_path / "first.yaml", dataset, epochs=20)
    first = run_training_recipe(first_recipe, out_dir=tmp_path / "first")
    assert first["checkpoint"]["epochs_completed"] == 20

    full_recipe = _recipe(tmp_path / "full.yaml", dataset, epochs=40)
    full = run_training_recipe(full_recipe, out_dir=tmp_path / "full")

    resumed_recipe = _recipe(tmp_path / "resumed.yaml", dataset, epochs=40)
    resumed_data = load_manifest(resumed_recipe)
    resumed_data["resume_from"] = str(tmp_path / "first")
    dump_yaml(resumed_data, resumed_recipe)
    resumed = run_training_recipe(resumed_recipe, out_dir=tmp_path / "resumed")
    assert resumed["start_epoch"] == 20
    assert resumed["resumed_from"].endswith("first/checkpoint.json")
    assert resumed["output_policy"]["digest"] == full["output_policy"]["digest"]
    assert resumed["loss"]["epochs"] == full["loss"]["epochs"]

    changed_recipe = _recipe(tmp_path / "changed-resume.yaml", dataset, epochs=41)
    changed_data = load_manifest(changed_recipe)
    changed_data["resume_from"] = str(tmp_path / "first")
    changed_data["learning_rate"] = 0.025
    dump_yaml(changed_data, changed_recipe)
    with pytest.raises(ConformanceError, match="resume contract mismatch"):
        run_training_recipe(changed_recipe, out_dir=tmp_path / "changed-resume")
    assert not (tmp_path / "changed-resume").exists()

    checkpoint_payload = tmp_path / "first" / "checkpoint.pt"
    checkpoint_payload.write_bytes(checkpoint_payload.read_bytes() + b"tampered")
    refused_recipe = _recipe(tmp_path / "refused-resume.yaml", dataset, epochs=41)
    refused_data = load_manifest(refused_recipe)
    refused_data["resume_from"] = str(tmp_path / "first")
    dump_yaml(refused_data, refused_recipe)
    with pytest.raises(ConformanceError, match="checkpoint mutation"):
        run_training_recipe(refused_recipe, out_dir=tmp_path / "refused-resume")
    assert not (tmp_path / "refused-resume").exists()


def test_training_refuses_nonfinite_splits_and_fractional_discrete_actions(
    tmp_path: Path,
) -> None:
    source_run = _source_run(tmp_path)
    selected_dir = tmp_path / "selected"
    select_episodes(source_runs=[source_run], query={}, out_dir=selected_dir)
    with pytest.raises(SchemaError, match="finite and positive"):
        materialize_dataset(
            selected_dir / "dataset.yaml",
            out_dir=tmp_path / "invalid-split",
            splits={"train": float("nan")},
        )
    assert not (tmp_path / "invalid-split").exists()

    episode_path = source_run / "trajectories" / "episode_0000.json"
    episode = json.loads(episode_path.read_text())
    episode["steps"][0]["actions"]["player_0"] = 1.75
    episode_path.write_text(json.dumps(episode), encoding="utf-8")
    fractional_selected = tmp_path / "fractional-selected"
    select_episodes(
        source_runs=[source_run],
        query={},
        out_dir=fractional_selected,
    )
    portable = tmp_path / "fractional-portable"
    materialize_dataset(
        fractional_selected / "dataset.yaml",
        out_dir=portable,
    )
    recipe = _recipe(
        tmp_path / "fractional-recipe.yaml",
        portable / "dataset.yaml",
    )
    with pytest.raises(ConformanceError, match="action must be an integer"):
        run_training_recipe(recipe, out_dir=tmp_path / "fractional-training")
    assert not (tmp_path / "fractional-training").exists()
