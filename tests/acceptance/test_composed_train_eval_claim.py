"""Gap #2: one composed acceptance journey.

select → materialize → train (BC) → interrupt/resume (same contract digest)
→ eval → file:// push/pull with stable policy + eval-intent digests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("pettingzoo")

from arena.conformance.fixtures import build_fixed_action_rps_policy
from arena.core.dataset import materialize_dataset, select_episodes
from arena.core.manifests import dump_yaml, evaluation_intent_digest, load_manifest
from arena.core.mirror import pull_artifact, push_artifact
from arena.core.sdk import Policy
from arena.runtime.evaluation import run_evaluation
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
    epochs: int,
    resume_from: Path | None = None,
) -> Path:
    payload: dict = {
        "schema": "arena.train/v1",
        "name": "composed-paper",
        "algorithm": "behavior_cloning",
        "algorithm_config": {},
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
    }
    if resume_from is not None:
        payload["resume_from"] = str(resume_from)
    dump_yaml(payload, path)
    return path


@pytest.mark.acceptance
@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_composed_select_train_resume_eval_file_claim(tmp_path: Path) -> None:
    source_run = _source_run(tmp_path)
    selected_dir = tmp_path / "selected"
    selected = select_episodes(
        source_runs=[source_run],
        query={"role": "player_0"},
        out_dir=selected_dir,
    )
    assert len(selected["episodes"]) == 1

    portable_dir = tmp_path / "portable-dataset"
    portable = materialize_dataset(selected_dir / "dataset.yaml", out_dir=portable_dir)
    assert portable["lineage"]["materialized"] is True
    dataset = portable_dir / "dataset.yaml"

    stage1 = run_training_recipe(
        _recipe(tmp_path / "stage1.yaml", dataset, epochs=20),
        out_dir=tmp_path / "train-stage1",
    )
    assert stage1["schema"] == "arena.train-run/v1"
    assert stage1["checkpoint"]["epochs_completed"] == 20
    stage1_contract = stage1["training_contract_digest"]
    assert stage1_contract.startswith("sha256:")

    uninterrupted = run_training_recipe(
        _recipe(tmp_path / "full.yaml", dataset, epochs=40),
        out_dir=tmp_path / "train-full",
    )

    resumed = run_training_recipe(
        _recipe(
            tmp_path / "resumed.yaml",
            dataset,
            epochs=40,
            resume_from=tmp_path / "train-stage1",
        ),
        out_dir=tmp_path / "train-resumed",
    )
    assert resumed["start_epoch"] == 20
    assert resumed["training_contract_digest"] == stage1_contract
    assert resumed["training_contract_digest"] == uninterrupted["training_contract_digest"]
    assert resumed["output_policy"]["digest"] == uninterrupted["output_policy"]["digest"]
    checkpoint = load_manifest(tmp_path / "train-stage1" / "checkpoint.json")
    assert checkpoint["training_contract_digest"] == stage1_contract

    trained_bundle = tmp_path / "train-resumed" / "policy.arena"
    trained = Policy.load(trained_bundle)
    assert trained.digest == resumed["output_policy"]["digest"]

    rock_path = build_fixed_action_rps_policy(
        tmp_path / "rock.arena",
        role=["player_0", "player_1"],
        action=0,
        name="rock",
    )
    rock = Policy.load(rock_path)

    suite = {
        "schema": "arena.evaluation/v0alpha1",
        "name": "composed-trained-vs-rock",
        "interaction": "parallel",
        "task": {
            "adapter": "pettingzoo-parallel",
            "env": "arena/competitive_rps_v0",
            "config": {"max_cycles": 1},
        },
        "assignments": {
            "player_0": {"kind": "policy", "policy": trained.digest},
            "player_1": {"kind": "policy", "policy": rock.digest},
        },
        "seeds": {"start": 0, "count": 2},
        "action_mode": "deterministic",
        "metrics": ["mean_return", "win_rate"],
    }
    intent = evaluation_intent_digest(suite)
    policy_index = {
        trained.digest: trained_bundle,
        rock.digest: Path(rock_path),
    }
    eval_run = run_evaluation(
        suite,
        policy_index=policy_index,
        out_dir=tmp_path / "eval-run",
        workers=1,
    )
    assert eval_run["evaluation_intent_digest"] == intent
    assert evaluation_intent_digest(suite) == intent
    assert (tmp_path / "eval-run" / "eval_run.json").exists()
    assert len(eval_run["cells"]) >= 1

    mirror = tmp_path / "file-store"
    pushed = push_artifact(trained_bundle, mirror.as_uri(), verify=True)
    assert pushed["identity"] == trained.digest
    assert pushed["uri"].endswith(f"#{trained.digest}")

    restored = tmp_path / "restored.arena"
    pulled = pull_artifact(pushed["uri"], restored, verify=True)
    assert pulled["identity"] == trained.digest
    assert Policy.load(restored).digest == trained.digest
    assert Policy.load(restored).digest == resumed["output_policy"]["digest"]
