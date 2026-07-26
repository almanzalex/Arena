"""Execute Arena 0.5 generalization boundaries as one local user journey."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from urllib.parse import quote

from arena.adapters.policy_custom_torch import (
    build_module,
    export_policy,
    verify_bundle_self,
)
from arena.conformance.fixtures import build_fixed_action_rps_policy
from arena.core.attestation import (
    generate_signing_keypair,
    sign_artifact,
    verify_artifact_attestation,
)
from arena.core.dataset import materialize_dataset, select_episodes
from arena.core.manifests import dump_yaml, load_manifest
from arena.core.mirror import pull_artifact, push_artifact
from arena.core.sdk import Match, Policy, Task
from arena.core.tasks import verify_task_equivalence
from arena.runtime.training import run_training_recipe

REPO_ROOT = Path(__file__).resolve().parents[2]


def _openspiel_policy(
    out: Path,
    *,
    game: str,
    observation_dim: int,
    action_n: int,
) -> Policy:
    import torch

    architecture = {
        "type": "mlp_categorical",
        "observation_dim": observation_dim,
        "hidden_dims": [16],
        "action_n": action_n,
    }
    module = build_module(architecture)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()
    bundle = export_policy(
        out_dir=out,
        name=f"{game}-first-legal",
        roles=["player_0", "player_1"],
        observation={
            "type": "Box",
            "shape": [observation_dim],
            "dtype": "float32",
            "low": 0.0,
            "high": 1.0,
        },
        action={
            "type": "Discrete",
            "n": action_n,
            "dtype": "int64",
            "masks": "required",
        },
        architecture=architecture,
        state_dict=module.state_dict(),
        preprocessing={"id": "normalize_v0", "mean": 0.0, "std": 1.0},
        reference_cases={
            "provenance": "source-conformance",
            "cases": [
                {
                    "observation": [0.0] * observation_dim,
                    "action_mask": [1] * action_n,
                    "mode": "deterministic",
                    "expected_action": 0,
                }
            ],
        },
        lineage={"game": game, "fixture": "0.5-generalization-demo"},
    )
    return Policy.load(bundle)


def _training_recipe(
    path: Path,
    dataset: Path,
    *,
    epochs: int,
    resume_from: Path | None = None,
) -> None:
    dump_yaml(
        {
            "schema": "arena.train/v1",
            "name": "imitate-paper-from-trajectories",
            "algorithm": "behavior_cloning",
            "dataset": str(dataset),
            "dataset_split": "train",
            "role": "player_0",
            "roles": ["player_0", "player_1"],
            "seed": 23,
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
                "hidden_dims": [8],
                "action_n": 3,
            },
            "preprocessing": {
                "id": "normalize_v0",
                "mean": 0.0,
                "std": 1.0,
            },
            **({"resume_from": str(resume_from)} if resume_from is not None else {}),
        },
        path,
    )


def run_demo(out: Path) -> dict:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # Flow 1: role-resolved removal/rejoin + joint birth with segment history.
    dynamic_policy = Policy.load(
        build_fixed_action_rps_policy(
            out / "dynamic.arena",
            role=["contestant"],
            action=0,
            name="dynamic-role-policy",
        )
    )
    dynamic_task = {
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
                "join_eligibility": {"contestant": [dynamic_policy.digest]},
            }
        },
    }
    dynamic_run = Match(
        task=Task.load(dynamic_task),
        assignments={"contestant": dynamic_policy},
    ).run(seeds=[3], out=out / "dynamic-run")
    dynamic_episode = json.loads(
        (out / "dynamic-run" / "trajectories" / "episode_0000.json").read_text()
    )

    # Flow 2: trajectories -> deterministic splits -> train -> resume -> reuse.
    paper = Policy.load(
        build_fixed_action_rps_policy(
            out / "paper.arena",
            role=["player_0", "player_1"],
            action=1,
            name="paper-teacher",
        )
    )
    rock = Policy.load(
        build_fixed_action_rps_policy(
            out / "rock.arena",
            role=["player_0", "player_1"],
            action=0,
            name="rock-opponent",
        )
    )
    rps_task = Task.load(
        {
            "adapter": "pettingzoo-parallel",
            "env": "arena/competitive_rps_v0",
            "interaction": "parallel",
            "config": {"max_cycles": 3},
        }
    )
    teacher_run = Match(
        task=rps_task,
        assignments={"player_0": paper, "player_1": rock},
    ).run(seeds=list(range(12)), out=out / "teacher-run")
    selected = select_episodes(
        source_runs=[out / "teacher-run"],
        query={"role": "player_0"},
        name="paper-demonstrations",
        out_dir=out / "selected",
    )
    portable = materialize_dataset(
        out / "selected" / "dataset.yaml",
        out_dir=out / "portable-dataset",
        splits={"train": 0.8, "validation": 0.2},
        split_seed=17,
    )
    stage1_recipe = out / "train-stage1.yaml"
    _training_recipe(
        stage1_recipe,
        out / "portable-dataset" / "dataset.yaml",
        epochs=25,
    )
    stage1 = run_training_recipe(stage1_recipe, out_dir=out / "training-stage1")
    resumed_recipe = out / "train-resumed.yaml"
    _training_recipe(
        resumed_recipe,
        out / "portable-dataset" / "dataset.yaml",
        epochs=50,
        resume_from=out / "training-stage1",
    )
    training = run_training_recipe(resumed_recipe, out_dir=out / "training")
    trained = Policy.load(out / "training" / "policy.arena")
    verification = verify_bundle_self(trained.root)
    reuse_run = Match(
        task=rps_task,
        assignments={"player_0": trained, "player_1": rock},
    ).run(seeds=[19], out=out / "trained-policy-run")

    # Flow 3: one game from each qualified OpenSpiel semantic family.
    openspiel_results: dict[str, dict] = {}
    for game, file_stem, observation_dim, action_n in (
        ("connect_four", "connect-four", 126, 7),
        ("kuhn_poker", "kuhn-poker", 11, 2),
        ("matrix_rps", "matrix-rps", 1, 3),
    ):
        game_policy = _openspiel_policy(
            out / f"{file_stem}.arena",
            game=game,
            observation_dim=observation_dim,
            action_n=action_n,
        )
        game_task = Task.load(
            REPO_ROOT / f"examples/tasks/openspiel-{file_stem}.yaml"
        )
        game_run = Match(
            task=game_task,
            assignments={"player_0": game_policy, "player_1": game_policy},
        ).run(seeds=[0], out=out / f"{file_stem}-run")
        equivalence = verify_task_equivalence(
            game_task.spec,
            None,
            load_manifest(
                REPO_ROOT / f"examples/tasks/openspiel-{file_stem}-trace.yaml"
            ),
        )
        openspiel_results[game] = {
            "outcome": game_run["outcome"],
            "equivalence_ok": equivalence["ok"],
            "trace_digest": equivalence["captured_trace_digest"],
        }

    # Flow 4: user-owned authenticity survives every mirrored store identity.
    key_dir = out / "keys"
    key_dir.mkdir()
    private_key = key_dir / "lab-private.pem"
    public_key = key_dir / "lab-public.pem"
    key = generate_signing_keypair(
        private_key=private_key,
        public_key=public_key,
    )
    attestation_path = out / "trained-policy.attestation.json"
    sign_artifact(
        trained.root,
        private_key=private_key,
        out=attestation_path,
        issuer="arena-0.5-demo-lab",
    )
    authenticity = verify_artifact_attestation(
        trained.root,
        attestation=attestation_path,
        public_key=public_key,
    )
    store_results = {}
    destinations = {
        "hf": "hf://models/lab/arena",
        "oci": "oci://registry.example/lab/arena",
        "wandb": "wandb://lab/project/arena",
        "mlflow": "mlflow://arena-experiment",
    }
    for name, base in destinations.items():
        mirror_root = out / "mirrors" / name
        destination = f"{base}?simulate={quote(str(mirror_root.resolve()), safe='/')}"
        pushed = push_artifact(trained.root, destination, verify=True)
        restored_path = out / "restored" / f"{name}.arena"
        pulled = pull_artifact(pushed["uri"], restored_path, verify=True)
        restored = Policy.load(restored_path)
        restored_signature = verify_artifact_attestation(
            restored_path,
            attestation=attestation_path,
            public_key=public_key,
        )
        store_results[name] = {
            "verified": pushed["verified"] and pulled["verified"],
            "identity_equal": restored.digest == trained.digest,
            "signature_valid": restored_signature["ok"],
            "uri": pushed["uri"],
        }

    summary = {
        "dynamic": {
            "outcome": dynamic_run["outcome"],
            "lifecycle_events": dynamic_run["episodes"][0]["lifecycle_events"],
            "resolver": dynamic_run["lifecycle_resolver"],
            "agent_0_segments": len(
                dynamic_episode["agent_segment_history"]["agent_0"]
            ),
        },
        "training": {
            "teacher_outcome": teacher_run["outcome"],
            "selected_episodes": len(selected["episodes"]),
            "portable_dataset_digest": portable["digest"],
            "split_counts": portable["splits"]["counts"],
            "examples": training["examples"],
            "loss_initial": training["loss"]["initial"],
            "loss_final": training["loss"]["final"],
            "verify_mode": verification["verify_mode"],
            "reuse_outcome": reuse_run["outcome"],
            "policy_digest": trained.digest,
            "resumed_from_epoch": training["start_epoch"],
            "checkpoint_digest": training["checkpoint"]["payload_digest"],
            "stage1_policy_digest": stage1["output_policy"]["digest"],
        },
        "openspiel": openspiel_results,
        "authenticity": {
            "verified": authenticity["ok"],
            "issuer": authenticity["issuer"],
            "key_id": key["key_id"],
        },
        "stores": store_results,
        "out": str(out),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / ".demo",
    )
    args = parser.parse_args()
    print(json.dumps(run_demo(args.out.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
