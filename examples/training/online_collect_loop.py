#!/usr/bin/env python3
"""Bounded online collection → bind → offline train (CPU spike).

Single-process loop only. Each round:

1. ``Match.run`` collects Arena trajectories for a known policy digest
2. ``select_bound_episodes`` stamps ``arena.dataset-binding/v1``
3. ``materialize_dataset`` copies a portable slice (provenance is re-bound —
   materialize does not yet preserve the binding block; see RFC 011)
4. Existing ``run_training_recipe`` (behavior cloning) trains offline

This is **not** distributed RL, Ray, PPO, or a product claim that Arena “does
online RL.” See ``rfcs/011-online-collection-dataset-binding.md``.

Recipe (from a checkout with torch + pettingzoo)::

    python -m pip install -e '.[torch,pettingzoo]'
    python examples/training/online_collect_loop.py --out ./arena-online-wedge
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Keep the spike on CPU even when a GPU is present.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from arena.adapters.policy_custom_torch import verify_bundle_self
from arena.conformance.fixtures import build_fixed_action_rps_policy, build_rps_policy
from arena.core.dataset import materialize_dataset
from arena.core.identity import digest_uri, sha256_bytes
from arena.core.manifests import dump_json, dump_yaml, load_manifest
from arena.core.sdk import Match, Policy, Task
from arena.dataset import (
    bind_dataset_provenance,
    select_bound_episodes,
    verify_dataset_provenance,
)
from arena.runtime.training import run_training_recipe

REPO_ROOT = Path(__file__).resolve().parents[2]
PILOT = "arena/competitive_rps_v0"
LEARNER_ROLE = "player_0"
OPPONENT_ROLE = "player_1"
MAX_CYCLES = 4


def _require_stack() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "online_collect_loop requires PyTorch. Install with: "
            "pip install -e '.[torch]'"
        ) from exc
    try:
        import pettingzoo  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "online_collect_loop requires PettingZoo. Install with: "
            "pip install -e '.[pettingzoo]'"
        ) from exc


def _task_spec() -> dict[str, Any]:
    return {
        "adapter": "pettingzoo-parallel",
        "env": PILOT,
        "config": {"max_cycles": MAX_CYCLES},
    }


def _task_identity_from_run(run_dir: Path) -> dict[str, str]:
    """Read env/adapter/version from recorded episodes (not a guessed string)."""
    traj = sorted((run_dir / "trajectories").glob("episode_*.json"))
    if not traj:
        raise RuntimeError(f"no trajectories under {run_dir}")
    episode = json.loads(traj[0].read_text(encoding="utf-8"))
    task = episode.get("task") or {}
    identity = {
        key: str(task[key])
        for key in ("env", "adapter", "version")
        if task.get(key) is not None
    }
    if "env" not in identity:
        raise RuntimeError(f"episode missing task.env: {traj[0]}")
    return identity


def _collect_round(
    out_dir: Path,
    *,
    learner: Policy,
    opponent: Policy,
    seeds: list[int],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    record = Match(
        task=Task.load(_task_spec()),
        assignments={LEARNER_ROLE: learner, OPPONENT_ROLE: opponent},
        action_mode="deterministic",
        failure_policy={
            "timeout_seconds": 30,
            "retain_incomplete": True,
            "retry": 0,
        },
    ).run(seeds=list(seeds), record=True, out=out_dir)
    outcome = record.get("outcome") or {}
    if int(outcome.get("episodes_completed") or 0) != len(seeds):
        raise RuntimeError(
            f"collection incomplete under {out_dir}: {outcome!r} "
            f"(failures={record.get('failures')!r})"
        )
    if int(outcome.get("failure_count") or 0) != 0:
        raise RuntimeError(f"collection recorded failures under {out_dir}: {outcome!r}")
    return out_dir


def _write_recipe(path: Path, dataset: Path, *, seed: int, epochs: int) -> Path:
    dump_yaml(
        {
            "schema": "arena.train/v1",
            "name": "online-collect-wedge-bc",
            "algorithm": "behavior_cloning",
            "dataset": str(dataset),
            "role": LEARNER_ROLE,
            "roles": [LEARNER_ROLE, OPPONENT_ROLE],
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
                "hidden_dims": [16],
                "action_n": 3,
            },
            "preprocessing": {"id": "normalize_v0", "mean": 0.0, "std": 1.0},
        },
        path,
    )
    return path


def _bind_materialize_train(
    round_dir: Path,
    *,
    source_run: Path,
    policy_digest: str,
    task: dict[str, str],
    seed: int,
    epochs: int,
) -> dict[str, Any]:
    selected_dir = round_dir / "selected"
    bound = select_bound_episodes(
        source_runs=[source_run],
        policy_digest=policy_digest,
        task=task,
        role=LEARNER_ROLE,
        name=f"online-wedge-{round_dir.name}",
        out_dir=selected_dir,
    )
    verify_dataset_provenance(
        bound,
        expect_policy=policy_digest,
        expect_task=task,
        dataset_path=selected_dir / "dataset.yaml",
    )

    portable_dir = round_dir / "portable-dataset"
    materialize_dataset(selected_dir / "dataset.yaml", out_dir=portable_dir)
    portable_path = portable_dir / "dataset.yaml"
    portable = load_manifest(portable_path)
    # Honest gap: materialize drops provenance; re-bind before train (RFC 011).
    rebound = bind_dataset_provenance(
        portable,
        policy_digest=policy_digest,
        task=task,
        role=LEARNER_ROLE,
        verify_episodes=True,
        dataset_path=portable_path,
    )
    dump_yaml(rebound, portable_path)
    dump_json(rebound, portable_dir / "dataset.json")
    verify_dataset_provenance(
        rebound,
        expect_policy=policy_digest,
        expect_task=task,
        dataset_path=portable_path,
    )

    recipe = _write_recipe(
        round_dir / "recipe.yaml",
        portable_path,
        seed=seed,
        epochs=epochs,
    )
    train = run_training_recipe(recipe, out_dir=round_dir / "train-run")
    bundle = round_dir / "train-run" / "policy.arena"
    policy = Policy.load(bundle)
    verification = verify_bundle_self(bundle)
    return {
        "bound_dataset_digest": bound["digest"],
        "portable_dataset_digest": rebound["digest"],
        "provenance": rebound["provenance"],
        "train": train,
        "policy_digest": policy.digest,
        "policy_path": str(bundle.resolve()),
        "verification": verification,
    }


def run_online_collect_loop(
    out: Path,
    *,
    rounds: int = 2,
    episodes_per_round: int = 4,
    epochs: int = 8,
    seed: int = 17,
) -> dict[str, Any]:
    """Execute bounded collect → bind → materialize → re-bind → offline train."""
    _require_stack()
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    if episodes_per_round < 1:
        raise ValueError("episodes_per_round must be >= 1")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"output directory must be empty or absent: {out}")
    out.mkdir(parents=True, exist_ok=True)

    learner = Policy.load(
        build_rps_policy(out / "seed-learner.arena", role=LEARNER_ROLE, seed=seed)
    )
    opponent = Policy.load(
        build_fixed_action_rps_policy(
            out / "opponent-rock.arena",
            role=OPPONENT_ROLE,
            action=0,
            name="rock-opponent",
        )
    )

    round_summaries: list[dict[str, Any]] = []
    current = learner
    for index in range(rounds):
        round_dir = out / f"round-{index:02d}"
        seeds = [seed + index * 100 + i for i in range(episodes_per_round)]
        collect_dir = _collect_round(
            round_dir / "collect-run",
            learner=current,
            opponent=opponent,
            seeds=seeds,
        )
        task = _task_identity_from_run(collect_dir)
        trained = _bind_materialize_train(
            round_dir,
            source_run=collect_dir,
            policy_digest=current.digest,
            task=task,
            seed=seed + index,
            epochs=epochs,
        )
        current = Policy.load(trained["policy_path"])
        round_summaries.append(
            {
                "round": index,
                "collect_seeds": seeds,
                "collector_policy_digest": trained["provenance"]["policy_digest"],
                "task": task,
                "bound_dataset_digest": trained["bound_dataset_digest"],
                "portable_dataset_digest": trained["portable_dataset_digest"],
                "trained_policy_digest": trained["policy_digest"],
                "examples": trained["train"]["examples"],
                "loss_initial": trained["train"]["loss"]["initial"],
                "loss_final": trained["train"]["loss"]["final"],
                "verification": trained["verification"]["verify_mode"],
            }
        )

    lineage = {
        "schema": "arena.online-collect-wedge/v1",
        "ok": True,
        "rfc": "rfcs/011-online-collection-dataset-binding.md",
        "not_in_scope": [
            "distributed collection",
            "Ray",
            "online RL algorithms (PPO/SAC/…)",
            "replay services",
        ],
        "rounds": round_summaries,
        "final_policy_digest": current.digest,
        "final_policy_path": str((out / f"round-{rounds - 1:02d}" / "train-run" / "policy.arena").resolve()),
    }
    report_identity = {
        key: value
        for key, value in lineage.items()
        if key not in {"ok"}
    }
    # Drop nested verification strings already recorded per round for a stable digest.
    lineage["report_digest"] = digest_uri(
        sha256_bytes(json.dumps(report_identity, sort_keys=True, default=str).encode())
    )
    lineage["out"] = str(out.resolve())
    dump_json(lineage, out / "result.json")
    dump_yaml(lineage, out / "result.yaml")
    return lineage


def main(argv: list[str] | None = None) -> int:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="./arena-online-wedge")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    result = run_online_collect_loop(
        Path(args.out).resolve(),
        rounds=args.rounds,
        episodes_per_round=args.episodes,
        epochs=args.epochs,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
