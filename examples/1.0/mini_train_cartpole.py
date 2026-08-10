"""Mini lab loop: CartPole collect → Arena BC train → verify → match/eval.

CPU-only, seconds not minutes. Reuses ``arena.runtime.training`` and the
``entrypoint_bundle`` match path so a single lab can prove train → export →
verify → seeded eval without a hosted control plane.

Recipe (from a checkout with torch + gymnasium/pettingzoo)::

    python -m pip install -e '.[torch,pettingzoo]'
    python examples/1.0/mini_train_cartpole.py --out ./arena-mini-train
    arena policy verify ./arena-mini-train/train-run/policy.arena
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

from arena.adapters.policy_custom_torch import load_runtime, verify_bundle_self
from arena.cli.main import main as arena_main
from arena.core.dataset import materialize_dataset, select_episodes
from arena.core.identity import digest_uri, sha256_bytes, sha256_file
from arena.core.manifests import RUN_SCHEMA, TRAJECTORY_SCHEMA, dump_json, dump_yaml
from arena.core.sdk import Match, Policy, Task
from arena.runtime.training import run_training_recipe

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[1]

AGENT = "agent"
CARTPOLE_ACTION = {
    "type": "Discrete",
    "n": 2,
    "dtype": "int64",
    "masks": "none",
}


def _load_cartpole_module() -> Any:
    path = EXAMPLE_DIR / "cartpole_parallel.py"
    spec = importlib.util.spec_from_file_location("arena_cartpole_parallel", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cartpole_observation_contract() -> dict[str, Any]:
    """Match-compatible Box contract derived from the Parallel wrapper space."""
    return dict(_load_cartpole_module().CARTPOLE_OBS)


def _require_stack() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "mini_train_cartpole requires PyTorch. Install with: "
            "pip install 'arena[torch]' (or pip install -e '.[torch]')"
        ) from exc
    try:
        import gymnasium  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "mini_train_cartpole requires Gymnasium. Install with: "
            "pip install 'arena[pettingzoo]' (or pip install gymnasium)"
        ) from exc


def _heuristic_action(obs: np.ndarray, rng: np.random.Generator, *, epsilon: float) -> int:
    if float(rng.random()) < epsilon:
        return int(rng.integers(0, 2))
    return 1 if float(obs[2] + 0.5 * obs[3]) > 0.0 else 0


def collect_teacher_run(
    out_dir: Path,
    *,
    episodes: int,
    seed: int,
    epsilon: float,
) -> Path:
    """Roll out heuristic(+ε) CartPole and write an Arena match-shaped run."""
    import gymnasium as gym

    out_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = out_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    episode_summaries: list[dict[str, Any]] = []
    for index in range(episodes):
        env = gym.make("CartPole-v1")
        obs, _info = env.reset(seed=seed + index)
        rng = np.random.default_rng(seed + 10_000 + index)
        steps: list[dict[str, Any]] = []
        total = 0.0
        done = False
        step_i = 0
        while not done:
            action = _heuristic_action(
                np.asarray(obs, dtype=np.float32), rng, epsilon=epsilon
            )
            next_obs, reward, terminated, truncated, _info = env.step(action)
            reward_f = float(reward)
            total += reward_f
            steps.append(
                {
                    "t": step_i,
                    "observations": {AGENT: np.asarray(obs, dtype=np.float32).tolist()},
                    "actions": {AGENT: int(action)},
                    "rewards": {AGENT: reward_f},
                    "terminations": {AGENT: bool(terminated)},
                    "truncations": {AGENT: bool(truncated)},
                    "action_masks": {},
                    "infos": {AGENT: {}},
                }
            )
            done = bool(terminated or truncated)
            obs = next_obs
            step_i += 1
        env.close()
        episode = {
            "schema": TRAJECTORY_SCHEMA,
            "seed": seed + index,
            "episode_index": index,
            "status": "completed",
            "action_mode": "stochastic",
            "task": {"env": "gymnasium/CartPole-v1", "adapter": "pettingzoo-parallel"},
            "agents": [AGENT],
            "role_map": {AGENT: AGENT},
            "policies": {"teacher": "heuristic+epsilon"},
            "returns": {AGENT: total},
            "steps": steps,
        }
        (traj_dir / f"episode_{index:04d}.json").write_text(
            json.dumps(episode), encoding="utf-8"
        )
        episode_summaries.append(
            {
                "episode_index": index,
                "seed": seed + index,
                "status": "completed",
                "steps": len(steps),
                "returns": {AGENT: total},
            }
        )
    run_record = {
        "schema": RUN_SCHEMA,
        "run_id": f"cartpole-teacher-{seed}",
        "task": {
            "adapter": "pettingzoo-parallel",
            "env": "gymnasium/CartPole-v1",
            "version": "example",
            "spec": {"env": "gymnasium/CartPole-v1"},
        },
        "assignments": {AGENT: {"name": "heuristic-teacher", "digest": "local:heuristic"}},
        "seeds": [seed + i for i in range(episodes)],
        "action_mode": "stochastic",
        "episodes": episode_summaries,
        "failures": [],
        "outcome": {
            "episodes_requested": episodes,
            "episodes_completed": episodes,
            "failure_count": 0,
        },
    }
    dump_yaml(run_record, out_dir / "run.yaml")
    dump_json(run_record, out_dir / "run.json")
    return out_dir


def _write_recipe(path: Path, dataset: Path, *, seed: int, epochs: int) -> Path:
    dump_yaml(
        {
            "schema": "arena.train/v1",
            "name": "cartpole-heuristic-bc",
            "algorithm": "behavior_cloning",
            "dataset": str(dataset),
            "role": AGENT,
            "roles": [AGENT],
            "seed": seed,
            "epochs": epochs,
            "batch_size": 64,
            "learning_rate": 0.01,
            "observation": _cartpole_observation_contract(),
            "action": CARTPOLE_ACTION,
            "architecture": {
                "type": "mlp_categorical",
                "observation_dim": 4,
                "hidden_dims": [32],
                "action_n": 2,
            },
            "preprocessing": {"id": "normalize_v0", "mean": 0.0, "std": 1.0},
        },
        path,
    )
    return path


def _cartpole_task_spec() -> dict[str, Any]:
    entry = EXAMPLE_DIR / "cartpole_parallel.py"
    return {
        "adapter": "pettingzoo-parallel",
        "env": "gymnasium/CartPole-v1",
        "interaction": "parallel",
        "trust_task_code": True,
        "source_revision": "examples/1.0/cartpole_parallel.py",
        "packaging": {
            "kind": "entrypoint_bundle",
            "root": str(EXAMPLE_DIR),
            "entrypoint": "cartpole_parallel.py",
            "digest": digest_uri(sha256_file(entry)),
            "factory": "parallel_env",
            "trust_task_code": True,
        },
    }


def gymnasium_eval(
    bundle: Path,
    *,
    seeds: list[int],
    action_mode: str = "deterministic",
) -> dict[str, Any]:
    """Short seeded CartPole eval via Gymnasium + Arena runtime."""
    import gymnasium as gym

    runtime = load_runtime(bundle)
    returns: list[float] = []
    for seed in seeds:
        env = gym.make("CartPole-v1")
        obs, _ = env.reset(seed=int(seed))
        runtime.reset(AGENT)
        total = 0.0
        done = False
        step_i = 0
        while not done:
            rng = np.random.default_rng(int(seed) + step_i)
            action = runtime.act(
                np.asarray(obs, dtype=np.float32).tolist(),
                mode=action_mode,
                rng=rng,
                agent_id=AGENT,
            )
            obs, reward, terminated, truncated, _ = env.step(int(action))
            total += float(reward)
            done = bool(terminated or truncated)
            step_i += 1
        env.close()
        returns.append(total)
    return {
        "seeds": list(seeds),
        "action_mode": action_mode,
        "returns": returns,
        "mean_return": float(statistics.fmean(returns)),
        "min_return": float(min(returns)),
        "max_return": float(max(returns)),
    }


def _match_mean_return(match_record: dict[str, Any]) -> float | None:
    values = [
        float((ep.get("returns") or {}).get(AGENT))
        for ep in match_record.get("episodes") or []
        if ep.get("status") == "completed" and (ep.get("returns") or {}).get(AGENT) is not None
    ]
    if not values:
        return None
    return float(statistics.fmean(values))


def run_mini_train(
    out: Path,
    *,
    episodes: int = 24,
    epochs: int = 40,
    seed: int = 7,
    epsilon: float = 0.15,
    eval_seeds: list[int] | None = None,
    match_seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Execute collect → materialize → train → verify → eval/match."""
    _require_stack()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"output directory must be empty or absent: {out}")
    out.mkdir(parents=True, exist_ok=True)
    eval_seeds = list(eval_seeds or [101, 102, 103, 104])
    match_seeds = list(match_seeds or [201, 202])

    teacher_run = collect_teacher_run(
        out / "teacher-run",
        episodes=episodes,
        seed=seed,
        epsilon=epsilon,
    )
    selected_dir = out / "selected"
    select_episodes(source_runs=[teacher_run], query={}, out_dir=selected_dir)
    portable_dir = out / "portable-dataset"
    portable = materialize_dataset(selected_dir / "dataset.yaml", out_dir=portable_dir)

    recipe = _write_recipe(
        out / "recipe.yaml",
        portable_dir / "dataset.yaml",
        seed=seed,
        epochs=epochs,
    )
    train = run_training_recipe(recipe, out_dir=out / "train-run")
    bundle = out / "train-run" / "policy.arena"
    policy = Policy.load(bundle)
    verification = verify_bundle_self(bundle)
    if arena_main(["policy", "verify", str(bundle)]) != 0:
        raise SystemExit("arena policy verify failed")

    gym_eval = gymnasium_eval(bundle, seeds=eval_seeds)
    random_returns: list[float] = []
    import gymnasium as gym

    for s in eval_seeds:
        env = gym.make("CartPole-v1")
        obs, _ = env.reset(seed=int(s))
        rng = np.random.default_rng(int(s) + 999)
        total = 0.0
        done = False
        while not done:
            obs, reward, terminated, truncated, _ = env.step(int(rng.integers(0, 2)))
            total += float(reward)
            done = bool(terminated or truncated)
        env.close()
        random_returns.append(total)
    random_mean = float(statistics.fmean(random_returns))

    match_record: dict[str, Any] | None = None
    try:
        import pettingzoo  # noqa: F401
    except ImportError:
        pettingzoo = None  # type: ignore[assignment]
    if pettingzoo is not None:
        task_spec = _cartpole_task_spec()
        dump_yaml(task_spec, out / "cartpole-task.yaml")
        match_record = Match(
            task=Task.load(task_spec),
            assignments={AGENT: policy},
            action_mode="deterministic",
            failure_policy={
                "timeout_seconds": 30,
                "retain_incomplete": True,
                "retry": 0,
            },
        ).run(seeds=match_seeds, out=out / "match-run")

    lineage = {
        "schema": "arena.mini-train-cartpole/v1",
        "ok": True,
        "dataset_digest": train.get("dataset_digest") or portable.get("digest"),
        "recipe_digest": train.get("recipe_digest"),
        "training_contract_digest": train.get("training_contract_digest"),
        "policy_digest": policy.digest,
        "verification": verification,
        "train": {
            "algorithm": train["algorithm"],
            "examples": train["examples"],
            "loss_initial": train["loss"]["initial"],
            "loss_final": train["loss"]["final"],
            "seed": train.get("seed", seed),
        },
        "gymnasium_eval": gym_eval,
        "random_baseline_mean_return": random_mean,
        "match": None
        if match_record is None
        else {
            "run_id": match_record.get("run_id"),
            "seeds": match_record.get("seeds"),
            "outcome": match_record.get("outcome"),
            "assignments": match_record.get("assignments"),
            "mean_return": _match_mean_return(match_record),
        },
    }
    report_identity = {k: v for k, v in lineage.items() if k != "verification"}
    lineage["report_digest"] = digest_uri(
        sha256_bytes(json.dumps(report_identity, sort_keys=True, default=str).encode())
    )
    lineage["out"] = str(out.resolve())
    lineage["policy_path"] = str(bundle.resolve())
    dump_json(lineage, out / "result.json")
    dump_yaml(lineage, out / "result.yaml")
    return lineage


def main(argv: list[str] | None = None) -> int:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="./arena-mini-train")
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epsilon", type=float, default=0.15)
    args = parser.parse_args(argv)
    result = run_mini_train(
        Path(args.out).resolve(),
        episodes=args.episodes,
        epochs=args.epochs,
        seed=args.seed,
        epsilon=args.epsilon,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
