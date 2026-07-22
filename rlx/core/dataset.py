"""Lineage-preserving trajectory dataset slices (RLX 0.2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rlx.core.identity import canonical_json, digest_uri, sha256_bytes
from rlx.core.manifests import DATASET_SCHEMA, dump_json, dump_yaml, validate_dataset_manifest


def _episode_matches(ep: dict[str, Any], *, query: dict[str, Any]) -> bool:
    if "seed" in query and ep.get("seed") != query["seed"]:
        return False
    if "policy" in query:
        policies = set((ep.get("assignments") or {}).values()) if isinstance(ep.get("assignments"), dict) else set()
        # Also check lineage on steps if present
        if query["policy"] not in policies and query["policy"] not in json.dumps(ep):
            return False
    if "role" in query and "opponent" in query:
        pass  # structural filters applied by caller with run metadata
    if "outcome" in query:
        outcomes = ep.get("outcomes") or {}
        role = query.get("role")
        if role and outcomes.get(role) != query["outcome"]:
            # Infer from returns when outcomes missing.
            returns = ep.get("returns") or {}
            if role in returns:
                r = float(returns[role])
                want = query["outcome"]
                if want == "win" and r <= 0:
                    return False
                if want == "loss" and r >= 0:
                    return False
                if want == "draw" and r != 0:
                    return False
            elif outcomes:
                return False
    return True


def select_episodes(
    *,
    source_runs: list[str | Path],
    query: dict[str, Any],
    name: str = "slice",
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Filter episodes from eval/match run directories without rewriting sources."""
    selected: list[dict[str, Any]] = []
    for run in source_runs:
        run_path = Path(run)
        traj_dirs = []
        if (run_path / "trajectories").is_dir():
            traj_dirs.append(run_path / "trajectories")
        else:
            traj_dirs.extend(sorted(run_path.glob("**/trajectories")))
        for traj_dir in traj_dirs:
            for ep_path in sorted(traj_dir.glob("episode_*.json")):
                ep = json.loads(ep_path.read_text(encoding="utf-8"))
                # Attach assignments from sibling run.yaml when present.
                run_yaml = traj_dir.parent / "run.yaml"
                if run_yaml.exists():
                    import yaml

                    run_rec = yaml.safe_load(run_yaml.read_text(encoding="utf-8")) or {}
                    assigns = {
                        k: (v.get("digest") if isinstance(v, dict) else v)
                        for k, v in (run_rec.get("assignments") or {}).items()
                    }
                    ep = {**ep, "assignments": assigns, "task": run_rec.get("task")}
                if "task" in query:
                    task_env = (ep.get("task") or {}).get("env")
                    if task_env and task_env != query["task"]:
                        continue
                if "opponent" in query:
                    assigns = ep.get("assignments") or {}
                    if query["opponent"] not in assigns.values():
                        continue
                if "policy" in query:
                    assigns = ep.get("assignments") or {}
                    if query["policy"] not in assigns.values():
                        continue
                if not _episode_matches(ep, query=query):
                    continue
                digest = digest_uri(sha256_bytes(ep_path.read_bytes()))
                selected.append(
                    {
                        "path": str(ep_path.resolve()),
                        "digest": digest,
                        "seed": ep.get("seed"),
                        "source_run": str(traj_dir.parent.resolve()),
                    }
                )

    dataset = {
        "schema": DATASET_SCHEMA,
        "name": name,
        "source_runs": [str(Path(r).resolve()) for r in source_runs],
        "episodes": selected,
        "query": query,
        "lineage": {
            "note": "episodes reference immutable source paths/digests; sources are not rewritten",
        },
    }
    validate_dataset_manifest(dataset)
    dataset["digest"] = digest_uri(sha256_bytes(canonical_json(dataset)))
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        dump_yaml(dataset, out_dir / "dataset.yaml")
        dump_json(dataset, out_dir / "dataset.json")
    return dataset
