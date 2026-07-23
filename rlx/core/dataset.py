"""Lineage-preserving trajectory dataset slices (RLX 0.2)."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

from rlx.core.errors import ConformanceError, SchemaError
from rlx.core.identity import canonical_json, digest_uri, sha256_bytes, sha256_file
from rlx.core.manifests import (
    DATASET_SCHEMA,
    dump_json,
    dump_yaml,
    load_manifest,
    validate_dataset_manifest,
)


def dataset_content_digest(dataset: dict[str, Any]) -> str:
    identity = {key: value for key, value in dataset.items() if key != "digest"}
    return digest_uri(sha256_bytes(canonical_json(identity)))


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
    dataset["digest"] = dataset_content_digest(dataset)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        dump_yaml(dataset, out_dir / "dataset.yaml")
        dump_json(dataset, out_dir / "dataset.json")
    return dataset


def _normalize_split_weights(splits: dict[str, float]) -> list[tuple[str, float]]:
    if not splits:
        raise SchemaError("dataset splits must not be empty")
    normalized: list[tuple[str, float]] = []
    total = 0.0
    for name, raw_weight in splits.items():
        if not str(name).strip():
            raise SchemaError("dataset split names must be non-empty")
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight <= 0:
            raise SchemaError(
                f"dataset split {name!r} weight must be finite and positive"
            )
        total += weight
        normalized.append((str(name), weight))
    return [(name, weight / total) for name, weight in normalized]


def _assign_split(
    *,
    entry_digest: str,
    index: int,
    split_seed: int,
    weights: list[tuple[str, float]],
) -> str:
    bucket_digest = sha256_bytes(
        canonical_json(
            {
                "episode_digest": entry_digest,
                "index": index,
                "split_seed": int(split_seed),
            }
        )
    )
    bucket = int(bucket_digest[:16], 16) / float(16**16)
    cumulative = 0.0
    for name, weight in weights:
        cumulative += weight
        if bucket < cumulative:
            return name
    return weights[-1][0]


def materialize_dataset(
    source: Path | str,
    *,
    out_dir: Path | str,
    splits: dict[str, float] | None = None,
    split_seed: int = 0,
) -> dict[str, Any]:
    """Copy a lineage slice into a portable, digest-verified dataset directory."""
    source_path = Path(source).resolve()
    dataset = validate_dataset_manifest(load_manifest(source_path))
    actual_parent = dataset_content_digest(dataset)
    declared_parent = dataset.get("digest")
    if declared_parent is not None and declared_parent != actual_parent:
        raise ConformanceError(
            f"dataset digest mismatch: declared {declared_parent}, actual {actual_parent}"
        )
    out_path = Path(out_dir)
    if out_path.exists() and (not out_path.is_dir() or any(out_path.iterdir())):
        raise SchemaError(f"materialize output must be empty or absent: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".rlx-dataset-", dir=str(out_path.parent)))
    try:
        split_weights = _normalize_split_weights(splits) if splits is not None else None
        split_counts: dict[str, int] = (
            {name: 0 for name, _weight in split_weights}
            if split_weights is not None
            else {}
        )
        episodes_dir = staging / "episodes"
        episodes_dir.mkdir()
        entries: list[dict[str, Any]] = []
        for index, entry in enumerate(dataset["episodes"]):
            if not isinstance(entry, dict) or not entry.get("path") or not entry.get("digest"):
                raise SchemaError(f"dataset.episodes[{index}] requires path and digest")
            episode_path = Path(str(entry["path"]))
            if not episode_path.is_absolute():
                episode_path = (source_path.parent / episode_path).resolve()
            if not episode_path.is_file():
                raise SchemaError(f"dataset episode not found: {episode_path}")
            actual = digest_uri(sha256_file(episode_path))
            if actual != entry["digest"]:
                raise ConformanceError(
                    f"dataset episode digest mismatch for {episode_path}: "
                    f"declared {entry['digest']}, actual {actual}"
                )
            relative = Path("episodes") / f"episode_{index:06d}.json"
            shutil.copy2(episode_path, staging / relative)
            portable_entry = {
                "path": str(relative),
                "digest": actual,
                "seed": entry.get("seed"),
            }
            if split_weights is not None:
                split = _assign_split(
                    entry_digest=actual,
                    index=index,
                    split_seed=split_seed,
                    weights=split_weights,
                )
                portable_entry["split"] = split
                split_counts[split] += 1
            entries.append(portable_entry)
        portable = {
            "schema": DATASET_SCHEMA,
            "name": dataset["name"],
            "source_runs": [f"dataset:{actual_parent}"],
            "episodes": entries,
            "query": dict(dataset.get("query") or {}),
            "lineage": {
                "parent_dataset_digest": actual_parent,
                "materialized": True,
                "episode_count": len(entries),
            },
        }
        if split_weights is not None:
            portable["splits"] = {
                "method": "sha256_bucket/v1",
                "seed": int(split_seed),
                "weights": {name: weight for name, weight in split_weights},
                "counts": split_counts,
            }
        validate_dataset_manifest(portable)
        portable["digest"] = dataset_content_digest(portable)
        dump_yaml(portable, staging / "dataset.yaml")
        dump_json(portable, staging / "dataset.json")
        if out_path.exists():
            out_path.rmdir()
        staging.replace(out_path)
        return portable
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
