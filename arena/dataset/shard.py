"""Sharded portable dataset materialization (RFC 011 spike).

Preserves the atomic publish contract from ``materialize_dataset`` (#4) while
placing episode files under ``episodes/shard_XXXX/``. Flat materialize remains
the default path in ``arena.core.dataset`` and is intentionally unchanged.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from arena.core.dataset import (
    _assign_split,
    _looks_like_complete_dataset,
    _normalize_split_weights,
    dataset_content_digest,
)
from arena.core.errors import ConformanceError, SchemaError
from arena.core.identity import digest_uri, sha256_file
from arena.core.io import publish_directory
from arena.core.manifests import (
    DATASET_SCHEMA,
    dump_json,
    dump_yaml,
    load_manifest,
    validate_dataset_manifest,
)

SHARD_METHOD = "index_mod/v1"


def shard_id_for_index(index: int, shard_count: int) -> int:
    """Deterministic shard assignment: ``index % shard_count``."""
    if shard_count < 1:
        raise SchemaError("shard_count must be a positive integer")
    if index < 0:
        raise SchemaError("episode index must be non-negative")
    return int(index) % int(shard_count)


def shard_dirname(shard_id: int, *, width: int = 4) -> str:
    return f"shard_{int(shard_id):0{width}d}"


def materialize_dataset_sharded(
    source: Path | str,
    *,
    out_dir: Path | str,
    shard_count: int,
    splits: dict[str, float] | None = None,
    split_seed: int = 0,
) -> dict[str, Any]:
    """Copy a lineage slice into a sharded, digest-verified portable directory.

    Same atomic staging guarantees as ``materialize_dataset``. Episode content
    digests are unchanged; relative paths include the shard directory, so the
    dataset content digest differs from a flat materialization of the same
    source (explicit non-goal in RFC 011).
    """
    if not isinstance(shard_count, int) or isinstance(shard_count, bool) or shard_count < 1:
        raise SchemaError("shard_count must be a positive integer")

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
    if out_path.exists() and out_path.is_dir() and not any(out_path.iterdir()):
        out_path.rmdir()

    split_weights = _normalize_split_weights(splits) if splits is not None else None

    def build(staging: Path) -> dict[str, Any]:
        split_counts: dict[str, int] = (
            {name: 0 for name, _weight in split_weights}
            if split_weights is not None
            else {}
        )
        shard_counts = {shard_dirname(i): 0 for i in range(shard_count)}
        episodes_dir = staging / "episodes"
        episodes_dir.mkdir()
        for shard in range(shard_count):
            (episodes_dir / shard_dirname(shard)).mkdir()

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
            sid = shard_id_for_index(index, shard_count)
            shard_name = shard_dirname(sid)
            relative = Path("episodes") / shard_name / f"episode_{index:06d}.json"
            shutil.copy2(episode_path, staging / relative)
            shard_counts[shard_name] += 1
            portable_entry: dict[str, Any] = {
                "path": str(relative),
                "digest": actual,
                "seed": entry.get("seed"),
                "shard": sid,
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

        portable: dict[str, Any] = {
            "schema": DATASET_SCHEMA,
            "name": dataset["name"],
            "source_runs": [f"dataset:{actual_parent}"],
            "episodes": entries,
            "query": dict(dataset.get("query") or {}),
            "lineage": {
                "parent_dataset_digest": actual_parent,
                "materialized": True,
                "episode_count": len(entries),
                "sharded": True,
                "shard_count": int(shard_count),
                "shard_method": SHARD_METHOD,
                "shard_counts": shard_counts,
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
        return portable

    def verify(staging: Path) -> None:
        if not _looks_like_complete_dataset(staging):
            raise ConformanceError(
                f"sharded materialize staging incomplete (missing dataset manifests): "
                f"{staging}"
            )
        published = validate_dataset_manifest(load_manifest(staging / "dataset.yaml"))
        actual = dataset_content_digest(published)
        declared = published.get("digest")
        if declared != actual:
            raise ConformanceError(
                f"sharded materialize staging digest mismatch: "
                f"declared {declared}, actual {actual}"
            )
        lineage = published.get("lineage") or {}
        if not lineage.get("sharded"):
            raise ConformanceError("sharded materialize staging missing lineage.sharded")
        if int(lineage.get("shard_count", -1)) != shard_count:
            raise ConformanceError("sharded materialize staging shard_count mismatch")
        if int(lineage.get("episode_count", -1)) != len(published.get("episodes") or []):
            raise ConformanceError("sharded materialize staging episode_count mismatch")
        for index, entry in enumerate(published.get("episodes") or []):
            expected = shard_id_for_index(index, shard_count)
            if entry.get("shard") != expected:
                raise ConformanceError(
                    f"sharded materialize staging shard id mismatch at {index}: "
                    f"expected {expected}, got {entry.get('shard')}"
                )
            expected_prefix = f"episodes/{shard_dirname(expected)}/"
            if not str(entry.get("path", "")).startswith(expected_prefix):
                raise ConformanceError(
                    f"sharded materialize staging path not under {expected_prefix}: "
                    f"{entry.get('path')}"
                )

    return publish_directory(out_path, build, verify=verify, replace=False)
