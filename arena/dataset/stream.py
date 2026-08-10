"""Stream-read verified episodes without copying into a portable tree.

Spike companion to RFC 012. Prefer ``materialize_dataset`` when the producer
run may disappear; use this iterator when sources remain available and a full
copy is unnecessary.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from arena.core.errors import ConformanceError, SchemaError
from arena.core.identity import digest_uri, sha256_file
from arena.core.manifests import load_manifest, validate_dataset_manifest
from arena.dataset.provenance import load_episode


def _resolve_root(
    dataset: dict[str, Any],
    dataset_path: Path | str | None,
) -> Path | None:
    if dataset_path is not None:
        path = Path(dataset_path)
        return path.parent if path.is_file() else path
    lineage = dataset.get("lineage") or {}
    if lineage.get("materialized"):
        return None
    return None


def _load_dataset(
    source: Path | str | dict[str, Any],
) -> tuple[dict[str, Any], Path | None]:
    if isinstance(source, dict):
        return validate_dataset_manifest(source), None
    path = Path(source)
    dataset = validate_dataset_manifest(load_manifest(path))
    return dataset, path


def iter_verified_episodes(
    source: Path | str | dict[str, Any],
    *,
    dataset_path: Path | str | None = None,
    split: str | None = None,
    verify_digests: bool = True,
) -> Iterator[tuple[int, dict[str, Any], dict[str, Any]]]:
    """Yield ``(index, entry, episode)`` with optional digest verification.

    Does not copy episode bytes. Relative paths require ``dataset_path`` (or a
    path-form ``source``) so files resolve under the dataset root.

    Non-goals (RFC 012): object-store readers, replacing atomic materialize,
    and recomputing the dataset content digest during iteration.
    """
    dataset, loaded_path = _load_dataset(source)
    root = _resolve_root(dataset, dataset_path if dataset_path is not None else loaded_path)
    if split is not None and not str(split).strip():
        raise SchemaError("split filter must be non-empty when provided")

    for index, entry in enumerate(dataset.get("episodes") or []):
        if not isinstance(entry, dict) or not entry.get("path"):
            raise SchemaError(f"dataset.episodes[{index}] requires path")
        if split is not None and entry.get("split") != split:
            continue
        path = Path(str(entry["path"]))
        if not path.is_absolute():
            if root is None:
                raise SchemaError(
                    f"dataset.episodes[{index}] has a relative path; pass "
                    "dataset_path (or a file source) so stream-read can resolve it"
                )
            path = (root / path).resolve()
        if not path.is_file():
            raise SchemaError(f"dataset episode not found: {path}")
        if verify_digests:
            declared = entry.get("digest")
            if not declared:
                raise SchemaError(f"dataset.episodes[{index}] requires digest")
            actual = digest_uri(sha256_file(path))
            if actual != declared:
                raise ConformanceError(
                    f"dataset episode digest mismatch for {path}: "
                    f"declared {declared}, actual {actual}",
                    code="DATASET_EPISODE_DIGEST_MISMATCH",
                    cause="episode bytes disagree with the declared content digest",
                    repair="Re-select or re-materialize the dataset from trusted sources.",
                    context={
                        "index": index,
                        "path": str(path),
                        "declared": declared,
                        "actual": actual,
                    },
                )
        yield index, entry, load_episode(path)
