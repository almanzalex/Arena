"""Bind datasets/trajectory slices to policy + task identity.

Offline datasets that only store episode digests can silently drift from the
policy/task that produced them. These helpers stamp an explicit provenance
binding and fail loud when episode bytes disagree with the claimed digests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from arena.core.dataset import dataset_content_digest
from arena.core.errors import ConformanceError, SchemaError
from arena.core.identity import parse_digest
from arena.core.manifests import validate_dataset_manifest

PROVENANCE_BINDING_SCHEMA = "arena.dataset-binding/v1"

_TASK_KEYS = ("env", "adapter", "version")


def task_identity(task: dict[str, Any] | str | None) -> dict[str, Any] | None:
    """Return the canonical task identity subset used for provenance binding."""
    if task is None:
        return None
    if isinstance(task, str):
        text = task.strip()
        if not text:
            raise SchemaError("task identity env must be non-empty")
        return {"env": text}
    if not isinstance(task, dict):
        raise SchemaError("task identity must be a mapping or env string")
    identity = {key: task[key] for key in _TASK_KEYS if key in task and task[key] is not None}
    if "env" not in identity or not str(identity["env"]).strip():
        raise SchemaError("task identity requires a non-empty env")
    return {key: str(identity[key]) for key in _TASK_KEYS if key in identity}


def _normalize_digest(value: str) -> str:
    return f"sha256:{parse_digest(value)}"


def episode_policy_digests(episode: dict[str, Any]) -> set[str]:
    """Collect policy digests declared on an episode (policies and/or assignments)."""
    digests: set[str] = set()
    for key in ("policies", "assignments"):
        mapping = episode.get(key)
        if not isinstance(mapping, dict):
            continue
        for raw in mapping.values():
            if isinstance(raw, dict):
                raw = raw.get("digest")
            if raw is None:
                continue
            digests.add(_normalize_digest(str(raw)))
    return digests


def episode_task_identity(episode: dict[str, Any]) -> dict[str, Any] | None:
    task = episode.get("task")
    if task is None:
        return None
    return task_identity(task)


def load_episode(path: Path | str, *, dataset_root: Path | None = None) -> dict[str, Any]:
    episode_path = Path(path)
    if not episode_path.is_absolute() and dataset_root is not None:
        episode_path = (dataset_root / episode_path).resolve()
    if not episode_path.is_file():
        raise SchemaError(f"dataset episode not found: {episode_path}")
    payload = json.loads(episode_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SchemaError(f"episode must be a JSON object: {episode_path}")
    return payload


def _resolve_dataset_root(dataset: dict[str, Any], dataset_path: Path | str | None) -> Path | None:
    if dataset_path is not None:
        path = Path(dataset_path)
        return path.parent if path.is_file() else path
    lineage = dataset.get("lineage") or {}
    if lineage.get("materialized"):
        # Materialized datasets use relative episode paths; caller must pass dataset_path.
        return None
    return None


def _episode_paths(dataset: dict[str, Any], *, dataset_root: Path | None) -> list[tuple[int, Path, dict[str, Any]]]:
    episodes = dataset.get("episodes")
    if not isinstance(episodes, list):
        raise SchemaError("dataset.episodes must be a list")
    resolved: list[tuple[int, Path, dict[str, Any]]] = []
    for index, entry in enumerate(episodes):
        if not isinstance(entry, dict) or not entry.get("path"):
            raise SchemaError(f"dataset.episodes[{index}] requires path")
        path = Path(str(entry["path"]))
        if not path.is_absolute():
            if dataset_root is None:
                raise SchemaError(
                    f"dataset.episodes[{index}] has a relative path; pass dataset_path "
                    "so provenance can resolve episode files"
                )
            path = (dataset_root / path).resolve()
        resolved.append((index, path, entry))
    return resolved


def _tasks_compatible(declared: dict[str, Any], observed: dict[str, Any] | None) -> bool:
    if observed is None:
        return False
    for key, value in declared.items():
        if observed.get(key) != value:
            return False
    return True


def _verify_episodes_against_binding(
    dataset: dict[str, Any],
    binding: dict[str, Any],
    *,
    dataset_root: Path | None,
) -> None:
    policy = _normalize_digest(str(binding["policy_digest"]))
    task = binding.get("task")
    role = binding.get("role")
    mismatches: list[dict[str, Any]] = []
    for index, path, _entry in _episode_paths(dataset, dataset_root=dataset_root):
        episode = load_episode(path)
        digests = episode_policy_digests(episode)
        if policy not in digests:
            mismatches.append(
                {
                    "index": index,
                    "path": str(path),
                    "expected_policy": policy,
                    "episode_policies": sorted(digests),
                    "kind": "policy_digest_mismatch",
                }
            )
            continue
        if role is not None:
            policies = episode.get("policies") or episode.get("assignments") or {}
            role_digest = policies.get(role)
            if isinstance(role_digest, dict):
                role_digest = role_digest.get("digest")
            if role_digest is None or _normalize_digest(str(role_digest)) != policy:
                mismatches.append(
                    {
                        "index": index,
                        "path": str(path),
                        "expected_policy": policy,
                        "role": role,
                        "role_policy": role_digest,
                        "kind": "role_policy_digest_mismatch",
                    }
                )
                continue
        if task is not None:
            observed = episode_task_identity(episode)
            if not _tasks_compatible(task, observed):
                mismatches.append(
                    {
                        "index": index,
                        "path": str(path),
                        "expected_task": task,
                        "episode_task": observed,
                        "kind": "task_identity_mismatch",
                    }
                )
    if mismatches:
        first = mismatches[0]
        raise ConformanceError(
            f"dataset provenance mismatch: {first['kind']} for episode "
            f"{first['index']} ({first['path']})",
            code="DATASET_PROVENANCE_MISMATCH",
            cause="episode bytes disagree with bound policy/task identity",
            repair=(
                "Re-select the slice with the correct --policy/--task, or unbind "
                "and re-bind only after verifying episode provenance."
            ),
            context={"mismatches": mismatches, "binding": binding},
        )


def bind_dataset_provenance(
    dataset: dict[str, Any],
    *,
    policy_digest: str,
    task: dict[str, Any] | str | None = None,
    role: str | None = None,
    verify_episodes: bool = True,
    dataset_path: Path | str | None = None,
) -> dict[str, Any]:
    """Stamp policy+task provenance onto a dataset and optionally verify episodes.

    Verification is fail-loud: any episode that does not carry the claimed policy
    digest (and optional task/role) raises ``ConformanceError``.
    """
    validate_dataset_manifest(dataset)
    policy = _normalize_digest(policy_digest)
    identity = task_identity(task)
    if role is not None and not str(role).strip():
        raise SchemaError("provenance role must be non-empty when provided")
    binding: dict[str, Any] = {
        "schema": PROVENANCE_BINDING_SCHEMA,
        "policy_digest": policy,
    }
    if identity is not None:
        binding["task"] = identity
    if role is not None:
        binding["role"] = str(role)
    bound = dict(dataset)
    bound["provenance"] = binding
    if verify_episodes:
        root = _resolve_dataset_root(bound, dataset_path)
        _verify_episodes_against_binding(bound, binding, dataset_root=root)
    bound["digest"] = dataset_content_digest(bound)
    return bound


def unbind_dataset_provenance(dataset: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``dataset`` without provenance binding (digest recomputed)."""
    validate_dataset_manifest(dataset)
    unbound = {key: value for key, value in dataset.items() if key != "provenance"}
    unbound["digest"] = dataset_content_digest(unbound)
    return unbound


def verify_dataset_provenance(
    dataset: dict[str, Any],
    *,
    expect_policy: str | None = None,
    expect_task: dict[str, Any] | str | None = None,
    dataset_path: Path | str | None = None,
) -> dict[str, Any]:
    """Verify a bound dataset against episode bytes and optional expected digests.

    Raises ``SchemaError`` when provenance is missing, and ``ConformanceError``
    when the bound or expected policy/task identity disagrees with episode bytes.
    """
    validate_dataset_manifest(dataset)
    binding = dataset.get("provenance")
    if not isinstance(binding, dict):
        raise SchemaError(
            "dataset is not provenance-bound; call bind_dataset_provenance first",
            code="DATASET_PROVENANCE_MISSING",
            repair="Bind the dataset with policy_digest (and optional task/role).",
        )
    if binding.get("schema") != PROVENANCE_BINDING_SCHEMA:
        raise SchemaError(
            f"unsupported provenance schema: {binding.get('schema')!r}",
            code="DATASET_PROVENANCE_SCHEMA",
        )
    if "policy_digest" not in binding:
        raise SchemaError("provenance binding missing policy_digest")

    bound_policy = _normalize_digest(str(binding["policy_digest"]))
    if expect_policy is not None:
        expected = _normalize_digest(expect_policy)
        if expected != bound_policy:
            raise ConformanceError(
                f"policy digest mismatch: expected {expected}, bound {bound_policy}",
                code="DATASET_POLICY_DIGEST_MISMATCH",
                cause="caller expected a different policy than the dataset binding",
                repair="Load the dataset bound to the intended policy digest.",
                context={"expected": expected, "bound": bound_policy},
            )

    bound_task = binding.get("task")
    if expect_task is not None:
        expected_task = task_identity(expect_task)
        assert expected_task is not None
        if not isinstance(bound_task, dict) or not _tasks_compatible(expected_task, bound_task):
            raise ConformanceError(
                f"task identity mismatch: expected {expected_task}, bound {bound_task}",
                code="DATASET_TASK_IDENTITY_MISMATCH",
                cause="caller expected a different task than the dataset binding",
                repair="Re-bind the dataset with the intended task identity.",
                context={"expected": expected_task, "bound": bound_task},
            )

    root = _resolve_dataset_root(dataset, dataset_path)
    _verify_episodes_against_binding(dataset, binding, dataset_root=root)
    return binding
