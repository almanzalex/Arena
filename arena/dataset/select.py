"""Select trajectory slices with required policy+task provenance binding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from arena.core.dataset import dataset_content_digest
from arena.core.errors import ConformanceError, SchemaError
from arena.core.identity import digest_uri, parse_digest, sha256_bytes
from arena.core.manifests import DATASET_SCHEMA, dump_json, dump_yaml, validate_dataset_manifest
from arena.dataset.provenance import (
    bind_dataset_provenance,
    episode_policy_digests,
    episode_task_identity,
    task_identity,
)


def _normalize_digest(value: str) -> str:
    return f"sha256:{parse_digest(value)}"


def _outcome_matches(ep: dict[str, Any], *, role: str, outcome: str) -> bool:
    outcomes = ep.get("outcomes") or {}
    if role in outcomes:
        return outcomes.get(role) == outcome
    returns = ep.get("returns") or {}
    if role not in returns:
        return not outcomes
    reward = float(returns[role])
    if outcome == "win":
        return reward > 0
    if outcome == "loss":
        return reward < 0
    if outcome == "draw":
        return reward == 0
    return False


def _episode_matches_query(ep: dict[str, Any], *, query: dict[str, Any]) -> bool:
    if "seed" in query and ep.get("seed") != query["seed"]:
        return False

    digests = episode_policy_digests(ep)
    if "policy" in query and _normalize_digest(str(query["policy"])) not in digests:
        return False
    if "opponent" in query and _normalize_digest(str(query["opponent"])) not in digests:
        return False

    if "task" in query:
        observed = episode_task_identity(ep)
        want = task_identity(query["task"])
        if want is None or observed is None:
            return False
        for key, value in want.items():
            if observed.get(key) != value:
                return False

    role = query.get("role")
    if role is not None and "policy" in query:
        policies = ep.get("policies") or ep.get("assignments") or {}
        role_digest = policies.get(role)
        if isinstance(role_digest, dict):
            role_digest = role_digest.get("digest")
        if role_digest is None:
            return False
        if _normalize_digest(str(role_digest)) != _normalize_digest(str(query["policy"])):
            return False

    if "outcome" in query:
        if not role:
            return False
        if not _outcome_matches(ep, role=str(role), outcome=str(query["outcome"])):
            return False
    return True


def _attach_run_assignments(ep: dict[str, Any], traj_dir: Path) -> dict[str, Any]:
    run_yaml = traj_dir.parent / "run.yaml"
    if not run_yaml.exists():
        return ep
    import yaml

    run_rec = yaml.safe_load(run_yaml.read_text(encoding="utf-8")) or {}
    assigns = {
        key: (value.get("digest") if isinstance(value, dict) else value)
        for key, value in (run_rec.get("assignments") or {}).items()
    }
    enriched = dict(ep)
    if assigns and "assignments" not in enriched:
        enriched["assignments"] = assigns
    if run_rec.get("task") and "task" not in enriched:
        enriched["task"] = run_rec["task"]
    return enriched


def _collect_episodes(
    *,
    source_runs: list[str | Path],
    query: dict[str, Any],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for run in source_runs:
        run_path = Path(run)
        traj_dirs: list[Path] = []
        if (run_path / "trajectories").is_dir():
            traj_dirs.append(run_path / "trajectories")
        else:
            traj_dirs.extend(sorted(run_path.glob("**/trajectories")))
        for traj_dir in traj_dirs:
            for ep_path in sorted(traj_dir.glob("episode_*.json")):
                ep = json.loads(ep_path.read_text(encoding="utf-8"))
                if not isinstance(ep, dict):
                    raise SchemaError(f"episode must be a JSON object: {ep_path}")
                ep = _attach_run_assignments(ep, traj_dir)
                if not _episode_matches_query(ep, query=query):
                    continue
                selected.append(
                    {
                        "path": str(ep_path.resolve()),
                        "digest": digest_uri(sha256_bytes(ep_path.read_bytes())),
                        "seed": ep.get("seed"),
                        "source_run": str(traj_dir.parent.resolve()),
                    }
                )
    return selected


def select_bound_episodes(
    *,
    source_runs: list[str | Path],
    query: dict[str, Any] | None = None,
    name: str = "slice",
    out_dir: Path | str | None = None,
    policy_digest: str | None = None,
    task: dict[str, Any] | str | None = None,
    role: str | None = None,
    require_policy: bool = True,
    allow_empty: bool = False,
) -> dict[str, Any]:
    """Filter episodes using episode-native ``policies``/task fields, then bind.

    Unlike ``arena.core.dataset.select_episodes`` (which primarily checks run
    ``assignments``), this path treats trajectory ``policies`` as authoritative
    provenance, stamps ``provenance``, and fail-loud verifies episode bytes.
    """
    query = dict(query or {})
    query_policy = query.get("policy")
    if policy_digest is not None and query_policy is not None:
        if parse_digest(str(policy_digest)) != parse_digest(str(query_policy)):
            raise ConformanceError(
                "policy digest mismatch between policy_digest and query['policy']",
                code="DATASET_POLICY_DIGEST_MISMATCH",
                cause="select arguments disagree on the bound policy identity",
                repair="Pass the same digest to policy_digest and query['policy'].",
                context={"policy_digest": policy_digest, "query_policy": query_policy},
            )
    bound_policy = policy_digest or query_policy
    if bound_policy is None and require_policy:
        raise SchemaError(
            "select_bound_episodes requires policy_digest (or query['policy']) "
            "so the dataset is bound to a policy identity",
            code="DATASET_POLICY_REQUIRED",
            repair="Pass policy_digest=... or include policy in the query.",
        )

    if task is not None and "task" not in query:
        identity = task_identity(task)
        assert identity is not None
        query = {**query, "task": identity["env"]}
    if role is not None and "role" not in query:
        query = {**query, "role": role}
    if bound_policy is not None and "policy" not in query:
        query = {**query, "policy": bound_policy}

    selected = _collect_episodes(source_runs=source_runs, query=query)
    dataset: dict[str, Any] = {
        "schema": DATASET_SCHEMA,
        "name": name,
        "source_runs": [str(Path(run).resolve()) for run in source_runs],
        "episodes": selected,
        "query": query,
        "lineage": {
            "note": (
                "episodes reference immutable source paths/digests; sources are "
                "not rewritten; provenance binds policy+task identity"
            ),
        },
    }
    validate_dataset_manifest(dataset)

    if not selected and not allow_empty:
        raise ConformanceError(
            "no episodes matched the provenance-bound select query",
            code="DATASET_SELECT_EMPTY",
            cause="filters excluded every trajectory under the source runs",
            repair="Widen the query or confirm the policy/task digests are present.",
            context={"query": query, "source_runs": dataset["source_runs"]},
        )

    if bound_policy is None:
        dataset["digest"] = dataset_content_digest(dataset)
        if out_dir is not None:
            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            dump_yaml(dataset, out_path / "dataset.yaml")
            dump_json(dataset, out_path / "dataset.json")
        return dataset

    bound = bind_dataset_provenance(
        dataset,
        policy_digest=str(bound_policy),
        task=task if task is not None else query.get("task"),
        role=role if role is not None else query.get("role"),
        verify_episodes=True,
    )
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        dump_yaml(bound, out_path / "dataset.yaml")
        dump_json(bound, out_path / "dataset.json")
    return bound
