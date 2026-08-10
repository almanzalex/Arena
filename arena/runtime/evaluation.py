"""Versioned evaluation suites: expand to match jobs and record sampling ledgers."""

from __future__ import annotations

import sys
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from arena.adapters.task_pettingzoo.adapter import describe_task
from arena.core.compatibility import compose_check
from arena.core.errors import (
    CompatibilityError,
    ExternalUnavailableError,
    IncompleteExecutionError,
    SchemaError,
    redact,
)
from arena.core.identity import canonical_json, digest_uri, sha256_bytes, sha256_canonical
from arena.core.io import atomic_write_bytes, publish_directory
from arena.core.manifests import (
    EVAL_RUN_V1_SCHEMA,
    dump_json,
    dump_yaml,
    evaluation_binding_digest,
    evaluation_content_digest,
    evaluation_intent_digest,
    evaluation_intent_projection,
    expand_seeds,
    load_manifest,
    task_content_digest,
    validate_eval_report_manifest,
    validate_eval_run_manifest,
    validate_evaluation_manifest,
)
from arena.core.population import assert_members_compatible_with_role, load_population
from arena.core.sdk import Policy, Task
from arena.core.store import LocalStore
from arena.core.supervisor import run_supervised
from arena.plugins import metrics as metrics_plugins
from arena.plugins import samplers as sampler_plugins


def _policy_from_digest_or_path(
    ref: str,
    *,
    policy_index: dict[str, Path],
) -> Policy:
    if ref in policy_index:
        return Policy.load(policy_index[ref])
    path = Path(ref)
    if path.exists():
        return Policy.load(path)
    raise SchemaError(
        f"cannot load policy {ref!r}; pass policy_index mapping digests to bundle paths"
    )


def load_evaluation(path: Path | str) -> dict[str, Any]:
    data = load_manifest(path)
    return validate_evaluation_manifest(data)


def _identity_suite(
    suite: dict[str, Any], *, policy_index: dict[str, Path]
) -> dict[str, Any]:
    """Replace movable policy paths/names with immutable policy digests."""
    assignments: dict[str, Any] = {}
    for role, spec in suite["assignments"].items():
        if isinstance(spec, str):
            try:
                assignments[role] = _policy_from_digest_or_path(
                    spec, policy_index=policy_index
                ).digest
            except SchemaError:
                assignments[role] = spec
        elif isinstance(spec, dict) and spec.get("kind", "policy") == "policy":
            key = "policy" if "policy" in spec else "ref"
            ref = str(spec[key])
            try:
                digest = _policy_from_digest_or_path(ref, policy_index=policy_index).digest
            except SchemaError:
                digest = ref
            assignments[role] = {**spec, key: digest}
        else:
            assignments[role] = spec
    return {**suite, "assignments": assignments}


def validate_evaluation(
    suite: dict[str, Any],
    *,
    populations: dict[str, dict[str, Any]] | None = None,
    policy_index: dict[str, Path] | None = None,
) -> dict[str, Any]:
    suite = validate_evaluation_manifest(suite)
    populations = populations or {}
    policy_index = policy_index or {}
    # Role swaps require transforms (already schema-enforced); check maps.
    for swap in suite.get("role_swaps") or []:
        mapping = swap.get("map") or {}
        if not mapping:
            raise SchemaError("role_swaps[].map must be non-empty")
        # Transform present but unknown kinds fail loud.
        transform = swap.get("transform")
        if transform not in {"identity", "symmetric", "declared"}:
            raise SchemaError(
                f"unsupported role_swaps.transform {transform!r}; "
                "use identity|symmetric|declared or register a role_transform case"
            )
    task = Task.load(suite["task"])
    interaction = suite.get("interaction", "parallel")
    task_interaction = task.spec.get("interaction", "parallel")
    if task_interaction != interaction:
        # Allow task to omit interaction when suite declares it.
        if "interaction" in task.spec and task_interaction != interaction:
            raise CompatibilityError(
                f"suite interaction {interaction!r} mismatches task interaction {task_interaction!r}"
            )
    for role, spec in suite["assignments"].items():
        if isinstance(spec, str) or (isinstance(spec, dict) and spec.get("kind", "policy") == "policy"):
            continue
        if isinstance(spec, dict) and spec.get("kind") in {"population", "crossplay"}:
            pop_ref = spec["population"]
            pop = populations.get(str(pop_ref))
            if pop is None and Path(str(pop_ref)).exists():
                pop = load_population(pop_ref)
            if pop is None:
                raise SchemaError(f"population {pop_ref!r} not provided for validate")
            assert_members_compatible_with_role(pop, role)
    return suite


def expand_evaluation_cells(
    suite: dict[str, Any],
    *,
    populations: dict[str, dict[str, Any]],
    sampling_seed: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Expand suite into concrete assignment cells + sampling ledger."""
    sampler_plugins.register_builtins()
    seeds = expand_seeds(suite["seeds"])
    sampling = suite.get("sampling") or {}
    default_sampler = sampling.get("kind", "enumerated_crossplay")
    cells: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    base_assignments: dict[str, Any] = {}
    cross_roles: list[tuple[str, dict[str, Any]]] = []

    for role, spec in suite["assignments"].items():
        if isinstance(spec, str):
            base_assignments[role] = {"kind": "policy", "policy": spec}
        elif isinstance(spec, dict) and spec.get("kind", "policy") == "policy":
            base_assignments[role] = {
                "kind": "policy",
                "policy": spec.get("policy") or spec.get("ref"),
            }
        elif isinstance(spec, dict) and spec.get("kind") in {"population", "crossplay"}:
            cross_roles.append((role, spec))
        else:
            raise SchemaError(f"unsupported assignment for {role}: {spec!r}")

    if not cross_roles:
        cells.append(
            {
                "cell_id": "cell-0",
                "assignments": {r: a["policy"] for r, a in base_assignments.items()},
                "seeds": seeds,
            }
        )
        return cells, ledger

    if len(cross_roles) > 2:
        raise SchemaError(
            "Arena supports at most two crossplay/population assignment roles "
            "(cartesian cross-play matrix)"
        )

    role_draws: list[tuple[str, list[dict[str, Any]]]] = []
    for role, spec in cross_roles:
        pop = populations.get(str(spec["population"]))
        if pop is None:
            raise SchemaError(f"population {spec['population']!r} missing for expansion")
        assert_members_compatible_with_role(pop, role)
        kind = spec.get("kind", "crossplay")
        sampler_kind = default_sampler if kind == "population" else "enumerated_crossplay"
        sampler = sampler_plugins.SAMPLERS.get(sampler_kind)
        draws = sampler.sample(
            list(pop["members"]),
            seed=int(sampling.get("seed", sampling_seed)),
            stream=f"eval:{role}",
            n=int(sampling.get("n", 1)),
        )
        ledger.extend([{**d, "role": role} for d in draws])
        role_draws.append((role, draws))

    # Cartesian product across crossplay roles (1-role = identity product).
    combos: list[dict[str, dict[str, Any]]] = [{}]
    for role, draws in role_draws:
        next_combos: list[dict[str, dict[str, Any]]] = []
        for base in combos:
            for draw in draws:
                next_combos.append({**base, role: draw})
        combos = next_combos

    for i, combo in enumerate(combos):
        assigns = {r: a["policy"] for r, a in base_assignments.items()}
        sampling_meta: dict[str, Any] = {}
        for role, draw in combo.items():
            assigns[role] = draw["policy"]
            sampling_meta[role] = draw
        cells.append(
            {
                "cell_id": f"cell-{i}",
                "assignments": assigns,
                "seeds": seeds,
                "candidate_policy": assigns.get("player_0"),
                "opponent_policy": assigns.get("player_1"),
                "sampling": sampling_meta if len(sampling_meta) > 1 else next(iter(sampling_meta.values())),
            }
        )
    return cells, ledger


def _execute_evaluation_cell(
    *,
    cell: dict[str, Any],
    suite: dict[str, Any],
    task_spec: dict[str, Any],
    task_info: dict[str, Any],
    task_digest: str,
    provider_lineage: dict[str, Any],
    policy_index: dict[str, Path],
    run_root: Path,
    record: bool,
) -> dict[str, Any]:
    from arena.plugins.interactions import get_interaction

    assignments: dict[str, Policy] = {}
    for role, pref in cell["assignments"].items():
        policy = _policy_from_digest_or_path(str(pref), policy_index=policy_index)
        meta = task_info["roles"].get(role)
        if meta is None:
            raise CompatibilityError(
                f"assignment role {role!r} not in task agents {list(task_info['roles'])}"
            )
        report = compose_check(
            policy=policy.manifest,
            role=role,
            expected_obs=meta.get("observation"),
            expected_act=meta.get("action"),
            action_mode=suite.get("action_mode"),
            task_provides_masks=bool(task_info.get("provides_masks")),
        )
        if not report.ok:
            raise CompatibilityError(str(report))
        assignments[role] = policy
    # Bind immutable digests into the claim surface (never leave movable paths).
    bound_assignments = {role: policy.digest for role, policy in assignments.items()}
    match_out = run_root / cell["cell_id"]
    result = get_interaction(suite.get("interaction", "parallel")).run_match(
        task_spec=task_spec,
        assignments=assignments,
        seeds=list(cell["seeds"]),
        action_mode=suite.get("action_mode", "deterministic"),
        record=record,
        out_dir=match_out,
        failure_policy=suite.get("failure_policy"),
    )
    episodes = []
    evidence_refs = []
    traj_dir = match_out / "trajectories"
    if traj_dir.exists():
        for ep_path in sorted(traj_dir.glob("episode_*.json")):
            ep = load_manifest(ep_path)
            episodes.append(
                {
                    "path": str(ep_path),
                    "seed": ep.get("seed"),
                    "returns": ep.get("returns") or _episode_returns(ep),
                    "outcomes": ep.get("outcomes") or {},
                }
            )
            evidence_refs.append(str(ep_path.relative_to(run_root)))
        bundle = traj_dir / "bundle.json"
        if bundle.exists():
            evidence_refs.insert(0, str(bundle.relative_to(run_root)))
    return {
        **cell,
        "assignments": bound_assignments,
        "run": result,
        "episodes": episodes,
        "evidence_refs": evidence_refs,
        "failures": len(result.get("failures") or []),
        "lineage": {
            "policy_digests": sorted(set(bound_assignments.values())),
            "task_digest": task_digest,
            "provider": provider_lineage,
        },
    }


def _supervised_failure_kind(exc: BaseException) -> str:
    """Map hard-budget executor faults onto ledger kinds (never fake success)."""
    code = getattr(exc, "code", None)
    if isinstance(exc, ExternalUnavailableError):
        if code == "EXTERNAL_TIMEOUT":
            return "timeout"
        if code in {"EXTERNAL_STDOUT_LIMIT", "EXTERNAL_STDERR_LIMIT"}:
            return "executor_budget"
        if code == "EXTERNAL_START_FAILED":
            return "executor_start_failed"
    message = str(exc).lower()
    if "timeout" in message or "wall time" in message:
        return "timeout"
    return "executor_failure"


def _bound_policy_assignments(
    cell: dict[str, Any], *, policy_index: dict[str, Path]
) -> dict[str, str]:
    """Resolve cell assignment refs to sha256 digests for claim binding."""
    bound: dict[str, str] = {}
    for role, pref in (cell.get("assignments") or {}).items():
        value = str(pref)
        if value.startswith("sha256:"):
            bound[role] = value
            continue
        try:
            bound[role] = _policy_from_digest_or_path(value, policy_index=policy_index).digest
        except SchemaError:
            # Leave unresolved refs for the ledger; report building refuses them.
            bound[role] = value
    return bound


def _supervised_cell_failure(
    *,
    cell: dict[str, Any],
    task_digest: str,
    provider_lineage: dict[str, Any],
    exc: BaseException,
    policy_index: dict[str, Path] | None = None,
) -> dict[str, Any]:
    safe_message = str(redact(str(exc)))
    kind = _supervised_failure_kind(exc)
    code = getattr(exc, "code", None)
    failures: list[dict[str, Any]] = []
    for index, seed in enumerate(cell["seeds"]):
        entry: dict[str, Any] = {
            "episode_index": index,
            "seed": seed,
            "kind": kind,
            "message": safe_message,
        }
        if isinstance(code, str) and code:
            entry["code"] = code
        failures.append(entry)
    bound_assignments = _bound_policy_assignments(cell, policy_index=policy_index or {})
    return {
        **cell,
        "assignments": bound_assignments,
        "run": {
            "failures": failures,
            "outcome": {
                "episodes_requested": len(cell["seeds"]),
                "episodes_completed": 0,
                "failure_count": len(failures),
            },
            "status": "failed",
        },
        "episodes": [],
        "evidence_refs": [],
        "failures": len(failures),
        "lineage": {
            "policy_digests": sorted(set(bound_assignments.values())),
            "task_digest": task_digest,
            "provider": provider_lineage,
        },
    }


def _run_cell_supervised(
    *,
    cell: dict[str, Any],
    suite: dict[str, Any],
    task_spec: dict[str, Any],
    task_info: dict[str, Any],
    task_digest: str,
    provider_lineage: dict[str, Any],
    policy_index: dict[str, Path],
    run_root: Path,
    record: bool,
    timeout_seconds: float,
    max_stdout_bytes: int,
    max_stderr_bytes: int,
) -> dict[str, Any]:
    request_id = str(uuid.uuid4())
    request: dict[str, Any] = {
        "schema": "arena.eval-cell-request/v1",
        "request_id": request_id,
        "cell": cell,
        "suite": suite,
        "task_spec": task_spec,
        "task_info": task_info,
        "task_digest": task_digest,
        "provider_lineage": provider_lineage,
        "policy_index": {
            key: str(value.resolve()) for key, value in policy_index.items()
        },
        "run_root": str(run_root.resolve()),
        "record": bool(record),
    }
    request_digest = digest_uri(sha256_bytes(canonical_json(request)))
    request["request_digest"] = request_digest
    with tempfile.TemporaryDirectory(prefix="arena-eval-cell-") as raw:
        request_path = Path(raw) / "request.json"
        response_path = Path(raw) / "response.json"
        atomic_write_bytes(request_path, canonical_json(request) + b"\n")
        completed = run_supervised(
            [
                sys.executable,
                "-m",
                "arena.runtime.eval_worker",
                str(request_path),
                str(response_path),
            ],
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"evaluation cell worker exited {completed.returncode}: "
                f"{completed.stderr[-2000:]}"
            )
        response = load_manifest(response_path, max_bytes=64 * 1024 * 1024)
    if response.get("schema") != "arena.eval-cell-response/v1":
        raise RuntimeError("evaluation cell worker returned an unsupported response")
    if response.get("request_id") != request_id:
        raise RuntimeError("evaluation cell worker request_id mismatch")
    if response.get("request_digest") != request_digest:
        raise RuntimeError("evaluation cell worker request digest mismatch")
    result = response.get("result")
    if response.get("ok") is not True or not isinstance(result, dict):
        raise RuntimeError("evaluation cell worker returned an invalid result")
    return result


def run_evaluation(
    suite: dict[str, Any],
    *,
    policy_index: dict[str, Path],
    populations: dict[str, dict[str, Any]] | None = None,
    store: LocalStore | None = None,
    out_dir: Path | None = None,
    workers: int = 1,
    record: bool = True,
    provider: str | None = None,
) -> dict[str, Any]:
    """Dispatch a suite through the registered evaluation-provider axis."""
    from arena.core.registry import EVAL_PROVIDERS, ensure_plugins_loaded

    suite = validate_evaluation_manifest(dict(suite))
    provider_kind = provider or str(suite.get("provider", "native"))
    if provider is not None:
        suite = {**suite, "provider": provider_kind}
    ensure_plugins_loaded()
    return EVAL_PROVIDERS.get(provider_kind).run(
        suite,
        identity_suite=_identity_suite(suite, policy_index=policy_index),
        policy_index=policy_index,
        populations=populations,
        store=store,
        out_dir=out_dir,
        workers=workers,
        record=record,
    )


def _run_native_evaluation(
    suite: dict[str, Any],
    *,
    policy_index: dict[str, Path],
    populations: dict[str, dict[str, Any]] | None = None,
    store: LocalStore | None = None,
    out_dir: Path | None = None,
    workers: int = 1,
    record: bool = True,
    provider_lineage: dict[str, Any] | None = None,
    identity_suite: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish an explicit evaluation output only after the run is coherent.

    Match writers operate inside a same-parent staging directory. A crash or
    validation failure removes that staging tree; readers can therefore never
    confuse a partially written explicit ``--out`` path for a completed run.
    """
    if out_dir is None:
        return _run_native_evaluation_impl(
            suite,
            policy_index=policy_index,
            populations=populations,
            store=store,
            out_dir=None,
            workers=workers,
            record=record,
            provider_lineage=provider_lineage,
            identity_suite=identity_suite,
        )
    final = Path(out_dir)
    stage_used: Path | None = None

    def build(stage: Path) -> dict[str, Any]:
        nonlocal stage_used
        stage_used = stage
        return _run_native_evaluation_impl(
            suite,
            policy_index=policy_index,
            populations=populations,
            store=store,
            out_dir=stage,
            workers=workers,
            record=record,
            provider_lineage=provider_lineage,
            identity_suite=identity_suite,
        )

    result = publish_directory(final, build, replace=True)
    if stage_used is None:  # pragma: no cover - publish_directory always invokes build
        raise RuntimeError("evaluation publication did not invoke its builder")
    old = str(stage_used)
    new = str(final)

    def published(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: published(item) for key, item in value.items()}
        if isinstance(value, list):
            return [published(item) for item in value]
        if isinstance(value, str) and value.startswith(old):
            return new + value[len(old) :]
        return value

    return published(result)


def _run_native_evaluation_impl(
    suite: dict[str, Any],
    *,
    policy_index: dict[str, Path],
    populations: dict[str, dict[str, Any]] | None = None,
    store: LocalStore | None = None,
    out_dir: Path | None = None,
    workers: int = 1,
    record: bool = True,
    provider_lineage: dict[str, Any] | None = None,
    identity_suite: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate, expand, run match jobs, and write an eval-run record.

    Independent cells may execute concurrently. ``executor.map`` and the final
    serialization retain deterministic cell order and seed mapping (EV-04).
    """
    workers = int(workers)
    if workers < 1:
        raise SchemaError("evaluation workers must be >= 1")
    populations = populations or {}
    suite = validate_evaluation(suite, populations=populations, policy_index=policy_index)
    interaction = suite.get("interaction", "parallel")
    cells, ledger = expand_evaluation_cells(suite, populations=populations)
    semantic_suite = identity_suite or suite
    suite_digest = evaluation_content_digest(semantic_suite)
    intent_digest = evaluation_intent_digest(semantic_suite)
    binding_digest = evaluation_binding_digest(
        suite,
        provider=str(suite.get("provider", "native")),
        workers=workers,
    )
    run_id = (
        "eval-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + f"-{sha256_canonical(suite_digest)[:8]}"
    )
    if out_dir is not None:
        run_root = Path(out_dir)
        run_root.mkdir(parents=True, exist_ok=True)
    elif store is not None:
        run_root = store.run_dir(run_id)
    else:
        run_root = Path.cwd() / "eval-runs" / run_id
        run_root.mkdir(parents=True, exist_ok=True)

    task_spec = dict(suite["task"])
    task_spec.setdefault("interaction", interaction)
    if interaction == "aec" and task_spec.get("env") in {
        "arena/competitive_rps_v0",
        "competitive_rps_v0",
        "arena/competitive_rps",
    }:
        task_spec["env"] = "arena/competitive_rps_aec_v0"
    task_info = describe_task(task_spec)
    identity_task = dict((identity_suite or suite)["task"])
    task_digest = task_content_digest(identity_task)
    provider_lineage = provider_lineage or {
        "kind": "native",
        "version": "arena-native-1",
        "config_digest": digest_uri(
            sha256_bytes(canonical_json((identity_suite or suite).get("provider_config") or {}))
        ),
    }
    if bool(task_info.get("dynamic_agents")) and interaction != "dynamic_aec":
        raise SchemaError(
            "Dynamic tasks require interaction=dynamic_aec plus explicit birth "
            "eligibility. Fixed-agent parallel/aec modes refuse lifecycle changes."
        )
    if interaction == "dynamic_aec" and not bool(task_info.get("dynamic_agents")):
        raise SchemaError(
            "interaction=dynamic_aec requires a task whose contract declares dynamic_agents=true"
        )
    def run_cell(cell: dict[str, Any]) -> dict[str, Any]:
        return _execute_evaluation_cell(
            cell=cell,
            suite=suite,
            task_spec=task_spec,
            task_info=task_info,
            task_digest=task_digest,
            provider_lineage=provider_lineage,
            policy_index=policy_index,
            run_root=run_root,
            record=record,
        )

    budgets = suite.get("budgets") or {}
    if not isinstance(budgets, dict):
        raise SchemaError("evaluation budgets must be a mapping")
    hard_timeout = budgets.get("timeout_seconds")
    executor_kind = budgets.get(
        "executor", "process" if hard_timeout is not None else "thread"
    )
    if executor_kind not in {"process", "thread"}:
        raise SchemaError("evaluation budgets.executor must be process|thread")
    if executor_kind == "process" and hard_timeout is None:
        raise SchemaError(
            "evaluation process executor requires budgets.timeout_seconds"
        )

    def run_cell_with_budget(cell: dict[str, Any]) -> dict[str, Any]:
        if executor_kind == "thread":
            return run_cell(cell)
        try:
            return _run_cell_supervised(
                cell=cell,
                suite=suite,
                task_spec=task_spec,
                task_info=task_info,
                task_digest=task_digest,
                provider_lineage=provider_lineage,
                policy_index=policy_index,
                run_root=run_root,
                record=record,
                timeout_seconds=float(hard_timeout),
                max_stdout_bytes=int(budgets.get("max_stdout_bytes", 1_048_576)),
                max_stderr_bytes=int(budgets.get("max_stderr_bytes", 1_048_576)),
            )
        except Exception as exc:  # noqa: BLE001 - every attempt is accounted.
            return _supervised_cell_failure(
                cell=cell,
                task_digest=task_digest,
                provider_lineage=provider_lineage,
                exc=exc,
                policy_index=policy_index,
            )

    if workers == 1 or len(cells) <= 1:
        cell_results = [run_cell_with_budget(cell) for cell in cells]
    else:
        with ThreadPoolExecutor(
            max_workers=min(workers, len(cells)),
            thread_name_prefix="arena-eval",
        ) as executor:
            cell_results = list(executor.map(run_cell_with_budget, cells))

    attempted = sum(len(cell["seeds"]) for cell in cell_results)
    completed = sum(
        int((cell.get("run") or {}).get("outcome", {}).get("episodes_completed", 0))
        for cell in cell_results
    )
    failed = sum(int(cell.get("failures", 0)) for cell in cell_results)
    state = (
        "complete"
        if completed == attempted and failed == 0
        else ("incomplete" if completed > 0 else "failed")
    )
    semantic_cells = [
        {
            "cell_id": cell["cell_id"],
            "assignments": cell["assignments"],
            "seeds": cell["seeds"],
            "episodes": [
                {
                    "seed": episode.get("seed"),
                    "returns": episode.get("returns") or {},
                    "outcomes": episode.get("outcomes") or {},
                }
                for episode in cell.get("episodes") or []
            ],
            "failures": [
                {
                    "seed": failure.get("seed"),
                    "kind": failure.get("kind"),
                    "episode_index": failure.get("episode_index"),
                }
                for failure in (cell.get("run") or {}).get("failures") or []
            ],
        }
        for cell in cell_results
    ]
    semantic_result_digest = digest_uri(
        sha256_bytes(
            canonical_json(
                {
                    "schema": "arena.evaluation-result/v1",
                    "evaluation_intent_digest": intent_digest,
                    "state": state,
                    "denominators": {
                        "attempted": attempted,
                        "completed": completed,
                        "failed": failed,
                    },
                    "cells": semantic_cells,
                }
            )
        )
    )
    eval_run = {
        "schema": EVAL_RUN_V1_SCHEMA,
        "run_id": run_id,
        "evaluation_digest": suite_digest,
        "evaluation_intent_digest": intent_digest,
        "execution_binding_digest": binding_digest,
        "semantic_result_digest": semantic_result_digest,
        "state": state,
        "denominators": {
            "attempted": attempted,
            "completed": completed,
            "failed": failed,
        },
        "evaluation_name": suite.get("name"),
        "interaction": interaction,
        "sampling_ledger": ledger,
        "cells": [
            {
                "cell_id": c["cell_id"],
                "assignments": c["assignments"],
                "seeds": c["seeds"],
                "sampling": c.get("sampling"),
                "failures": c["failures"],
                "evidence_refs": c["evidence_refs"],
                "candidate_policy": c.get("candidate_policy"),
                "opponent_policy": c.get("opponent_policy"),
                "lineage": c["lineage"],
            }
            for c in cell_results
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task_digest": task_digest,
        "provider": provider_lineage,
    }
    validate_eval_run_manifest(eval_run)
    dump_yaml(eval_run, run_root / "eval_run.yaml")
    dump_json(eval_run, run_root / "eval_run.json")
    dump_yaml(identity_suite or suite, run_root / "suite.yaml")
    dump_json(evaluation_intent_projection(semantic_suite), run_root / "intent.json")
    # Attach rich cell results for metrics (not all duplicated into schema-minimal yaml).
    eval_run["cell_results"] = cell_results
    eval_run["suite"] = suite
    eval_run["run_dir"] = str(run_root)
    if store is not None:
        obj = store.put_bytes(canonical_json({k: eval_run[k] for k in eval_run if k != "cell_results"}))
        store.set_ref(f"evals/runs/{run_id}", obj)
        eval_run["object_digest"] = obj
    return eval_run


def _episode_returns(ep: dict[str, Any]) -> dict[str, float]:
    returns: dict[str, float] = {}
    for step in ep.get("steps") or []:
        rewards = step.get("rewards") or {}
        for agent, r in rewards.items():
            returns[agent] = returns.get(agent, 0.0) + float(r)
    return returns


def _collect_policy_digests(cells: list[dict[str, Any]]) -> list[str]:
    """Bind every cell assignment digest into the report claim surface."""
    digests: set[str] = set()
    for cell in cells:
        assignments = cell.get("assignments") or {}
        if isinstance(assignments, dict):
            for value in assignments.values():
                if isinstance(value, str) and value.startswith("sha256:"):
                    digests.add(value)
        lineage = cell.get("lineage") or {}
        for value in lineage.get("policy_digests") or []:
            if isinstance(value, str) and value.startswith("sha256:"):
                digests.add(value)
    return sorted(digests)


def build_eval_report(eval_run: dict[str, Any]) -> dict[str, Any]:
    state = eval_run.get("state", "complete")
    suite = eval_run.get("suite") or {}
    failure_policy = suite.get("failure_policy") or {}
    allow_missing = failure_policy.get("missingness", "fail") == "allow"
    denominators = eval_run.get("denominators") or {}
    failed = int(denominators.get("failed", 0))
    attempted = int(denominators.get("attempted", 0))
    completed = int(denominators.get("completed", 0))
    max_failed = int(failure_policy.get("max_failed_episodes", 0))
    if state != "complete" and (not allow_missing or failed > max_failed):
        raise IncompleteExecutionError(
            "refusing to report an incomplete evaluation: "
            f"state={state}, failed={failed}; set failure_policy.missingness=allow "
            "with an explicit max_failed_episodes threshold to opt in",
            code="EVALUATION_INCOMPLETE",
            cause=str(state),
            repair=(
                "Inspect eval_run denominators and failure ledgers, repair the "
                "recorded failures, or set failure_policy.missingness=allow with "
                "max_failed_episodes covering the failed count."
            ),
            context={
                "state": state,
                "attempted": attempted,
                "completed": completed,
                "failed": failed,
                "missingness": failure_policy.get("missingness", "fail"),
                "max_failed_episodes": max_failed,
            },
        )
    metrics_plugins.register_builtins()
    metric_kinds = suite.get("metrics") or ["payoff_matrix", "mean_return"]
    cells = eval_run.get("cell_results") or []
    # Enrich cells from minimal eval_run if needed.
    if not cells:
        cells = list(eval_run.get("cells") or [])
    policy_digests = _collect_policy_digests(cells)
    if not policy_digests:
        raise SchemaError(
            "refusing to report an evaluation with no bound policy digests; "
            "cells must record sha256 assignments so claims bind policy+suite identity"
        )
    for required in (
        "evaluation_digest",
        "evaluation_intent_digest",
        "semantic_result_digest",
    ):
        if not eval_run.get(required):
            raise SchemaError(
                f"refusing to report evaluation missing suite identity field: {required}"
            )
    computed = {}
    for kind in metric_kinds:
        name = kind if isinstance(kind, str) else kind.get("kind")
        metric = metrics_plugins.METRICS.get(str(name))
        computed[str(name)] = metric.compute(cells)
    ledger = list(eval_run.get("sampling_ledger") or [])
    sampling_ledger_digest = digest_uri(sha256_bytes(canonical_json(ledger)))
    population_digests: list[str] = []
    seen_pops: set[str] = set()
    for spec in (suite.get("assignments") or {}).values():
        if isinstance(spec, dict) and spec.get("kind") in {"population", "crossplay"}:
            pref = str(spec.get("population") or "")
            if pref.startswith("sha256:") and pref not in seen_pops:
                seen_pops.add(pref)
                population_digests.append(pref)
    report = {
        "schema": "arena.eval-report/v1",
        "evaluation_digest": eval_run["evaluation_digest"],
        "evaluation_intent_digest": eval_run["evaluation_intent_digest"],
        "execution_binding_digest": eval_run.get("execution_binding_digest"),
        "semantic_result_digest": eval_run["semantic_result_digest"],
        "policy_digests": policy_digests,
        "evaluation_name": eval_run.get("evaluation_name") or suite.get("name"),
        "state": state,
        "denominators": eval_run.get("denominators"),
        "eval_run_digest": eval_run.get("object_digest")
        or digest_uri(sha256_bytes(canonical_json(eval_run.get("cells")))),
        "sampling_ledger": ledger,
        "sampling_ledger_digest": sampling_ledger_digest,
        "population_digests": population_digests,
        "metrics": computed,
        "provider": eval_run.get("provider") or {"kind": "native"},
        "task_digest": eval_run.get("task_digest"),
        "nontransitivity_warning": (computed.get("payoff_matrix") or {}).get(
            "nontransitivity_warning"
        ),
    }
    validate_eval_report_manifest(report)
    return report
