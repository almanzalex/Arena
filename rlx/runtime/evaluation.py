"""Versioned evaluation suites: expand to match jobs and record sampling ledgers."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rlx.adapters.task_pettingzoo.adapter import describe_task
from rlx.core.compatibility import compose_check
from rlx.core.errors import CompatibilityError, SchemaError
from rlx.core.identity import canonical_json, digest_uri, sha256_bytes, sha256_canonical
from rlx.core.manifests import (
    EVAL_RUN_SCHEMA,
    dump_json,
    dump_yaml,
    evaluation_content_digest,
    expand_seeds,
    load_manifest,
    task_content_digest,
    validate_eval_run_manifest,
    validate_evaluation_manifest,
)
from rlx.core.population import assert_members_compatible_with_role, load_population
from rlx.core.sdk import Policy, Task
from rlx.core.store import LocalStore
from rlx.plugins import metrics as metrics_plugins
from rlx.plugins import samplers as sampler_plugins


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
            "RLX supports at most two crossplay/population assignment roles "
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
    from rlx.core.registry import EVAL_PROVIDERS, ensure_plugins_loaded

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
    """Validate, expand, run match jobs, and write an eval-run record.

    Independent cells may execute concurrently. ``executor.map`` and the final
    serialization retain deterministic cell order and seed mapping (EV-04).
    """
    workers = int(workers)
    if workers < 1:
        raise SchemaError("evaluation workers must be >= 1")
    populations = populations or {}
    suite = validate_evaluation(suite, populations=populations, policy_index=policy_index)
    from rlx.plugins.interactions import get_interaction

    interaction = suite.get("interaction", "parallel")
    _run = get_interaction(interaction).run_match
    cells, ledger = expand_evaluation_cells(suite, populations=populations)
    suite_digest = evaluation_content_digest(identity_suite or suite)
    run_id = f"eval-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{sha256_canonical(suite_digest)[:8]}"
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
        "rlx/competitive_rps_v0",
        "competitive_rps_v0",
        "rlx/competitive_rps",
    }:
        task_spec["env"] = "rlx/competitive_rps_aec_v0"
    task_info = describe_task(task_spec)
    identity_task = dict((identity_suite or suite)["task"])
    task_digest = task_content_digest(identity_task)
    provider_lineage = provider_lineage or {
        "kind": "native",
        "version": "rlx-native-0.5",
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
        match_out = run_root / cell["cell_id"]
        result = _run(
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
                ep = json.loads(ep_path.read_text(encoding="utf-8"))
                episodes.append(
                    {
                        "path": str(ep_path),
                        "seed": ep.get("seed"),
                        "returns": ep.get("returns") or _episode_returns(ep),
                        "outcomes": ep.get("outcomes") or {},
                    }
                )
                evidence_refs.append(str(ep_path))
            bundle = traj_dir / "bundle.json"
            if bundle.exists():
                evidence_refs.insert(0, str(bundle))
        return {
            **cell,
            "run": result,
            "episodes": episodes,
            "evidence_refs": evidence_refs,
            "failures": len(result.get("failures") or []),
            "lineage": {
                "policy_digests": sorted(set(cell["assignments"].values())),
                "task_digest": task_digest,
                "provider": provider_lineage,
            },
        }

    if workers == 1 or len(cells) <= 1:
        cell_results = [run_cell(cell) for cell in cells]
    else:
        with ThreadPoolExecutor(
            max_workers=min(workers, len(cells)),
            thread_name_prefix="rlx-eval",
        ) as executor:
            cell_results = list(executor.map(run_cell, cells))

    eval_run = {
        "schema": EVAL_RUN_SCHEMA,
        "run_id": run_id,
        "evaluation_digest": suite_digest,
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


def build_eval_report(eval_run: dict[str, Any]) -> dict[str, Any]:
    metrics_plugins.register_builtins()
    suite = eval_run.get("suite") or {}
    metric_kinds = suite.get("metrics") or ["payoff_matrix", "mean_return"]
    cells = eval_run.get("cell_results") or []
    # Enrich cells from minimal eval_run if needed.
    if not cells:
        cells = list(eval_run.get("cells") or [])
    computed = {}
    for kind in metric_kinds:
        name = kind if isinstance(kind, str) else kind.get("kind")
        metric = metrics_plugins.METRICS.get(str(name))
        computed[str(name)] = metric.compute(cells)
    report = {
        "schema": "rlx.eval-report/v0alpha1",
        "evaluation_digest": eval_run["evaluation_digest"],
        "eval_run_digest": eval_run.get("object_digest")
        or digest_uri(sha256_bytes(canonical_json(eval_run.get("cells")))),
        "metrics": computed,
        "provider": eval_run.get("provider") or {"kind": "native"},
        "task_digest": eval_run.get("task_digest"),
        "nontransitivity_warning": (computed.get("payoff_matrix") or {}).get(
            "nontransitivity_warning"
        ),
    }
    return report
