"""One-shot cross-play matrix: policies → population → suite → report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from arena.core.errors import SchemaError
from arena.core.manifests import dump_yaml
from arena.core.population import create_population, write_population_yaml
from arena.core.sdk import Policy
from arena.core.store import LocalStore
from arena.runtime.evaluation import build_eval_report, run_evaluation, validate_evaluation


def _default_task(
    *,
    env: str,
    adapter: str = "pettingzoo-parallel",
    interaction: str = "parallel",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task: dict[str, Any] = {
        "adapter": adapter,
        "env": env,
        "interaction": interaction,
    }
    if config:
        task["config"] = dict(config)
    return task


def synthesize_crossplay_artifacts(
    policy_paths: list[Path | str],
    *,
    task: dict[str, Any],
    store: LocalStore,
    name: str = "crossplay-matrix",
    population_ref: str | None = None,
    roles: tuple[str, str] = ("player_0", "player_1"),
    seeds: dict[str, Any] | None = None,
    action_mode: str = "deterministic",
    metrics: list[str] | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Build a population + cartesian cross-play suite from bare policy bundles.

    Returns paths/digests plus in-memory manifests ready for ``run_evaluation``.
    """
    paths = [Path(p).resolve() for p in policy_paths]
    if len(paths) < 2:
        raise SchemaError(
            "cross-play matrix requires at least two policy bundles; "
            "pass --policy twice (or more)."
        )
    for path in paths:
        if not path.exists():
            raise SchemaError(f"policy bundle not found: {path}")

    members = [{"policy": str(path), "weight": 1.0} for path in paths]
    pop = create_population(
        name=f"{name}-population",
        members=members,
        store=store,
        ref=population_ref or f"populations/{name}",
    )
    role_0, role_1 = roles
    suite: dict[str, Any] = {
        "schema": "arena.evaluation/v0alpha1",
        "name": name,
        "interaction": task.get("interaction", "parallel"),
        "task": dict(task),
        "assignments": {
            role_0: {"kind": "crossplay", "population": pop["digest"]},
            role_1: {"kind": "crossplay", "population": pop["digest"]},
        },
        "seeds": seeds or {"start": 0, "count": 1},
        "action_mode": action_mode,
        "metrics": metrics or ["payoff_matrix", "mean_return", "win_rate"],
        "sampling": {"kind": "enumerated_crossplay", "seed": 0},
    }
    policy_index: dict[str, Path] = {}
    for path in paths:
        digest = Policy.load(path).digest
        policy_index[digest] = path
        policy_index[path.name] = path
    populations = {pop["digest"]: pop}

    validate_evaluation(suite, populations=populations, policy_index=policy_index)

    artifacts: dict[str, Any] = {
        "population": pop,
        "suite": suite,
        "policy_index": policy_index,
        "populations": populations,
    }
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pop_path = out_dir / "population.yaml"
        suite_path = out_dir / "evaluation.yaml"
        write_population_yaml(pop, pop_path)
        dump_yaml(suite, suite_path)
        artifacts["population_path"] = pop_path
        artifacts["suite_path"] = suite_path
    return artifacts


def run_crossplay_matrix(
    policy_paths: list[Path | str],
    *,
    out_dir: Path | str,
    env: str | None = None,
    task: dict[str, Any] | None = None,
    adapter: str = "pettingzoo-parallel",
    interaction: str = "parallel",
    config: dict[str, Any] | None = None,
    store: LocalStore | None = None,
    name: str = "crossplay-matrix",
    population_ref: str | None = None,
    roles: tuple[str, str] = ("player_0", "player_1"),
    seeds: dict[str, Any] | None = None,
    workers: int = 1,
    provider: str | None = None,
) -> dict[str, Any]:
    """Policies → population → cross-play matrix → non-transitivity-aware report.

    Writes ``eval_run.json``, ``report.json``, and resolved YAML under ``out_dir``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if store is None:
        try:
            store = LocalStore.find()
        except Exception:
            store = LocalStore(out)
            if not (store.arena / "workspace.toml").exists():
                store.init()

    if task is None:
        if not env:
            raise SchemaError("run_crossplay_matrix requires --env or an explicit task mapping")
        task = _default_task(
            env=env, adapter=adapter, interaction=interaction, config=config
        )
    elif env is not None and "env" not in task:
        task = {**task, "env": env}

    # Do not write YAML into ``out`` before ``run_evaluation``: native evaluation
    # publishes via ``publish_directory(..., replace=True)``, which replaces the
    # entire destination tree and would drop pre-written manifests.
    synthesized = synthesize_crossplay_artifacts(
        policy_paths,
        task=task,
        store=store,
        name=name,
        population_ref=population_ref,
        roles=roles,
        seeds=seeds,
        out_dir=None,
    )
    eval_run = run_evaluation(
        synthesized["suite"],
        policy_index=synthesized["policy_index"],
        populations=synthesized["populations"],
        store=store,
        out_dir=out,
        workers=workers,
        provider=provider,
    )
    report = build_eval_report(eval_run)
    from arena.core.manifests import dump_json, dump_yaml

    dump_json(report, out / "report.json")
    dump_yaml(report, out / "report.yaml")
    write_population_yaml(synthesized["population"], out / "population.yaml")
    dump_yaml(synthesized["suite"], out / "evaluation.yaml")
    return {
        "population": synthesized["population"],
        "population_digest": synthesized["population"]["digest"],
        "suite": synthesized["suite"],
        "run_dir": eval_run["run_dir"],
        "run_id": eval_run["run_id"],
        "cells": len(eval_run["cells"]),
        "sampling_ledger": eval_run["sampling_ledger"],
        "sampling_ledger_digest": report.get("sampling_ledger_digest"),
        "evaluation_digest": eval_run["evaluation_digest"],
        "evaluation_intent_digest": eval_run.get("evaluation_intent_digest"),
        "execution_binding_digest": eval_run.get("execution_binding_digest"),
        "semantic_result_digest": eval_run.get("semantic_result_digest"),
        "eval_run_digest": report.get("eval_run_digest"),
        "state": eval_run.get("state"),
        "nontransitivity_warning": report.get("nontransitivity_warning"),
        "report": report,
    }
