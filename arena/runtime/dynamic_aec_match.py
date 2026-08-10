"""Lifecycle-aware AEC match runner for explicitly qualified dynamic tasks."""

from __future__ import annotations

import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from arena.adapters.policy_custom_torch import load_runtime
from arena.adapters.task_pettingzoo.adapter import (
    describe_task,
    extract_action_mask,
    extract_observation,
    make_env,
)
from arena.core.action_cases import validate_runtime_action
from arena.core.compatibility import compose_check
from arena.core.errors import CompatibilityError, ConformanceError, RuntimeFailure, SchemaError
from arena.core.identity import sha256_canonical
from arena.core.manifests import RUN_SCHEMA, TRAJECTORY_SCHEMA, dump_json, dump_yaml
from arena.core.sdk import Policy
from arena.runtime.seed_protocol import policy_rng
from arena.runtime.trajectory import TrajectoryWriter


def _check_agent(
    agent: str,
    policy: Policy,
    *,
    policy_role: str | None = None,
    task_info: dict[str, Any],
    action_mode: str,
) -> None:
    meta = task_info["roles"].get(agent)
    if meta is None:
        raise CompatibilityError(
            f"dynamic agent {agent!r} has no declared role contract; "
            "births must be described before execution"
        )
    report = compose_check(
        policy=policy.manifest,
        role=policy_role or agent,
        expected_obs=meta.get("observation"),
        expected_act=meta.get("action"),
        action_mode=action_mode,
        task_provides_masks=bool(task_info.get("provides_masks")),
    )
    report.raise_for_errors()


def run_dynamic_aec_match(
    *,
    task_spec: dict[str, Any],
    assignments: dict[str, Policy],
    seeds: list[int],
    action_mode: str = "deterministic",
    record: bool = True,
    out_dir: Path | None = None,
    failure_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task_spec = {**task_spec, "interaction": "dynamic_aec"}
    task_info = describe_task(task_spec)
    if not bool(task_info.get("dynamic_agents")):
        raise SchemaError("dynamic_aec task must declare a changing agents lifecycle")

    from arena.plugins.lifecycle import resolve_lifecycle_plan

    lifecycle_plan = resolve_lifecycle_plan(task_spec, task_info, assignments)
    for agent, binding in lifecycle_plan.bindings.items():
        _check_agent(
            agent,
            binding.policy,
            policy_role=binding.policy_role,
            task_info=task_info,
            action_mode=action_mode,
        )

    policy_runtimes = {
        agent: load_runtime(binding.policy.root)
        for agent, binding in lifecycle_plan.bindings.items()
    }
    failure_policy = failure_policy or {
        "timeout_seconds": 60,
        "retain_incomplete": True,
        "retry": 0,
    }
    timeout = float(failure_policy.get("timeout_seconds", 60))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + sha256_canonical(
        {"seeds": seeds, "assignments": {k: v.digest for k, v in sorted(assignments.items())}}
    )[:8]
    out_dir = Path(out_dir or (Path.cwd() / "runs" / run_id))
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = TrajectoryWriter(out_dir / "trajectories") if record else None
    failures: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []

    for episode_index, seed in enumerate(seeds):
        started = time.monotonic()
        try:
            summary = _run_episode(
                task_spec=task_spec,
                task_info=task_info,
                lifecycle_plan=lifecycle_plan,
                policy_runtimes=policy_runtimes,
                seed=int(seed),
                episode_index=episode_index,
                action_mode=action_mode,
                timeout=timeout,
                started=started,
                writer=writer,
            )
            episodes.append(summary)
        except RuntimeFailure as exc:
            failures.append(
                {
                    "episode_index": episode_index,
                    "seed": seed,
                    "kind": exc.kind,
                    "message": str(exc),
                    "agent": exc.agent,
                    "details": exc.details,
                }
            )
            if failure_policy.get("retain_incomplete", True) and writer is not None:
                partial = exc.details.get("partial_episode")
                if partial:
                    writer.write_episode(partial)
            episodes.append(
                {
                    "episode_index": episode_index,
                    "seed": seed,
                    "status": exc.kind,
                    "steps": exc.details.get("steps", 0),
                }
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "episode_index": episode_index,
                    "seed": seed,
                    "kind": "crash",
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            episodes.append(
                {"episode_index": episode_index, "seed": seed, "status": "crash", "steps": 0}
            )

    if writer is not None:
        writer.finalize(
            task_info=task_info,
            assignments={
                key: policy.digest for key, policy in assignments.items()
            },
            seeds=seeds,
            action_mode=action_mode,
            failures=failures,
        )
    run = {
        "schema": RUN_SCHEMA,
        "run_id": run_id,
        "interaction": "dynamic_aec",
        "task": {
            "adapter": task_info["adapter"],
            "env": task_info["env"],
            "version": task_info["version"],
            "spec": _public(task_spec),
        },
        "assignments": {
            agent: {"name": policy.name, "digest": policy.digest, "path": str(policy.root)}
            for agent, policy in assignments.items()
        },
        "lifecycle_resolver": lifecycle_plan.kind,
        "resolved_agents": {
            agent: {
                "assignment_key": binding.assignment_key,
                "policy_role": binding.policy_role,
                "policy_digest": binding.policy.digest,
            }
            for agent, binding in lifecycle_plan.bindings.items()
        },
        "seeds": seeds,
        "action_mode": action_mode,
        "episodes": episodes,
        "failures": failures,
        "outcome": {
            "episodes_requested": len(seeds),
            "episodes_completed": sum(1 for item in episodes if item.get("status") == "completed"),
            "failure_count": len(failures),
        },
        "failure_policy": failure_policy,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    dump_yaml(run, out_dir / "run.yaml")
    dump_json(run, out_dir / "run.json")
    return run


def _run_episode(
    *,
    task_spec: dict[str, Any],
    task_info: dict[str, Any],
    lifecycle_plan: Any,
    policy_runtimes: dict[str, Any],
    seed: int,
    episode_index: int,
    action_mode: str,
    timeout: float,
    started: float,
    writer: TrajectoryWriter | None,
) -> dict[str, Any]:
    env = make_env(task_spec)
    try:
        env.reset(seed=seed)
        initial = list(env.agents)
        activated = set(initial)
        for agent in initial:
            policy_runtimes[agent].reset(agent)
        possible = list(
            task_info.get("possible_agents") or lifecycle_plan.bindings
        )
        returns = {agent: 0.0 for agent in possible}
        segments = {
            agent: {
                "joined_step": 0,
                "left_step": None,
                "origin": "reset",
                "segment_index": 0,
                "assignment_key": lifecycle_plan.binding_for(agent).assignment_key,
                "policy_digest": lifecycle_plan.binding_for(agent).policy.digest,
            }
            for agent in initial
        }
        segment_history = {agent: [dict(segment)] for agent, segment in segments.items()}
        steps: list[dict[str, Any]] = []

        while env.agents:
            if time.monotonic() - started > timeout:
                raise RuntimeFailure(
                    "episode timeout",
                    kind="timeout",
                    episode_index=episode_index,
                    details={
                        "steps": len(steps),
                        "partial_episode": _episode_record(
                            episode_index=episode_index,
                            seed=seed,
                            steps=steps,
                            returns=returns,
                            segments=segments,
                            segment_history=segment_history,
                            task_info=task_info,
                            lifecycle_plan=lifecycle_plan,
                            action_mode=action_mode,
                            status="timeout",
                        ),
                    },
                )
            agent = env.agent_selection
            if agent is None or agent not in lifecycle_plan.bindings:
                raise RuntimeFailure(
                    f"missing declared policy for selected dynamic agent {agent!r}",
                    kind="lifecycle_error",
                    episode_index=episode_index,
                    agent=agent,
                    details={"steps": len(steps)},
                )
            if agent not in activated:
                _check_agent(
                    agent,
                    lifecycle_plan.binding_for(agent).policy,
                    policy_role=lifecycle_plan.binding_for(agent).policy_role,
                    task_info=task_info,
                    action_mode=action_mode,
                )
                policy_runtimes[agent].reset(agent)
                activated.add(agent)

            before = list(env.agents)
            raw_obs = env.observe(agent)
            obs = extract_observation(raw_obs)
            mask = extract_action_mask(raw_obs)
            info = (getattr(env, "infos", {}) or {}).get(agent) or {}
            if mask is None:
                mask = extract_action_mask(info)
            try:
                action = policy_runtimes[agent].act(
                    obs,
                    mode=action_mode,
                    action_mask=mask,
                    rng=policy_rng(seed, agent, len(steps)),
                    agent_id=agent,
                )
                validate_runtime_action(
                    action,
                    action=task_info["roles"][agent]["action"],
                    agent=agent,
                )
            except (ConformanceError, SchemaError) as exc:
                raise RuntimeFailure(
                    str(exc),
                    kind="policy_failure",
                    episode_index=episode_index,
                    agent=agent,
                    details={"steps": len(steps)},
                ) from exc

            env.step(action)
            after = list(env.agents)
            joined = [item for item in after if item not in before]
            left = [item for item in before if item not in after]
            join_details: list[dict[str, Any]] = []
            step_index = len(steps)
            for born in joined:
                try:
                    binding = lifecycle_plan.verify_join(born)
                    _check_agent(
                        born,
                        binding.policy,
                        policy_role=binding.policy_role,
                        task_info=task_info,
                        action_mode=action_mode,
                    )
                except (CompatibilityError, SchemaError) as exc:
                    raise RuntimeFailure(
                        str(exc),
                        kind="lifecycle_error",
                        episode_index=episode_index,
                        agent=born,
                        details={"steps": step_index},
                    ) from exc
                policy_runtimes[born].reset(born)
                activated.add(born)
                segment = {
                    "joined_step": step_index,
                    "left_step": None,
                    "origin": "rejoin" if born in segment_history else "birth",
                    "segment_index": len(segment_history.get(born, [])),
                    "assignment_key": binding.assignment_key,
                    "policy_digest": binding.policy.digest,
                }
                segments[born] = segment
                segment_history.setdefault(born, []).append(segment)
                join_details.append(
                    {
                        "agent": born,
                        "origin": segment["origin"],
                        "segment_index": segment["segment_index"],
                        "assignment_key": binding.assignment_key,
                        "policy_role": binding.policy_role,
                        "policy_digest": binding.policy.digest,
                    }
                )
            for removed in left:
                segments.setdefault(
                    removed,
                    {"joined_step": 0, "left_step": None, "origin": "reset"},
                )["left_step"] = step_index
                if segment_history.get(removed):
                    segment_history[removed][-1]["left_step"] = step_index
                policy_runtimes[removed].reset_agent(removed)

            rewards = {
                item: float((getattr(env, "rewards", {}) or {}).get(item, 0.0))
                for item in possible
            }
            for item, reward in rewards.items():
                returns[item] += reward
            steps.append(
                {
                    "agent_selection": agent,
                    "agents_alive_before": before,
                    "agents_alive": after,
                    "join_events": joined,
                    "join_event_details": join_details,
                    "leave_events": left,
                    "observations": {agent: _jsonable(obs)},
                    "actions": {agent: _jsonable(action)},
                    "rewards": rewards,
                    "terminations": {
                        item: bool((getattr(env, "terminations", {}) or {}).get(item, False))
                        for item in possible
                    },
                    "truncations": {
                        item: bool((getattr(env, "truncations", {}) or {}).get(item, False))
                        for item in possible
                    },
                    "infos": {agent: _jsonable(info)},
                }
            )

        episode = _episode_record(
            episode_index=episode_index,
            seed=seed,
            steps=steps,
            returns=returns,
            segments=segments,
            segment_history=segment_history,
            task_info=task_info,
            lifecycle_plan=lifecycle_plan,
            action_mode=action_mode,
            status="completed",
        )
        if writer is not None:
            writer.write_episode(episode)
        return {
            "episode_index": episode_index,
            "seed": seed,
            "status": "completed",
            "steps": len(steps),
            "returns": returns,
            "lifecycle_events": sum(
                len(step["join_events"]) + len(step["leave_events"]) for step in steps
            ),
        }
    finally:
        env.close()


def _episode_record(**kwargs: Any) -> dict[str, Any]:
    task_info = kwargs["task_info"]
    lifecycle_plan = kwargs["lifecycle_plan"]
    return {
        "schema": TRAJECTORY_SCHEMA,
        "episode_index": kwargs["episode_index"],
        "seed": kwargs["seed"],
        "steps": kwargs["steps"],
        "returns": kwargs["returns"],
        "status": kwargs["status"],
        "interaction": "dynamic_aec",
        "action_mode": kwargs["action_mode"],
        "task": {
            "env": task_info["env"],
            "adapter": task_info["adapter"],
            "version": task_info["version"],
        },
        "agents": task_info.get("possible_agents") or list(lifecycle_plan.bindings),
        "initial_agents": task_info.get("agents") or [],
        "agent_segments": kwargs["segments"],
        "agent_segment_history": kwargs["segment_history"],
        "role_map": {
            agent: binding.policy_role
            for agent, binding in lifecycle_plan.bindings.items()
        },
        "policies": {
            agent: binding.policy.digest
            for agent, binding in lifecycle_plan.bindings.items()
        },
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _public(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _public(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, (list, tuple)):
        return [_public(item) for item in value]
    return value
