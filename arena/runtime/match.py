"""Match execution and trajectory recording."""

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
from arena.core.errors import (
    CompatibilityError,
    ConformanceError,
    RuntimeFailure,
    SchemaError,
    TaskRuntimeError,
)
from arena.core.identity import sha256_canonical
from arena.runtime.seed_protocol import policy_rng
from arena.core.manifests import RUN_SCHEMA, TRAJECTORY_SCHEMA, dump_json, dump_yaml
from arena.core.sdk import Policy
from arena.runtime.trajectory import TrajectoryWriter


def _validate_action(
    action: Any,
    *,
    agent: str,
    task_info: dict[str, Any],
    episode_index: int,
    step_i: int,
) -> None:
    """Reject out-of-bounds / wrong-type actions before they reach the env.

    Switches on the declared action-type case (Discrete / MultiDiscrete / Box / Dict).
    Never silently skips non-Discrete spaces or coerces scalar↔vector↔Dict.
    """
    act_space = task_info.get("roles", {}).get(agent, {}).get("action", {})
    try:
        validate_runtime_action(action, action=act_space, agent=agent)
    except (ConformanceError, SchemaError) as e:
        raise RuntimeFailure(
            str(e),
            kind="invalid_action",
            episode_index=episode_index,
            agent=agent,
            details={"steps": step_i, "action": _jsonable(action), "action_space": act_space},
        ) from e



def run_match(
    *,
    task_spec: dict[str, Any],
    assignments: dict[str, Policy],
    seeds: list[int],
    action_mode: str = "deterministic",
    record: bool = True,
    out_dir: Path | None = None,
    failure_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    failure_policy = failure_policy or {
        "timeout_seconds": 60,
        "retain_incomplete": True,
        "retry": 0,
    }
    timeout = float(failure_policy.get("timeout_seconds", 60))
    retain_incomplete = bool(failure_policy.get("retain_incomplete", True))

    task_info = describe_task(task_spec)
    roles_meta = task_info["roles"]

    # Pre-run compatibility: assignment keys are agent ids; policy must allow that agent as role.
    for agent_or_role, policy in assignments.items():
        meta = roles_meta.get(agent_or_role)
        if meta is None:
            raise CompatibilityError(
                f"assignment key {agent_or_role!r} is not an agent in task; "
                f"known agents: {list(roles_meta)}"
            )
        report = compose_check(
            policy=policy.manifest,
            role=agent_or_role,
            expected_obs=meta.get("observation"),
            expected_act=meta.get("action"),
            action_mode=action_mode,
            task_provides_masks=bool(task_info.get("provides_masks")),
        )
        report.raise_for_errors()

    # Load runtimes
    runtimes = {k: load_runtime(v.root) for k, v in assignments.items()}

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + sha256_canonical(
        {
            "seeds": seeds,
            "assignments": {k: v.digest for k, v in sorted(assignments.items())},
        }
    )[:8]
    if out_dir is None:
        out_dir = Path.cwd() / "runs" / run_id
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    writer = TrajectoryWriter(out_dir / "trajectories") if record else None
    failures: list[dict[str, Any]] = []
    episode_summaries: list[dict[str, Any]] = []

    for ep_idx, seed in enumerate(seeds):
        ep_start = time.monotonic()
        status = "completed"
        try:
            summary = _run_episode(
                task_spec=task_spec,
                task_info=task_info,
                runtimes=runtimes,
                assignments=assignments,
                seed=int(seed),
                episode_index=ep_idx,
                action_mode=action_mode,
                writer=writer,
                timeout=timeout,
                ep_start=ep_start,
            )
            episode_summaries.append(summary)
        except RuntimeFailure as e:
            status = e.kind
            fail = {
                "episode_index": ep_idx,
                "seed": seed,
                "kind": e.kind,
                "message": str(e),
                "agent": e.agent,
                "details": e.details,
            }
            failures.append(fail)
            if retain_incomplete and writer is not None and e.details.get("partial_episode"):
                writer.write_episode(e.details["partial_episode"])
            episode_summaries.append(
                {
                    "episode_index": ep_idx,
                    "seed": seed,
                    "status": status,
                    "steps": e.details.get("steps", 0),
                }
            )
        except Exception as e:  # noqa: BLE001 — must account for all failures
            failures.append(
                {
                    "episode_index": ep_idx,
                    "seed": seed,
                    "kind": "crash",
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                }
            )
            episode_summaries.append(
                {"episode_index": ep_idx, "seed": seed, "status": "crash", "steps": 0}
            )

    if writer is not None:
        writer.finalize(
            task_info=task_info,
            assignments={k: v.digest for k, v in assignments.items()},
            seeds=seeds,
            action_mode=action_mode,
            failures=failures,
        )

    run_record = {
        "schema": RUN_SCHEMA,
        "run_id": run_id,
        "task": {
            "adapter": task_info["adapter"],
            "env": task_info["env"],
            "version": task_info["version"],
            "spec": _public_spec(task_spec),
        },
        "assignments": {
            k: {"name": v.name, "digest": v.digest, "path": str(v.root)} for k, v in assignments.items()
        },
        "seeds": seeds,
        "action_mode": action_mode,
        "episodes": episode_summaries,
        "failures": failures,
        "outcome": {
            "episodes_requested": len(seeds),
            "episodes_completed": sum(1 for e in episode_summaries if e.get("status") == "completed"),
            "failure_count": len(failures),
        },
        "failure_policy": failure_policy,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    dump_yaml(run_record, out_dir / "run.yaml")
    dump_json(run_record, out_dir / "run.json")
    return run_record


def _run_episode(
    *,
    task_spec: dict[str, Any],
    task_info: dict[str, Any],
    runtimes: dict[str, Any],
    assignments: dict[str, Policy],
    seed: int,
    episode_index: int,
    action_mode: str,
    writer: TrajectoryWriter | None,
    timeout: float,
    ep_start: float,
) -> dict[str, Any]:
    # Trust for entrypoint_bundle lives on the task spec so patched make_env(spec)
    # call sites (tests / adapters) stay single-argument compatible.
    try:
        env = make_env(task_spec)
    except TaskRuntimeError as e:
        raise RuntimeFailure(
            str(e),
            kind=e.kind,
            episode_index=episode_index,
            details={**e.details, "steps": 0},
        ) from e
    try:
        try:
            obs, infos = env.reset(seed=seed)
        except TaskRuntimeError as e:
            raise RuntimeFailure(
                str(e),
                kind=e.kind,
                episode_index=episode_index,
                details={**e.details, "steps": 0},
            ) from e
        for agent, rt in runtimes.items():
            rt.reset(agent)

        steps: list[dict[str, Any]] = []
        step_i = 0
        returns: dict[str, float] = {a: 0.0 for a in task_info["agents"]}

        while env.agents:
            if time.monotonic() - ep_start > timeout:
                partial = _episode_dict(
                    seed, episode_index, task_info, assignments, steps, action_mode, status="timeout"
                )
                raise RuntimeFailure(
                    f"episode {episode_index} exceeded timeout {timeout}s",
                    kind="timeout",
                    episode_index=episode_index,
                    details={"steps": step_i, "partial_episode": partial},
                )

            actions: dict[str, Any] = {}
            masks: dict[str, Any] = {}
            for agent in list(env.agents):
                if agent not in runtimes:
                    raise RuntimeFailure(
                        f"no policy assigned for agent {agent}",
                        kind="invalid_action",
                        episode_index=episode_index,
                        agent=agent,
                        details={"steps": step_i},
                    )
                raw = obs[agent]
                o = extract_observation(raw)
                mask = extract_action_mask(raw)
                if mask is not None:
                    masks[agent] = np.asarray(mask).tolist()
                rng = policy_rng(seed, agent, step_i)
                try:
                    action = runtimes[agent].act(
                        o,
                        mode=action_mode,
                        action_mask=np.asarray(mask) if mask is not None else None,
                        rng=rng,
                        agent_id=agent,
                    )
                except Exception as e:  # noqa: BLE001
                    raise RuntimeFailure(
                        f"policy failure for {agent}: {e}",
                        kind="policy_failure",
                        episode_index=episode_index,
                        agent=agent,
                        details={"steps": step_i},
                    ) from e
                _validate_action(
                    action,
                    agent=agent,
                    task_info=task_info,
                    episode_index=episode_index,
                    step_i=step_i,
                )
                actions[agent] = action

            try:
                next_obs, rewards, terminations, truncations, next_infos = env.step(actions)
            except TaskRuntimeError as e:
                partial = _episode_dict(
                    seed,
                    episode_index,
                    task_info,
                    assignments,
                    steps,
                    action_mode,
                    status=e.kind,
                )
                raise RuntimeFailure(
                    str(e),
                    kind=e.kind,
                    episode_index=episode_index,
                    details={**e.details, "steps": step_i, "partial_episode": partial},
                ) from e

            step_rec = {
                "t": step_i,
                "observations": {a: _jsonable(extract_observation(obs[a])) for a in actions},
                "actions": {a: _jsonable(actions[a]) for a in actions},
                "rewards": {a: float(rewards.get(a, 0.0)) for a in actions},
                "terminations": {a: bool(terminations.get(a, False)) for a in actions},
                "truncations": {a: bool(truncations.get(a, False)) for a in actions},
                "action_masks": masks,
                "infos": {a: _jsonable(infos.get(a, {})) for a in actions},
            }
            steps.append(step_rec)
            for a, r in rewards.items():
                returns[a] = returns.get(a, 0.0) + float(r)

            for agent, terminated in terminations.items():
                if terminated and agent in runtimes:
                    runtimes[agent].reset_agent(agent)

            obs, infos = next_obs, next_infos
            step_i += 1

        episode = _episode_dict(
            seed, episode_index, task_info, assignments, steps, action_mode, status="completed"
        )
        episode["returns"] = returns
        if writer is not None:
            writer.write_episode(episode)
        return {
            "episode_index": episode_index,
            "seed": seed,
            "status": "completed",
            "steps": len(steps),
            "returns": returns,
        }
    finally:
        env.close()


def _episode_dict(seed, episode_index, task_info, assignments, steps, action_mode, status):
    return {
        "schema": TRAJECTORY_SCHEMA,
        "episode_index": episode_index,
        "seed": seed,
        "status": status,
        "action_mode": action_mode,
        "task": {"env": task_info["env"], "adapter": task_info["adapter"], "version": task_info["version"]},
        "agents": task_info["agents"],
        "role_map": {a: a for a in task_info["agents"]},
        "policies": {k: v.digest for k, v in assignments.items()},
        "steps": steps,
    }


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _public_spec(value: Any) -> Any:
    """Drop private runtime injection keys from persisted task provenance."""
    if isinstance(value, dict):
        return {
            key: _public_spec(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, (list, tuple)):
        return [_public_spec(item) for item in value]
    return value
