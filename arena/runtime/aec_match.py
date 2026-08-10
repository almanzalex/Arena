"""PettingZoo AEC match runner (0.2). Shares trajectory schema with Parallel."""

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
from arena.core.manifests import RUN_SCHEMA, TRAJECTORY_SCHEMA, dump_json, dump_yaml
from arena.core.sdk import Policy
from arena.runtime.trajectory import TrajectoryWriter


def run_aec_match(
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

    task_spec = dict(task_spec)
    task_spec["interaction"] = "aec"
    task_info = describe_task(task_spec)
    if bool(task_info.get("dynamic_agents")):
        raise SchemaError(
            "Dynamic agent birth/removal requires interaction=dynamic_aec. "
            "Fixed-agent AEC refuses lifecycle changes."
        )
    roles_meta = task_info["roles"]

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

    runtimes = {k: load_runtime(v.root) for k, v in assignments.items()}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + sha256_canonical(
        {"seeds": seeds, "assignments": {k: v.digest for k, v in sorted(assignments.items())}}
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
        try:
            summary = _run_aec_episode(
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
            failures.append(
                {
                    "episode_index": ep_idx,
                    "seed": seed,
                    "kind": e.kind,
                    "message": str(e),
                    "agent": e.agent,
                    "details": e.details,
                }
            )
            if retain_incomplete and writer is not None and e.details.get("partial_episode"):
                writer.write_episode(e.details["partial_episode"])
            episode_summaries.append(
                {
                    "episode_index": ep_idx,
                    "seed": seed,
                    "status": e.kind,
                    "steps": e.details.get("steps", 0),
                }
            )
        except Exception as e:  # noqa: BLE001
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
        "interaction": "aec",
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


def _run_aec_episode(
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
            env.reset(seed=seed)
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
        returns = {a: 0.0 for a in assignments}
        # Accumulate rewards until all agents have acted once per joint tick for recording.
        pending_obs: dict[str, Any] = {}
        pending_actions: dict[str, Any] = {}
        pending_rewards: dict[str, float] = {}
        pending_terms: dict[str, bool] = {}
        pending_truncs: dict[str, bool] = {}
        pending_infos: dict[str, Any] = {}
        agents_this_tick: set[str] = set()

        while env.agents:
            if time.monotonic() - ep_start > timeout:
                raise RuntimeFailure(
                    "episode timeout",
                    kind="timeout",
                    episode_index=episode_index,
                    details={
                        "steps": step_i,
                        "partial_episode": {
                            "episode_index": episode_index,
                            "seed": seed,
                            "steps": steps,
                            "returns": returns,
                            "status": "timeout",
                        },
                    },
                )
            agent = env.agent_selection
            # PettingZoo classic AEC keeps done agents in `env.agents` until a
            # dead-step advances them with action=None. Do not call policies.
            if bool((getattr(env, "terminations", {}) or {}).get(agent, False)) or bool(
                (getattr(env, "truncations", {}) or {}).get(agent, False)
            ):
                try:
                    env.step(None)
                except TaskRuntimeError as e:
                    raise RuntimeFailure(
                        str(e),
                        kind=e.kind,
                        episode_index=episode_index,
                        agent=agent,
                        details={**e.details, "steps": step_i},
                    ) from e
                if hasattr(env, "rewards"):
                    for a, r in (env.rewards or {}).items():
                        if a in assignments:
                            rr = float(r)
                            pending_rewards[a] = pending_rewards.get(a, 0.0) + rr
                            returns[a] = returns.get(a, 0.0) + rr
                if agents_this_tick and (
                    agents_this_tick >= set(assignments) or not env.agents
                ):
                    steps.append(
                        {
                            "observations": dict(pending_obs),
                            "actions": dict(pending_actions),
                            "rewards": dict(pending_rewards),
                            "terminations": dict(pending_terms),
                            "truncations": dict(pending_truncs),
                            "infos": dict(pending_infos),
                        }
                    )
                    step_i += 1
                    pending_obs.clear()
                    pending_actions.clear()
                    pending_rewards.clear()
                    pending_terms.clear()
                    pending_truncs.clear()
                    pending_infos.clear()
                    agents_this_tick.clear()
                if not env.agents:
                    break
                continue
            if agent not in runtimes:
                raise RuntimeFailure(
                    f"missing policy for agent {agent!r}",
                    kind="invalid_action",
                    episode_index=episode_index,
                    agent=agent,
                    details={"steps": step_i},
                )
            raw_obs = env.observe(agent)
            obs = extract_observation(raw_obs)
            mask = extract_action_mask(raw_obs)
            info = {}
            if hasattr(env, "infos") and agent in getattr(env, "infos", {}):
                info = env.infos[agent] or {}
                mask = mask if mask is not None else (extract_action_mask(info) if info else None)
            # Some AEC envs expose masks via observation dict — keep None if absent.
            try:
                action = runtimes[agent].act(
                    obs,
                    mode="deterministic" if action_mode == "deterministic" else "stochastic",
                    action_mask=mask,
                    rng=np.random.default_rng(seed + step_i),
                    agent_id=agent,
                )
            except ConformanceError as e:
                raise RuntimeFailure(
                    str(e),
                    kind="policy_failure",
                    episode_index=episode_index,
                    agent=agent,
                    details={"steps": step_i},
                ) from e
            act_space = task_info["roles"][agent]["action"]
            try:
                validate_runtime_action(action, action=act_space, agent=agent)
            except (ConformanceError, SchemaError) as e:
                raise RuntimeFailure(
                    str(e),
                    kind="invalid_action",
                    episode_index=episode_index,
                    agent=agent,
                    details={"steps": step_i, "action": action},
                ) from e

            pending_obs[agent] = _jsonable(obs)
            pending_actions[agent] = _jsonable(action)
            pending_infos[agent] = _jsonable(info)
            agents_this_tick.add(agent)
            try:
                env.step(action)
            except TaskRuntimeError as e:
                raise RuntimeFailure(
                    str(e),
                    kind=e.kind,
                    episode_index=episode_index,
                    agent=agent,
                    details={**e.details, "steps": step_i},
                ) from e
            # PettingZoo AEC: after a resolving step, rewards may be set for all agents.
            if hasattr(env, "rewards"):
                for a, r in (env.rewards or {}).items():
                    if a in assignments:
                        rr = float(r)
                        pending_rewards[a] = pending_rewards.get(a, 0.0) + rr
                        returns[a] = returns.get(a, 0.0) + rr
            pending_terms[agent] = bool(env.terminations.get(agent, False))
            pending_truncs[agent] = bool(env.truncations.get(agent, False))
            for a in assignments:
                if hasattr(env, "terminations"):
                    pending_terms[a] = bool(env.terminations.get(a, False))
                if hasattr(env, "truncations"):
                    pending_truncs[a] = bool(env.truncations.get(a, False))

            # Joint record when all assigned agents acted, or flush a partial final
            # cycle when a sequential game terminates before every player acts.
            if agents_this_tick >= set(assignments) or not env.agents:
                steps.append(
                    {
                        "observations": dict(pending_obs),
                        "actions": dict(pending_actions),
                        "rewards": dict(pending_rewards),
                        "terminations": dict(pending_terms),
                        "truncations": dict(pending_truncs),
                        "infos": dict(pending_infos),
                    }
                )
                step_i += 1
                pending_obs.clear()
                pending_actions.clear()
                pending_rewards.clear()
                pending_terms.clear()
                pending_truncs.clear()
                pending_infos.clear()
                agents_this_tick.clear()

            if not env.agents:
                break

        episode = {
            "schema": TRAJECTORY_SCHEMA,
            "episode_index": episode_index,
            "seed": seed,
            "steps": steps,
            "returns": returns,
            "status": "completed",
            "interaction": "aec",
            "action_mode": action_mode,
            "task": {
                "env": task_info["env"],
                "adapter": task_info["adapter"],
                "version": task_info["version"],
            },
            "agents": task_info["agents"],
            "role_map": {agent: agent for agent in task_info["agents"]},
            "policies": {key: policy.digest for key, policy in assignments.items()},
        }
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
        if hasattr(env, "close"):
            env.close()


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


def _public_spec(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _public_spec(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, (list, tuple)):
        return [_public_spec(item) for item in value]
    return value
