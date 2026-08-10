#!/usr/bin/env python3
"""Exercise real free environments through the Arena Match API.

Runs:
1. Gymnasium CartPole-v1 (via PettingZoo Parallel entrypoint_bundle)
2. OpenSpiel frozen tic_tac_toe (stable openspiel:// catalog)

Fails loud when required extras are missing. Does **not** treat unset OpenEnv /
Gimitest / cloud-store credentials as success — those are reported as skipped
with an explicit reason (and non-zero exit if ``--require CAP`` is set).

Usage (repo root of this worktree / checkout)::

    python examples/integrations/run_real_env_match.py --out /tmp/arena-env-smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CARTPOLE_ENTRY = Path(__file__).resolve().parent / "gymnasium_cartpole_parallel.py"


def _require_import(name: str, *, extra: str) -> None:
    try:
        __import__(name)
    except ImportError as exc:
        raise SystemExit(
            f"missing dependency {name!r}; install with: pip install 'arena[{extra}]'\n"
            f"detail: {exc}"
        ) from exc


def _doctor_capability(name: str) -> dict[str, Any]:
    from arena.core.support import capability_report

    return capability_report(name)


def _fail_loud_if_unqualified(name: str) -> dict[str, Any]:
    report = _doctor_capability(name)
    if report["local_status"] != "ready":
        raise SystemExit(
            f"capability {name!r} is {report['local_status']}: "
            f"{report.get('repair') or 'see arena doctor --capability ' + name}"
        )
    if report.get("credentials_required") and report.get("authentication_attempted"):
        raise SystemExit(
            f"capability {name!r} unexpectedly attempted authentication during doctor; "
            "refusing to treat live credentials as a smoke success"
        )
    return report


def _export_cartpole_policy(out: Path, *, observation: dict[str, Any]) -> Path:
    import torch

    from arena.adapters.policy_custom_torch import build_module, export_policy

    architecture = {
        "type": "mlp_categorical",
        "observation_dim": 4,
        "hidden_dims": [16],
        "action_n": 2,
    }
    module = build_module(architecture)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()
        # Prefer action 1 slightly so the episode moves.
        last = None
        for layer in module.net:  # type: ignore[attr-defined]
            if isinstance(layer, torch.nn.Linear):
                last = layer
        assert last is not None
        last.bias[1] = 1.0
    action = {"type": "Discrete", "n": 2, "dtype": "int64", "masks": "none"}
    return export_policy(
        out_dir=out,
        name="cartpole-smoke",
        roles=["agent"],
        observation=observation,
        action=action,
        architecture=architecture,
        state_dict=module.state_dict(),
        preprocessing={"id": "normalize_v0", "mean": 0.0, "std": 1.0},
        modes=["deterministic", "stochastic"],
        lineage={"fixture": "env-smoke", "env": "CartPole-v1"},
    )


def _export_openspiel_policy(out: Path) -> Path:
    import torch

    from arena.adapters.policy_custom_torch import build_module, export_policy

    architecture = {
        "type": "mlp_categorical",
        "observation_dim": 27,
        "hidden_dims": [16],
        "action_n": 9,
    }
    module = build_module(architecture)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()
    return export_policy(
        out_dir=out,
        name="openspiel-ttt-first-legal",
        roles=["player_0", "player_1"],
        observation={
            "type": "Box",
            "shape": [27],
            "dtype": "float32",
            "low": 0.0,
            "high": 1.0,
        },
        action={
            "type": "Discrete",
            "n": 9,
            "dtype": "int64",
            "masks": "required",
        },
        architecture=architecture,
        state_dict=module.state_dict(),
        preprocessing={"id": "normalize_v0", "mean": 0.0, "std": 1.0},
        modes=["deterministic", "stochastic"],
        lineage={"fixture": "env-smoke", "game": "tic_tac_toe"},
    )


def run_cartpole_match(out_dir: Path) -> dict[str, Any]:
    _require_import("gymnasium", extra="pettingzoo")
    _require_import("pettingzoo", extra="pettingzoo")
    _require_import("torch", extra="torch")

    from arena.adapters.task_pettingzoo.adapter import describe_task
    from arena.core.identity import digest_uri, sha256_file
    from arena.core.sdk import Match, Policy, Task

    digest = digest_uri(sha256_file(CARTPOLE_ENTRY))
    task_spec = {
        "adapter": "pettingzoo-parallel",
        "env": "examples/integrations/gymnasium_cartpole_parallel",
        "interaction": "parallel",
        "trust_task_code": True,
        "config": {"max_cycles": 50},
        "packaging": {
            "kind": "entrypoint_bundle",
            "root": str(CARTPOLE_ENTRY.parent),
            "entrypoint": CARTPOLE_ENTRY.name,
            "digest": digest,
            "factory": "parallel_env",
            "trust_task_code": True,
        },
        "source_revision": "gymnasium:CartPole-v1",
    }
    # Export against the live task contract (CartPole uses ±inf velocity bounds).
    observation = describe_task(task_spec)["roles"]["agent"]["observation"]
    policy_path = _export_cartpole_policy(
        out_dir / "cartpole-policy.arena", observation=observation
    )
    task = Task.load(task_spec)
    result = Match(
        task=task,
        assignments={"agent": Policy.load(policy_path)},
    ).run(seeds=[0, 1], out=out_dir / "cartpole-match")
    if result["outcome"]["failure_count"]:
        raise SystemExit(f"CartPole match failed: {json.dumps(result['outcome'])}")
    if result["outcome"]["episodes_completed"] != 2:
        raise SystemExit(f"CartPole match incomplete: {json.dumps(result['outcome'])}")
    return {
        "env": "CartPole-v1",
        "adapter": "entrypoint_bundle+pettingzoo-parallel",
        "policy_digest": Policy.load(policy_path).digest,
        "outcome": result["outcome"],
        "run_id": result["run_id"],
    }


def run_openspiel_match(out_dir: Path) -> dict[str, Any]:
    _require_import("pyspiel", extra="openspiel")
    _require_import("torch", extra="torch")
    _fail_loud_if_unqualified("openspiel")

    from arena.core.sdk import Match, Policy, Task

    policy_path = _export_openspiel_policy(out_dir / "openspiel-policy.arena")
    task_path = REPO_ROOT / "examples/tasks/openspiel-tic-tac-toe.yaml"
    policy = Policy.load(policy_path)
    result = Match(
        task=Task.load(task_path),
        assignments={"player_0": policy, "player_1": policy},
    ).run(seeds=[0, 1, 2], out=out_dir / "openspiel-match")
    if result["outcome"]["failure_count"]:
        raise SystemExit(f"OpenSpiel match failed: {json.dumps(result['outcome'])}")
    if result["outcome"]["episodes_completed"] != 3:
        raise SystemExit(f"OpenSpiel match incomplete: {json.dumps(result['outcome'])}")
    return {
        "env": "openspiel://tic_tac_toe",
        "adapter": "openspiel",
        "policy_digest": policy.digest,
        "outcome": result["outcome"],
        "run_id": result["run_id"],
    }


def probe_optional_capabilities(required: set[str]) -> dict[str, Any]:
    """Report OpenEnv / Gimitest readiness without faking live success."""
    probes: dict[str, Any] = {}
    for name in ("openenv", "gimitest", "hf", "wandb", "mlflow", "oci"):
        try:
            report = _doctor_capability(name)
        except Exception as exc:  # noqa: BLE001 - surface doctor faults loudly
            probes[name] = {"ok": False, "error": str(exc)}
            if name in required:
                raise SystemExit(f"required capability {name!r} probe failed: {exc}") from exc
            continue
        ready = report["local_status"] == "ready"
        # Credential-backed capabilities must never count as authenticated smoke.
        authenticated = bool(report.get("authentication_attempted"))
        ok = ready and not authenticated
        probes[name] = {
            "ok": ok,
            "local_status": report["local_status"],
            "release_status": report["release_status"],
            "credentials_required": report.get("credentials_required"),
            "authentication_attempted": authenticated,
            "repair": report.get("repair"),
            "isolated_python_env": report.get("isolated_python_env"),
            "isolated_probe": report.get("isolated_probe"),
        }
        if name in required and not ok:
            raise SystemExit(
                f"required capability {name!r} not ready "
                f"(local_status={report['local_status']}, "
                f"auth_attempted={authenticated}): {report.get('repair')}"
            )
    return probes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/arena-env-smoke"),
        help="directory for exported policies and match runs",
    )
    parser.add_argument(
        "--skip-cartpole",
        action="store_true",
        help="skip Gymnasium CartPole Match smoke",
    )
    parser.add_argument(
        "--skip-openspiel",
        action="store_true",
        help="skip OpenSpiel frozen-game Match smoke",
    )
    parser.add_argument(
        "--require",
        action="append",
        default=[],
        help="fail if this capability is not locally ready (repeatable)",
    )
    args = parser.parse_args(argv)
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "schema": "arena.env-smoke/v1",
        "ok": True,
        "matches": {},
        "optional_capabilities": probe_optional_capabilities(set(args.require)),
    }
    if not args.skip_cartpole:
        summary["matches"]["cartpole"] = run_cartpole_match(out_dir)
    if not args.skip_openspiel:
        summary["matches"]["openspiel_tic_tac_toe"] = run_openspiel_match(out_dir)

    (out_dir / "env-smoke-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    raise SystemExit(main())
