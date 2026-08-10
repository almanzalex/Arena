#!/usr/bin/env python3
"""Export portable RPS policies and run parallel + AEC classic/rps_v2 matches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _stable_outcome_digest(run: dict[str, Any]) -> str:
    from arena.core.identity import canonical_json, digest_uri, sha256_bytes

    projection = {
        "task": run.get("task"),
        "seeds": run.get("seeds"),
        "action_mode": run.get("action_mode"),
        "outcome": run.get("outcome"),
        "assignments": {
            role: {"name": meta.get("name"), "digest": meta.get("digest")}
            for role, meta in (run.get("assignments") or {}).items()
        },
        "episode_returns": [
            ep.get("returns") for ep in (run.get("episodes") or [])
        ],
    }
    return digest_uri(sha256_bytes(canonical_json(projection)))


def run_multiagent_demo(*, out: Path, seeds: list[int] | None = None) -> dict[str, Any]:
    from arena.conformance.fixtures import build_fixed_action_rps_policy
    from arena.core.manifests import load_manifest
    from arena.core.sdk import Match, Policy, Task

    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    policies_dir = out / "policies"
    policies_dir.mkdir(parents=True, exist_ok=True)
    seed_list = list(seeds) if seeds is not None else [0, 1, 2]

    rock = build_fixed_action_rps_policy(
        policies_dir / "rock.arena",
        role=["player_0", "player_1"],
        action=0,
        name="rock",
    )
    paper = build_fixed_action_rps_policy(
        policies_dir / "paper.arena",
        role=["player_0", "player_1"],
        action=1,
        name="paper",
    )
    assignments = {
        "player_0": Policy.load(rock),
        "player_1": Policy.load(paper),
    }

    parallel_task_path = ROOT / "examples" / "tasks" / "pettingzoo-classic-rps.yaml"
    aec_task_path = ROOT / "examples" / "tasks" / "pettingzoo-classic-rps-aec.yaml"
    parallel_task = Task.load(parallel_task_path)
    aec_task = Task.load(aec_task_path)

    parallel_run = Match(
        task=parallel_task,
        assignments=assignments,
        action_mode="deterministic",
    ).run(seeds=seed_list, record=True, out=out / "parallel")
    aec_run = Match(
        task=aec_task,
        assignments=assignments,
        action_mode="deterministic",
    ).run(seeds=seed_list, record=True, out=out / "aec")

    summary: dict[str, Any] = {
        "schema": "arena.demo-multiagent/v1",
        "ok": True,
        "env": "classic/rps_v2",
        "seeds": seed_list,
        "policies": {
            "player_0": {
                "path": "policies/rock.arena",
                "digest": assignments["player_0"].digest,
                "name": assignments["player_0"].name,
            },
            "player_1": {
                "path": "policies/paper.arena",
                "digest": assignments["player_1"].digest,
                "name": assignments["player_1"].name,
            },
        },
        "tasks": {
            "parallel": {
                "path": str(parallel_task_path.relative_to(ROOT)),
                "digest": load_manifest(parallel_task_path).get("digest"),
                "interaction": "parallel",
            },
            "aec": {
                "path": str(aec_task_path.relative_to(ROOT)),
                "digest": load_manifest(aec_task_path).get("digest"),
                "interaction": "aec",
            },
        },
        "parallel": {
            "outcome": parallel_run["outcome"],
            "outcome_digest": _stable_outcome_digest(parallel_run),
            "episode_returns": [ep.get("returns") for ep in parallel_run.get("episodes") or []],
        },
        "aec": {
            "outcome": aec_run["outcome"],
            "outcome_digest": _stable_outcome_digest(aec_run),
            "episode_returns": [ep.get("returns") for ep in aec_run.get("episodes") or []],
        },
    }
    # Rock vs paper: player_0 loses every seeded episode under both interactions.
    summary["parity"] = {
        "episode_returns_equal": summary["parallel"]["episode_returns"]
        == summary["aec"]["episode_returns"],
        "expected_returns": {"player_0": -1.0, "player_1": 1.0},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="./arena-ma-demo")
    parser.add_argument("--json", action="store_true", help="Print summary JSON to stdout")
    args = parser.parse_args(argv)
    summary = run_multiagent_demo(out=Path(args.out))
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(
            f"ok={summary['ok']} parallel_digest={summary['parallel']['outcome_digest']} "
            f"aec_digest={summary['aec']['outcome_digest']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
