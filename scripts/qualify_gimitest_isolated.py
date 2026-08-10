#!/usr/bin/env python3
"""Qualify Gimitest from a resolver-isolated interpreter and emit R-06 evidence.

Requires ``ARENA_GIMITEST_PYTHON`` (see ``scripts/bootstrap_gimitest_worker.sh``).

This produces local isolated-worker evidence for release-index binding. It does
**not** flip the support matrix to stable; claimed-platform release CI must
repeat the proof against the exact release artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "qualifications" / "gimitest" / "R-06-gimitest.json"
EVIDENCE_SCHEMA = "arena.gimitest-qualification-evidence/v1"


def _die(message: str, code: int = 2) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(code)


def _suite(
    *,
    name: str,
    provider: str,
    provider_config: dict[str, Any],
    rock: Path,
    paper: Path,
) -> dict[str, Any]:
    return {
        "schema": "arena.evaluation/v0alpha1",
        "name": name,
        "provider": provider,
        "provider_config": provider_config,
        "interaction": "parallel",
        "task": {
            "adapter": "pettingzoo-parallel",
            "env": "arena/competitive_rps_v0",
            "interaction": "parallel",
            "config": {"max_cycles": 1},
        },
        "assignments": {"player_0": str(rock), "player_1": str(paper)},
        "seeds": [0],
        "action_mode": "deterministic",
        "metrics": ["mean_return"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="R-06 evidence JSON path (default: docs/qualifications/gimitest/R-06-gimitest.json)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Scratch directory for eval runs (default: temporary under --out parent)",
    )
    args = parser.parse_args(argv)

    isolated = os.environ.get("ARENA_GIMITEST_PYTHON", "").strip()
    if not isolated:
        _die(
            "ARENA_GIMITEST_PYTHON is unset. Run:\n"
            "  scripts/bootstrap_gimitest_worker.sh\n"
            "  export ARENA_GIMITEST_PYTHON=…\n"
            "  python scripts/qualify_gimitest_isolated.py"
        )
    isolated_path = Path(isolated)
    if not isolated_path.is_absolute():
        _die(f"ARENA_GIMITEST_PYTHON must be absolute, got {isolated!r}")
    if not isolated_path.is_file() or not os.access(isolated_path, os.X_OK):
        _die(f"ARENA_GIMITEST_PYTHON is not an executable file: {isolated}")
    # Compare abspaths, not resolve(): resolve() follows venv symlinks to the base
    # interpreter and would falsely treat distinct venvs as identical.
    parent_python = Path(os.path.abspath(sys.executable))
    worker_python = Path(os.path.abspath(str(isolated_path)))
    if parent_python == worker_python:
        _die(
            "ARENA_GIMITEST_PYTHON must point at a separate interpreter from the "
            f"parent process ({parent_python}). Bootstrap a fresh worker venv."
        )

    # Ensure the checkout is importable when invoked with a system/venv python.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from arena import __version__
    from arena.core.manifests import dump_json
    from arena.core.support import capability_report, doctor_report
    from arena.runtime.evaluation import run_evaluation

    doctor = doctor_report("gimitest")
    capability = capability_report("gimitest")
    if capability.get("local_status") != "ready":
        _die(
            "arena doctor --capability gimitest is not ready:\n"
            + json.dumps(capability, indent=2, sort_keys=True)
        )

    rock = (ROOT / "examples/eval/demo/rock.arena").resolve()
    paper = (ROOT / "examples/eval/demo/paper.arena").resolve()
    if not rock.is_dir() or not paper.is_dir():
        _die("missing examples/eval/demo/{rock,paper}.arena fixtures")

    work = args.work_dir or (args.out.parent / "_runs")
    work.mkdir(parents=True, exist_ok=True)

    isolation = {
        "mode": "subprocess",
        "python": str(worker_python),
        "timeout_seconds": 120,
    }
    native = run_evaluation(
        _suite(
            name="gimitest-r06-native",
            provider="native",
            provider_config={},
            rock=rock,
            paper=paper,
        ),
        policy_index={},
        out_dir=work / "native",
    )
    gimitest = run_evaluation(
        _suite(
            name="gimitest-r06-semantic-noop",
            provider="gimitest",
            provider_config={
                "semantic": {},
                "suite": "base-hooks",
                "test_class": "gimitest.gtest:GTest",
                "parameters": {"purpose": "R-06 isolated semantic-noop"},
                "isolation": isolation,
            },
            rock=rock,
            paper=paper,
        ),
        policy_index={},
        out_dir=work / "gimitest-noop",
    )
    transformed = run_evaluation(
        _suite(
            name="gimitest-r06-non-noop",
            provider="gimitest",
            provider_config={
                "semantic": {
                    "test_class": (
                        "arena.adapters.eval_gimitest.scenarios:RewardTransformScenario"
                    ),
                    "parameters": {"reward_scale": -1.0},
                },
                "suite": "base-hooks",
                "test_class": (
                    "arena.adapters.eval_gimitest.scenarios:RewardTransformScenario"
                ),
                "parameters": {
                    "purpose": "R-06 isolated non-no-op reward transform",
                    "reward_scale": -1.0,
                },
                "isolation": isolation,
            },
            rock=rock,
            paper=paper,
        ),
        policy_index={},
        out_dir=work / "gimitest-non-noop",
    )

    checks: dict[str, dict[str, Any]] = {
        "worker_distinct_from_parent": {
            "ok": parent_python != worker_python,
            "parent_python": str(parent_python),
            "worker_python": str(worker_python),
        },
        "doctor_ready": {
            "ok": capability.get("local_status") == "ready",
            "usable_today": capability.get("usable_today"),
            "release_status": capability.get("release_status"),
            "isolated_probe": capability.get("isolated_probe"),
        },
        "semantic_noop_matches_native_intent_and_result": {
            "ok": (
                native["evaluation_intent_digest"]
                == gimitest["evaluation_intent_digest"]
                and native["semantic_result_digest"] == gimitest["semantic_result_digest"]
                and native["execution_binding_digest"]
                != gimitest["execution_binding_digest"]
            ),
        },
        "non_noop_changes_intent_and_result": {
            "ok": (
                transformed["evaluation_intent_digest"]
                != native["evaluation_intent_digest"]
                and transformed["semantic_result_digest"]
                != native["semantic_result_digest"]
            ),
            "returns": (
                transformed.get("cell_results") or [{}]
            )[0]
            .get("episodes", [{}])[0]
            .get("returns"),
        },
        "worker_lineage_recorded": {
            "ok": bool((gimitest.get("provider") or {}).get("worker")),
            "provider": gimitest.get("provider"),
        },
        "runs_complete": {
            "ok": all(
                run.get("state") == "complete"
                for run in (native, gimitest, transformed)
            ),
            "states": {
                "native": native.get("state"),
                "gimitest": gimitest.get("state"),
                "transformed": transformed.get("state"),
            },
        },
    }
    ok = all(item["ok"] for item in checks.values())
    # Preview remains honest: local evidence is not a stable matrix flip.
    stable_claim = False
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "gate": "R-06",
        "status": "pass" if ok else "failed",
        "mode": "local-isolated",
        "release": __version__,
        "generated_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "parent_python": str(parent_python),
        "isolated_python": str(worker_python),
        "doctor_ok": bool(doctor.get("ok")),
        "capability": {
            "local_status": capability.get("local_status"),
            "usable_today": capability.get("usable_today"),
            "release_status": capability.get("release_status"),
            "evidence": capability.get("evidence"),
            "isolated_probe": capability.get("isolated_probe"),
        },
        "checks": checks,
        "digests": {
            "native_evaluation_intent_digest": native["evaluation_intent_digest"],
            "native_semantic_result_digest": native["semantic_result_digest"],
            "native_execution_binding_digest": native["execution_binding_digest"],
            "gimitest_evaluation_intent_digest": gimitest["evaluation_intent_digest"],
            "gimitest_semantic_result_digest": gimitest["semantic_result_digest"],
            "gimitest_execution_binding_digest": gimitest["execution_binding_digest"],
            "non_noop_evaluation_intent_digest": transformed[
                "evaluation_intent_digest"
            ],
            "non_noop_semantic_result_digest": transformed["semantic_result_digest"],
            "non_noop_execution_binding_digest": transformed[
                "execution_binding_digest"
            ],
        },
        "stable_claim": stable_claim,
        "bind_as": "evidence/R-06-gimitest.json",
        "notes": (
            "Local resolver-isolated Gimitest qualification. Support matrix stays "
            "preview until claimed-platform release CI repeats this proof against "
            "the exact wheel and binds the report into the release evidence index."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dump_json(evidence, args.out)
    print(json.dumps({"ok": ok, "out": str(args.out), "stable_claim": stable_claim}, indent=2))
    if not ok:
        print(json.dumps(evidence, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
