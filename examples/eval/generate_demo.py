#!/usr/bin/env python3
"""Generate a runnable cyclic RPS eval fixture under examples/eval/demo/.

Usage (from repo root, with torch+pettingzoo installed):

  python examples/eval/generate_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from rlx.adapters.policy_custom_torch import (
    _embed_reference_cases,
    generate_reference_cases,
    load_runtime,
    verify_bundle_self,
)
from rlx.conformance.fixtures import build_fixed_action_rps_policy
from rlx.core.manifests import dump_yaml


def _with_source_cases(bundle: Path) -> Path:
    runtime = load_runtime(bundle)
    _embed_reference_cases(
        bundle,
        generate_reference_cases(
            runtime,
            observation=runtime.manifest["observation"],
            action=runtime.manifest["action"],
        ),
        provenance="source-conformance",
    )
    verify_bundle_self(bundle)
    return bundle


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite examples/eval/demo if it already exists",
    )
    args = parser.parse_args()
    out = Path(__file__).resolve().parent / "demo"
    if out.exists():
        if not args.force:
            # Empty leftover dirs from a failed prior run should not block.
            if any(out.iterdir()):
                print(f"refusing to overwrite existing {out} (pass --force)", file=sys.stderr)
                return 2
            out.rmdir()
        else:
            import shutil

            shutil.rmtree(out)
    out.mkdir(parents=True)
    roles = ["player_0", "player_1"]
    _with_source_cases(build_fixed_action_rps_policy(out / "rock.rlx", role=roles, action=0, name="rock"))
    _with_source_cases(build_fixed_action_rps_policy(out / "paper.rlx", role=roles, action=1, name="paper"))
    _with_source_cases(
        build_fixed_action_rps_policy(out / "scissors.rlx", role=roles, action=2, name="scissors")
    )
    dump_yaml(
        {
            "schema": "rlx.population/v0alpha1",
            "name": "cyclic-rps",
            "members": [
                {"policy": "./rock.rlx", "weight": 1.0, "tags": ["rock"]},
                {"policy": "./paper.rlx", "weight": 1.0, "tags": ["paper"]},
                {"policy": "./scissors.rlx", "weight": 1.0, "tags": ["scissors"]},
            ],
        },
        out / "population.yaml",
    )
    dump_yaml(
        {
            "schema": "rlx.evaluation/v0alpha1",
            "name": "cyclic-matrix",
            "interaction": "parallel",
            "task": {
                "adapter": "pettingzoo-parallel",
                "env": "rlx/competitive_rps_v0",
                "config": {"max_cycles": 1},
            },
            "assignments": {
                "player_0": {"kind": "crossplay", "population": "./population.yaml"},
                "player_1": {"kind": "crossplay", "population": "./population.yaml"},
            },
            "seeds": {"start": 0, "count": 1},
            "action_mode": "deterministic",
            "metrics": ["payoff_matrix", "mean_return", "win_rate"],
        },
        out / "evaluation.yaml",
    )
    print(f"Wrote runnable fixture to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
