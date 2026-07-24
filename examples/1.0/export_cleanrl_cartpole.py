"""Export a pinned CleanRL CartPole DQN checkpoint, then verify source-free."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from rlx.adapters.policy_custom_torch import (
    export_module_from_checkpoint,
    verify_bundle_self,
)
from rlx.core.sdk import Policy

PINNED_CLEANRL_COMMIT = "fe8d8a03c41a7ef5b523e2e354bd01c363e786bb"


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleanrl-checkout", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    checkout = Path(args.cleanrl_checkout).resolve()
    revision = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != PINNED_CLEANRL_COMMIT:
        raise SystemExit(
            f"CleanRL checkout must be {PINNED_CLEANRL_COMMIT}, got {revision}"
        )
    cases = [
        {"observation": [0.0, 0.0, 0.0, 0.0], "mode": "deterministic"},
        {"observation": [0.05, 0.2, -0.03, -0.4], "mode": "deterministic"},
        {"observation": [-0.08, -0.6, 0.12, 0.8], "mode": "deterministic"},
        {"observation": [0.2, 1.5, -0.2, -1.0], "mode": "deterministic"},
    ]
    bundle = export_module_from_checkpoint(
        module_ref="examples.integrations.cleanrl_bridge:build_actor",
        out_dir=args.out,
        role="agent",
        name="cleanrl-cartpole-dqn",
        observation={
            "type": "Box",
            "shape": [4],
            "dtype": "float32",
            "low": [-4.8, -3.4028235e38, -0.41887903, -3.4028235e38],
            "high": [4.8, 3.4028235e38, 0.41887903, 3.4028235e38],
        },
        action={
            "type": "Discrete",
            "n": 2,
            "dtype": "int64",
            "masks": "none",
        },
        source=args.checkpoint,
        reference_cases=cases,
        source_revision=f"cleanrl@{revision}",
    )
    verification = verify_bundle_self(bundle)
    policy = Policy.load(bundle)
    print(
        json.dumps(
            {
                "schema": "rlx.cleanrl-export-proof/v1",
                "ok": True,
                "cleanrl_commit": revision,
                "policy_digest": policy.digest,
                "verification": verification,
                "out": str(Path(bundle).resolve()),
                "consumer_command": (
                    "python -c \"from rlx import Policy; "
                    f"print(Policy.load('{Path(bundle).resolve()}').digest)\""
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
