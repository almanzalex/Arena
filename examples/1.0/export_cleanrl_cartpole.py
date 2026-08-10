"""Export a pinned CleanRL CartPole DQN checkpoint, then verify source-free.

Requires a CleanRL git checkout at the pinned commit plus a trained checkpoint.
For a CleanRL-free rehearsal of the same BYO TorchScript producer path, use::

    python examples/byo/export_cartpole_mlp.py --out ./byo-cartpole.arena
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from arena.adapters.policy_custom_torch import (
    export_module_from_checkpoint,
    verify_bundle_self,
)
from arena.core.sdk import Policy

PINNED_CLEANRL_COMMIT = "fe8d8a03c41a7ef5b523e2e354bd01c363e786bb"
BYO_FALLBACK = (
    "No CleanRL checkout available. Skip this producer rehearsal, or use the "
    "self-contained BYO path (no CleanRL required):\n"
    "  python examples/byo/export_cartpole_mlp.py --out ./byo-cartpole.arena\n"
    f"Pinned CleanRL commit when you do have a checkout: {PINNED_CLEANRL_COMMIT}"
)


def _skip(message: str) -> int:
    print(message, file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    parser = argparse.ArgumentParser(
        description=(
            "Export a pinned CleanRL CartPole DQN checkpoint via BYO TorchScript. "
            "Requires --cleanrl-checkout at the pinned commit; otherwise exits 2 "
            "with a pointer to examples/byo/export_cartpole_mlp.py."
        )
    )
    parser.add_argument("--cleanrl-checkout", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    checkout = Path(args.cleanrl_checkout).resolve()
    checkpoint = Path(args.checkpoint).resolve()

    if not checkout.is_dir():
        return _skip(
            f"CleanRL checkout not found at {checkout}.\n{BYO_FALLBACK}"
        )
    if not checkpoint.is_file():
        return _skip(
            f"CleanRL checkpoint not found at {checkpoint}.\n{BYO_FALLBACK}"
        )

    try:
        revision = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return _skip(
            f"Could not read CleanRL git revision at {checkout}: {exc}\n{BYO_FALLBACK}"
        )
    if revision != PINNED_CLEANRL_COMMIT:
        return _skip(
            f"CleanRL checkout must be {PINNED_CLEANRL_COMMIT}, got {revision}.\n"
            f"{BYO_FALLBACK}"
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
        source=str(checkpoint),
        reference_cases=cases,
        source_revision=f"cleanrl@{revision}",
    )
    verification = verify_bundle_self(bundle)
    policy = Policy.load(bundle)
    print(
        json.dumps(
            {
                "schema": "arena.cleanrl-export-proof/v1",
                "ok": True,
                "cleanrl_commit": revision,
                "policy_digest": policy.digest,
                "verification": verification,
                "out": str(Path(bundle).resolve()),
                "consumer_command": (
                    "python -c \"from arena import Policy; "
                    f"print(Policy.load('{Path(bundle).resolve()}').digest)\""
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
