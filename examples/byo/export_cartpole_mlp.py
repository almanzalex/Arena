"""Export a self-contained CartPole MLP as a BYO TorchScript policy bundle.

Does not require a CleanRL checkout. Optional ``--source`` loads weights;
otherwise a deterministic in-repo actor is scripted.

Example::

    python examples/byo/export_cartpole_mlp.py --out ./byo-cartpole.arena
    arena policy verify ./byo-cartpole.arena
    arena inspect ./byo-cartpole.arena --json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

# Repo root on sys.path so ``examples.byo…`` resolves when run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from arena.adapters.policy_custom_torch import (  # noqa: E402
    export_module_from_checkpoint,
    verify_bundle_self,
)
from arena.core.sdk import Policy  # noqa: E402
from examples.byo.cartpole_mlp import (  # noqa: E402
    CARTPOLE_ACTION,
    CARTPOLE_OBSERVATION,
    REFERENCE_CASES,
    build_actor,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export a self-contained CartPole MLP via BYO TorchScript "
            "(no CleanRL checkout required)."
        )
    )
    parser.add_argument("--out", required=True, help="Output .arena bundle directory")
    parser.add_argument(
        "--source",
        default=None,
        help="Optional state_dict checkpoint; omitted → deterministic demo weights",
    )
    parser.add_argument(
        "--source-revision",
        default="examples/byo@cartpole-mlp",
        help="Lineage label recorded on the manifest (excluded from content digest)",
    )
    parser.add_argument("--role", default="agent")
    parser.add_argument("--name", default="byo-cartpole-mlp")
    args = parser.parse_args(argv)

    source = args.source
    tmp_ckpt: Path | None = None
    if source is None:
        import torch

        tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        tmp_ckpt = Path(tmp.name)
        tmp.close()
        torch.save(build_actor().state_dict(), tmp_ckpt)
        source = str(tmp_ckpt)

    try:
        bundle = export_module_from_checkpoint(
            module_ref="examples.byo.cartpole_mlp:build_actor",
            out_dir=args.out,
            role=args.role,
            name=args.name,
            observation=CARTPOLE_OBSERVATION,
            action=CARTPOLE_ACTION,
            source=source,
            reference_cases=REFERENCE_CASES,
            source_revision=args.source_revision,
        )
    finally:
        if tmp_ckpt is not None:
            tmp_ckpt.unlink(missing_ok=True)

    verification = verify_bundle_self(bundle)
    policy = Policy.load(bundle)
    print(
        json.dumps(
            {
                "schema": "arena.byo-export-proof/v1",
                "ok": True,
                "export_path": "byo-torchscript",
                "module_ref": "examples.byo.cartpole_mlp:build_actor",
                "policy_digest": policy.digest,
                "verification": verification,
                "out": str(Path(bundle).resolve()),
                "consumer_command": (
                    "python -c \"from arena import Policy; "
                    f"print(Policy.load('{Path(bundle).resolve()}').digest)\""
                ),
                "cleanrl_note": (
                    "For the pinned CleanRL producer rehearsal see "
                    "examples/1.0/export_cleanrl_cartpole.py "
                    "(requires a CleanRL checkout at the pinned commit)."
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
