#!/usr/bin/env python3
"""Fail-closed Hugging Face live store qualification.

Without ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` this exits non-zero and writes
``mode=credential-missing`` evidence. It never treats ``?simulate=`` as live
success and never flips the support matrix to stable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from arena.core.store_hf import HF_LIVE_RECIPE, qualify_hf_live


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Live HF push/pull qualification. Fail-closed when credentials are "
            "missing (mode=credential-missing, exit non-zero)."
        )
    )
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "destination",
        nargs="?",
        default=None,
        help="hf:// URI (default: ARENA_HF_LIVE_DEST or hf://models/ORG/REPO/arena placeholder)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Restored artifact path (only used when credentials are present)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Write arena.store-qualification/v1 JSON here (including credential-missing)",
    )
    args = parser.parse_args(argv)

    destination = args.destination
    if destination is None:
        import os

        destination = os.environ.get("ARENA_HF_LIVE_DEST") or "hf://models/ORG/REPO/arena"

    report = qualify_hf_live(
        args.source,
        destination,
        report_path=args.report,
        restored_out=args.out,
    )
    print(json.dumps(report, indent=2))
    if report.get("mode") == "credential-missing" or not report.get("ok"):
        print(
            "HF live qualification did not pass. "
            f"mode={report.get('mode')!r} ok={report.get('ok')!r}. "
            f"Recipe: {HF_LIVE_RECIPE}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
