"""Require a detect-secrets JSON scan to contain no unresolved findings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    document = json.loads(args.report.read_text(encoding="utf-8"))
    results = document.get("results")
    if not isinstance(results, dict):
        raise ValueError("detect-secrets report has no results mapping")
    findings = sum(
        len(items) for items in results.values() if isinstance(items, list)
    )
    if findings:
        raise ValueError(
            f"detect-secrets found {findings} unresolved candidate secret(s)"
        )
    print(json.dumps({"ok": True, "findings": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
