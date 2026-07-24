"""Fail a release when a pip-audit report is incomplete or vulnerable."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def check_report(document: dict[str, Any], *, allowed_unpublished: set[str]) -> None:
    dependencies = document.get("dependencies")
    if not isinstance(dependencies, list) or not dependencies:
        raise ValueError("pip-audit report has no dependency records")
    skipped: set[str] = set()
    vulnerabilities: list[str] = []
    for dependency in dependencies:
        if not isinstance(dependency, dict) or not isinstance(dependency.get("name"), str):
            raise ValueError("pip-audit dependency record is malformed")
        name = dependency["name"]
        reason = dependency.get("skip_reason")
        if reason:
            skipped.add(name)
        vulns = dependency.get("vulns")
        if isinstance(vulns, list):
            vulnerabilities.extend(
                f"{name}:{item.get('id', 'unknown')}"
                for item in vulns
                if isinstance(item, dict)
            )
    unexpected_skips = sorted(skipped - allowed_unpublished)
    missing_expected = sorted(allowed_unpublished - skipped)
    if unexpected_skips or missing_expected:
        raise ValueError(
            "pip-audit coverage mismatch: "
            f"unexpected_skips={unexpected_skips}, missing_expected={missing_expected}"
        )
    if vulnerabilities:
        raise ValueError(
            "known dependency vulnerabilities: " + ", ".join(vulnerabilities)
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--allow-unpublished", action="append", default=[])
    args = parser.parse_args()
    document = json.loads(args.report.read_text(encoding="utf-8"))
    check_report(document, allowed_unpublished=set(args.allow_unpublished))
    print(
        json.dumps(
            {
                "ok": True,
                "dependencies": len(document["dependencies"]),
                "allowed_unpublished": sorted(args.allow_unpublished),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
