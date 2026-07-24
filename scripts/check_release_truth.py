"""Fail CI when package, support, schema, README, and CI release claims drift."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    namespace: dict[str, str] = {}
    exec((ROOT / "rlx" / "_version.py").read_text(encoding="utf-8"), namespace)
    version = namespace["VERSION"]
    support = json.loads(
        (ROOT / "rlx" / "support-matrix.json").read_text(encoding="utf-8")
    )
    registry = json.loads(
        (ROOT / "rlx" / "schema-registry.json").read_text(encoding="utf-8")
    )
    assert support["release"] == version
    assert registry["release"] == version
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"RLX {version}" in readme
    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    )
    rows = workflow["jobs"]["test"]["strategy"]["matrix"]["include"]
    actual = {
        (row["os"], str(row["python-version"]), row["expected-arch"])
        for row in rows
    }
    required = {
        ("ubuntu-24.04", "3.12", "x86_64"),
        ("ubuntu-24.04", "3.13", "x86_64"),
        ("macos-15", "3.12", "arm64"),
        ("macos-15", "3.13", "arm64"),
    }
    assert required <= actual
    release_workflow = (
        ROOT / ".github" / "workflows" / "release-candidate.yml"
    ).read_text(encoding="utf-8")
    for required_release_step in (
        "actions/attest@v4",
        "cyclonedx-py environment",
        "check_pip_audit.py",
        "bandit -q -r rlx -ll",
        "check_secret_scan.py",
        "gimitest-boundary-proof.json",
        "SHA256SUMS",
    ):
        assert required_release_step in release_workflow
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
