"""Fail CI when package, support, schema, README, and CI release claims drift."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    namespace: dict[str, str] = {}
    exec((ROOT / "arena" / "_version.py").read_text(encoding="utf-8"), namespace)
    version = namespace["VERSION"]
    support = json.loads(
        (ROOT / "arena" / "support-matrix.json").read_text(encoding="utf-8")
    )
    registry = json.loads(
        (ROOT / "arena" / "schema-registry.json").read_text(encoding="utf-8")
    )
    assert support["release"] == version
    assert registry["release"] == version
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"Arena {version}" in readme
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
    # Experimental scaffolding must stay present and marked allow-failure so it
    # cannot silently become a required stable claim.
    experimental = {
        (row["os"], str(row["python-version"]), row["expected-arch"])
        for row in rows
        if row.get("tier") == "experimental"
    }
    required_experimental = {
        ("windows-latest", "3.12", "amd64"),
        ("windows-latest", "3.13", "amd64"),
        ("ubuntu-24.04-arm", "3.12", "aarch64"),
        ("macos-13", "3.12", "x86_64"),
    }
    assert required_experimental <= experimental
    for row in rows:
        if row.get("tier") == "experimental":
            assert row.get("allow-failure") is True
        if (row["os"], str(row["python-version"]), row["expected-arch"]) in required:
            assert row.get("tier", "stable") == "stable"
            assert row.get("allow-failure") is False
    platforms = {
        (str(row["os"]), str(row["arch"]), str(row["status"]))
        for row in support.get("platforms", [])
    }
    assert ("linux", "x86_64", "stable") in platforms
    assert ("darwin", "arm64", "stable") in platforms
    assert ("win32", "amd64", "experimental") in platforms
    assert ("linux", "aarch64", "experimental") in platforms
    assert ("darwin", "x86_64", "experimental") in platforms
    release_workflow = (
        ROOT / ".github" / "workflows" / "release-candidate.yml"
    ).read_text(encoding="utf-8")
    for required_release_step in (
        "actions/attest@v4",
        "cyclonedx-py environment",
        "check_pip_audit.py",
        "bandit -q -r arena -ll",
        "check_secret_scan.py",
        "gimitest-boundary-proof.json",
        "SHA256SUMS",
    ):
        assert required_release_step in release_workflow
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
