"""Tests for the local file:// catalog stub (not a hosted catalog)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arena.cli.main import main
from arena.core.catalog import CATALOG_LIST_SCHEMA, CATALOG_MODE, list_local_catalog
from arena.core.errors import StoreError
from arena.core.mirror import push_artifact
from arena.core.sdk import Policy


def test_list_local_catalog_empty_root(tmp_path: Path) -> None:
    root = tmp_path / "empty-mirror"
    root.mkdir()
    report = list_local_catalog(root)
    assert report["schema"] == CATALOG_LIST_SCHEMA
    assert report["mode"] == CATALOG_MODE
    assert report["hosted"] is False
    assert report["count"] == 0
    assert report["artifacts"] == []
    assert "not a hosted" in report["note"].lower()


def test_list_local_catalog_after_file_push(tmp_path: Path) -> None:
    source = Path("examples/eval/demo/rock.arena").resolve()
    expected = Policy.load(source).digest
    mirror = tmp_path / "mirror"
    pushed = push_artifact(source, mirror.as_uri(), verify=True)
    assert pushed["identity"] == expected

    report = list_local_catalog(mirror)
    assert report["hosted"] is False
    assert report["count"] == 1
    entry = report["artifacts"][0]
    assert entry["identity"] == expected
    assert entry["kind"] == "policy"
    assert entry["uri"] == f"{mirror.resolve().as_uri()}#{expected}"
    assert entry["file_count"] >= 1
    assert entry["bytes"] > 0

    via_uri = list_local_catalog(mirror.resolve().as_uri())
    assert via_uri["artifacts"][0]["identity"] == expected


def test_list_local_catalog_skips_invalid_descriptor(tmp_path: Path) -> None:
    root = tmp_path / "mirror"
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "not-a-digest.json").write_text('{"schema": "nope"}\n', encoding="utf-8")
    report = list_local_catalog(root)
    assert report["count"] == 0
    assert report["warnings"]
    assert any("skipping" in warning for warning in report["warnings"])


def test_list_local_catalog_rejects_missing_and_remote(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    with pytest.raises(StoreError, match="does not exist"):
        list_local_catalog(missing)

    with pytest.raises(StoreError, match="file:///"):
        list_local_catalog("hf://models/lab/arena")

    not_dir = tmp_path / "file.txt"
    not_dir.write_text("x", encoding="utf-8")
    with pytest.raises(StoreError, match="must be a directory"):
        list_local_catalog(not_dir)


def test_cli_catalog_local_json(tmp_path: Path, capsys) -> None:
    source = Path("examples/eval/demo/rock.arena").resolve()
    mirror = tmp_path / "cli-mirror"
    push_artifact(source, mirror.as_uri(), verify=True)

    assert main(["catalog", "local", str(mirror), "--json"]) == 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["schema"] == "arena.cli-result/v1"
    assert envelope["command"] == "catalog local"
    payload = envelope["data"]
    assert payload["schema"] == CATALOG_LIST_SCHEMA
    assert payload["hosted"] is False
    assert payload["count"] == 1
    assert payload["artifacts"][0]["identity"].startswith("sha256:")

    assert main(["catalog", "local", str(mirror)]) == 0
    human = capsys.readouterr().out
    assert "hosted: False" in human
    assert "local-file-stub" in human
    assert "not a hosted" in human.lower()
