from __future__ import annotations

from pathlib import Path

import pytest

from rlx.cli.main import main
from rlx.core.errors import StoreError
from rlx.core.mirror import (
    FileStoreAdapter,
    HuggingFaceStoreAdapter,
    build_mirror_artifact,
    pull_artifact,
    push_artifact,
)
from rlx.core.sdk import Policy


def test_i02_file_round_trip_preserves_policy_identity(tmp_path: Path) -> None:
    source = Path("examples/eval/demo/rock.rlx").resolve()
    expected = Policy.load(source).digest
    mirror = tmp_path / "mirror"
    pushed = push_artifact(source, mirror.as_uri(), verify=True)
    assert pushed["identity"] == expected
    assert pushed["uri"].endswith(f"#{expected}")

    restored = tmp_path / "restored.rlx"
    result = pull_artifact(pushed["uri"], restored, verify=True)
    assert result["identity"] == expected
    assert Policy.load(restored).digest == expected
    for original in sorted(source.rglob("*")):
        if original.is_file():
            assert (restored / original.relative_to(source)).read_bytes() == original.read_bytes()


def test_file_pull_verify_rejects_mutated_blob(tmp_path: Path) -> None:
    source = Path("examples/eval/demo/rock.rlx").resolve()
    artifact = build_mirror_artifact(source)
    mirror = tmp_path / "mirror"
    uri = FileStoreAdapter().push(artifact, mirror.as_uri(), verify=True)
    entry = artifact.descriptor()["files"][0]
    digest_hex = entry["digest"].removeprefix("sha256:")
    blob = mirror / "objects" / digest_hex[:2] / digest_hex[2:]
    blob.write_bytes(b"mutated")
    with pytest.raises(StoreError, match="digest mismatch"):
        FileStoreAdapter().pull(uri, tmp_path / "bad", verify=True)


def test_hf_adapter_uses_backend_credentials_and_preserves_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    remote: dict[str, bytes] = {}

    class FakeApi:
        def upload_file(self, *, path_or_fileobj, path_in_repo, **kwargs):
            assert kwargs["repo_id"] == "lab/artifacts"
            assert kwargs["repo_type"] == "dataset"
            assert kwargs["revision"] == "main"
            remote[path_in_repo] = path_or_fileobj.read()

    monkeypatch.setattr(HuggingFaceStoreAdapter, "_api", staticmethod(lambda: FakeApi()))

    def fake_download(*, filename, **kwargs):
        assert kwargs["repo_id"] == "lab/artifacts"
        assert kwargs["repo_type"] == "dataset"
        assert kwargs["revision"] == "main"
        path = tmp_path / "downloads" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(remote[filename])
        return str(path)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    source = Path("examples/eval/demo/rock.rlx").resolve()
    artifact = build_mirror_artifact(source)
    adapter = HuggingFaceStoreAdapter()
    uri = adapter.push(
        artifact,
        "hf://datasets/lab/artifacts/rlx?revision=main",
        verify=True,
    )
    assert uri.endswith(f"#{artifact.identity}")
    restored = tmp_path / "hf-restored.rlx"
    result = adapter.pull(uri, restored, verify=True)
    assert result["identity"] == artifact.identity
    assert Policy.load(restored).digest == artifact.identity


def test_hf_push_verify_rejects_remote_blob_mutation(tmp_path: Path, monkeypatch) -> None:
    remote: dict[str, bytes] = {}

    class FakeApi:
        def upload_file(self, *, path_or_fileobj, path_in_repo, **kwargs):
            del kwargs
            remote[path_in_repo] = path_or_fileobj.read()

    monkeypatch.setattr(HuggingFaceStoreAdapter, "_api", staticmethod(lambda: FakeApi()))
    object_downloads = 0

    def fake_download(*, filename, **kwargs):
        nonlocal object_downloads
        del kwargs
        data = remote[filename]
        if "/objects/" in f"/{filename}":
            object_downloads += 1
            data = b"mutated-in-transit"
        path = tmp_path / "downloads" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return str(path)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    artifact = build_mirror_artifact(Path("examples/eval/demo/rock.rlx"))
    with pytest.raises(StoreError, match="push verification failed"):
        HuggingFaceStoreAdapter().push(
            artifact,
            "hf://datasets/lab/artifacts/rlx?revision=main",
            verify=True,
        )
    assert object_downloads >= 1


def test_policy_mirror_rejects_undeclared_files_and_symlinks(tmp_path: Path) -> None:
    import shutil

    source = tmp_path / "policy.rlx"
    shutil.copytree(Path("examples/eval/demo/rock.rlx"), source)
    (source / "undeclared.txt").write_text("not identity-bound", encoding="utf-8")
    with pytest.raises(StoreError, match="outside its content-addressed manifest"):
        build_mirror_artifact(source)

    (source / "undeclared.txt").unlink()
    (source / "link").symlink_to(source / "policy.yaml")
    with pytest.raises(StoreError, match="refusing symlink"):
        build_mirror_artifact(source)


def test_push_pull_cli_verified_round_trip(tmp_path: Path, capsys) -> None:
    import json

    source = Path("examples/eval/demo/rock.rlx").resolve()
    mirror = tmp_path / "mirror"
    assert main(["push", str(source), mirror.as_uri(), "--verify", "--json"]) == 0
    pushed = json.loads(capsys.readouterr().out)
    restored = tmp_path / "cli-restored.rlx"
    assert main(
        ["pull", pushed["uri"], "--out", str(restored), "--verify", "--json"]
    ) == 0
    pulled = json.loads(capsys.readouterr().out)
    assert pulled["identity"] == pushed["identity"]
    assert Policy.load(restored).digest == Policy.load(source).digest
