from __future__ import annotations

import io
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from arena.cli.main import main
from arena.core.errors import StoreError
from arena.core.mirror import (
    FileStoreAdapter,
    HuggingFaceStoreAdapter,
    OCIStoreAdapter,
    _validate_descriptor,
    build_mirror_artifact,
    pull_artifact,
    push_artifact,
)
from arena.core.sdk import Policy


def test_i02_file_round_trip_preserves_policy_identity(tmp_path: Path) -> None:
    source = Path("examples/eval/demo/rock.arena").resolve()
    expected = Policy.load(source).digest
    mirror = tmp_path / "mirror"
    pushed = push_artifact(source, mirror.as_uri(), verify=True)
    assert pushed["identity"] == expected
    assert pushed["uri"].endswith(f"#{expected}")

    restored = tmp_path / "restored.arena"
    result = pull_artifact(pushed["uri"], restored, verify=True)
    assert result["identity"] == expected
    assert Policy.load(restored).digest == expected
    for original in sorted(source.rglob("*")):
        if original.is_file():
            assert (restored / original.relative_to(source)).read_bytes() == original.read_bytes()


def test_file_store_interrupted_publish_leaves_no_pullable_descriptor(
    tmp_path: Path, monkeypatch
) -> None:
    source = Path("examples/eval/demo/rock.arena").resolve()
    artifact = build_mirror_artifact(source)
    mirror = tmp_path / "mirror"
    import arena.core.mirror as mirror_mod

    real_atomic_create = mirror_mod.atomic_create_bytes

    def fail_descriptor(path, data, **kwargs):
        destination = Path(path)
        if destination.parent.name == "artifacts" and destination.suffix == ".json":
            raise OSError("simulated interrupt before descriptor publish")
        return real_atomic_create(path, data, **kwargs)

    monkeypatch.setattr(mirror_mod, "atomic_create_bytes", fail_descriptor)
    with pytest.raises(StoreError, match="conflicting mirror descriptor|simulated interrupt"):
        FileStoreAdapter().push(artifact, mirror.as_uri(), verify=False)

    artifacts_dir = mirror / "artifacts"
    assert not artifacts_dir.exists() or not any(artifacts_dir.glob("*.json"))
    identity_hex = artifact.identity.removeprefix("sha256:")
    assert not (mirror / "artifacts" / f"{identity_hex}.json").exists()
    uri = f"{mirror.as_uri()}#{artifact.identity}"
    with pytest.raises(StoreError, match="mirror descriptor not found"):
        FileStoreAdapter().pull(uri, tmp_path / "should-not-exist", verify=True)


def test_file_round_trip_preserves_object_and_directory_digests(tmp_path: Path) -> None:
    from arena.core.identity import digest_uri, sha256_bytes

    obj = tmp_path / "blob.bin"
    obj.write_bytes(b"digest-stable-object")
    obj_artifact = build_mirror_artifact(obj)
    obj_mirror = tmp_path / "obj-mirror"
    obj_uri = FileStoreAdapter().push(obj_artifact, obj_mirror.as_uri(), verify=True)
    obj_out = tmp_path / "obj-out"
    obj_restored = pull_artifact(obj_uri, obj_out, verify=True)
    assert obj_restored["identity"] == obj_artifact.identity
    assert digest_uri(sha256_bytes((obj_out / "blob.bin").read_bytes())) == obj_artifact.identity

    directory = tmp_path / "tree"
    directory.mkdir()
    (directory / "a.txt").write_text("alpha", encoding="utf-8")
    (directory / "nested").mkdir()
    (directory / "nested" / "b.txt").write_text("beta", encoding="utf-8")
    dir_artifact = build_mirror_artifact(directory)
    assert dir_artifact.kind == "directory"
    dir_mirror = tmp_path / "dir-mirror"
    dir_uri = FileStoreAdapter().push(dir_artifact, dir_mirror.as_uri(), verify=True)
    dir_out = tmp_path / "dir-out"
    dir_restored = pull_artifact(dir_uri, dir_out, verify=True)
    assert dir_restored["identity"] == dir_artifact.identity
    assert dir_restored["identity"] == build_mirror_artifact(dir_out).identity


def test_file_pull_verify_rejects_mutated_blob(tmp_path: Path) -> None:
    source = Path("examples/eval/demo/rock.arena").resolve()
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
            return SimpleNamespace(oid="a" * 40)

    monkeypatch.setattr(HuggingFaceStoreAdapter, "_api", staticmethod(lambda: FakeApi()))

    def fake_download(*, filename, **kwargs):
        assert kwargs["repo_id"] == "lab/artifacts"
        assert kwargs["repo_type"] == "dataset"
        assert kwargs["revision"] == "a" * 40
        path = tmp_path / "downloads" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(remote[filename])
        return str(path)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    source = Path("examples/eval/demo/rock.arena").resolve()
    artifact = build_mirror_artifact(source)
    adapter = HuggingFaceStoreAdapter()
    uri = adapter.push(
        artifact,
        "hf://datasets/lab/artifacts/arena?revision=main",
        verify=True,
    )
    assert uri.endswith(f"#{artifact.identity}")
    assert f"revision={'a' * 40}" in uri
    restored = tmp_path / "hf-restored.arena"
    result = adapter.pull(uri, restored, verify=True)
    assert result["identity"] == artifact.identity
    assert Policy.load(restored).digest == artifact.identity


def test_hf_pull_resolves_movable_revision_once_before_any_download(
    tmp_path: Path, monkeypatch
) -> None:
    remote: dict[str, bytes] = {}
    repo_info_calls = 0

    class FakeApi:
        def upload_file(self, *, path_or_fileobj, path_in_repo, **kwargs):
            del kwargs
            remote[path_in_repo] = path_or_fileobj.read()
            return SimpleNamespace(oid="c" * 40)

        def repo_info(self, **kwargs):
            nonlocal repo_info_calls
            assert kwargs["revision"] == "moving-tag"
            repo_info_calls += 1
            return SimpleNamespace(sha="c" * 40)

    monkeypatch.setattr(HuggingFaceStoreAdapter, "_api", staticmethod(lambda: FakeApi()))
    seen_revisions: list[str] = []

    def fake_download(*, filename, **kwargs):
        seen_revisions.append(kwargs["revision"])
        path = tmp_path / "downloads" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(remote[filename])
        return str(path)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    artifact = build_mirror_artifact(Path("examples/eval/demo/rock.arena"))
    adapter = HuggingFaceStoreAdapter()
    adapter.push(
        artifact,
        "hf://datasets/lab/artifacts/arena?revision=main",
        verify=False,
    )
    movable = (
        "hf://datasets/lab/artifacts/arena?revision=moving-tag"
        f"#{artifact.identity}"
    )
    result = adapter.pull(movable, tmp_path / "restored", verify=False)
    assert result["identity_verified"] is True
    assert repo_info_calls == 1
    assert seen_revisions and set(seen_revisions) == {"c" * 40}


def test_hf_push_verify_rejects_remote_blob_mutation(tmp_path: Path, monkeypatch) -> None:
    remote: dict[str, bytes] = {}

    class FakeApi:
        def upload_file(self, *, path_or_fileobj, path_in_repo, **kwargs):
            del kwargs
            remote[path_in_repo] = path_or_fileobj.read()
            return SimpleNamespace(oid="b" * 40)

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
    artifact = build_mirror_artifact(Path("examples/eval/demo/rock.arena"))
    with pytest.raises(StoreError, match="push verification failed"):
        HuggingFaceStoreAdapter().push(
            artifact,
            "hf://datasets/lab/artifacts/arena?revision=main",
            verify=True,
        )
    assert object_downloads >= 1


def test_policy_mirror_rejects_undeclared_files_and_symlinks(tmp_path: Path) -> None:
    import shutil

    source = tmp_path / "policy.arena"
    shutil.copytree(Path("examples/eval/demo/rock.arena"), source)
    (source / "undeclared.txt").write_text("not identity-bound", encoding="utf-8")
    with pytest.raises(StoreError, match="outside its content-addressed manifest"):
        build_mirror_artifact(source)

    (source / "undeclared.txt").unlink()
    (source / "link").symlink_to(source / "policy.yaml")
    with pytest.raises(StoreError, match="refusing symlink"):
        build_mirror_artifact(source)


def test_mirror_rejects_portable_case_and_unicode_path_collisions() -> None:
    digest = "sha256:" + ("a" * 64)
    descriptor = {
        "schema": "arena.mirror/v1",
        "identity": "sha256:" + ("b" * 64),
        "kind": "directory",
        "files": [
            {"path": "Policy.yaml", "digest": digest, "size": 1},
            {"path": "policy.yaml", "digest": digest, "size": 1},
        ],
    }
    with pytest.raises(StoreError, match="portable mirror path collision"):
        _validate_descriptor(descriptor)

    descriptor["files"] = [
        {"path": "café.txt", "digest": digest, "size": 1},
        {"path": "cafe\u0301.txt", "digest": digest, "size": 1},
    ]
    with pytest.raises(StoreError, match="Unicode NFC"):
        _validate_descriptor(descriptor)


def test_oci_extraction_rejects_expansion_budget_and_links(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("arena.core.mirror.MAX_MIRROR_BYTES", 10)
    oversized = tmp_path / "oversized.tar"
    with tarfile.open(oversized, "w") as archive:
        info = tarfile.TarInfo("mirror/blob")
        info.size = 11
        archive.addfile(info, io.BytesIO(b"x" * 11))
    with pytest.raises(StoreError, match="expanded byte limit"):
        OCIStoreAdapter._extract_archive(oversized, tmp_path / "oversized-out")

    linked = tmp_path / "linked.tar"
    with tarfile.open(linked, "w") as archive:
        info = tarfile.TarInfo("mirror/link")
        info.type = tarfile.SYMTYPE
        info.linkname = "../../escape"
        archive.addfile(info)
    with pytest.raises(StoreError, match="unsafe OCI"):
        OCIStoreAdapter._extract_archive(linked, tmp_path / "linked-out")

def test_push_pull_cli_verified_round_trip(tmp_path: Path, capsys) -> None:
    import json

    source = Path("examples/eval/demo/rock.arena").resolve()
    mirror = tmp_path / "mirror"
    assert main(["push", str(source), mirror.as_uri(), "--verify", "--json"]) == 0
    pushed = json.loads(capsys.readouterr().out)
    restored = tmp_path / "cli-restored.arena"
    assert main(
        ["pull", pushed["uri"], "--out", str(restored), "--verify", "--json"]
    ) == 0
    pulled = json.loads(capsys.readouterr().out)
    assert pulled["identity"] == pushed["identity"]
    assert Policy.load(restored).digest == Policy.load(source).digest
