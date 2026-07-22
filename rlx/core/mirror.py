"""Content-preserving artifact mirroring for file:// and hf:// stores."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urldefrag, urlparse

from rlx.core.errors import StoreError
from rlx.core.identity import (
    canonical_json,
    digest_uri,
    parse_digest,
    sha256_bytes,
)
from rlx.core.sdk import Policy
from rlx.core.store import LocalStore

MIRROR_SCHEMA = "rlx.mirror/v1"


@dataclass(frozen=True)
class MirrorArtifact:
    identity: str
    kind: str
    files: dict[str, bytes]

    def descriptor(self) -> dict[str, Any]:
        entries = [
            {
                "path": path,
                "digest": digest_uri(sha256_bytes(data)),
                "size": len(data),
            }
            for path, data in sorted(self.files.items())
        ]
        return {
            "schema": MIRROR_SCHEMA,
            "identity": self.identity,
            "kind": self.kind,
            "files": entries,
        }


def build_mirror_artifact(source: Path | str) -> MirrorArtifact:
    path = Path(source)
    if path.exists():
        if path.is_dir():
            files: dict[str, bytes] = {}
            for file in sorted(path.rglob("*")):
                if file.is_symlink():
                    raise StoreError(f"refusing symlink in mirrored artifact: {file}")
                if file.is_file():
                    files[str(file.relative_to(path))] = file.read_bytes()
            if not files:
                raise StoreError(f"cannot mirror empty directory: {path}")
            if "policy.yaml" in files:
                policy = Policy.load(path)
                identity = policy.digest
                kind = "policy"
                declared = {"policy.yaml"}
                for key, entry in policy.manifest.get("payloads", {}).items():
                    if not isinstance(entry, dict) or not entry.get("path") or not entry.get("digest"):
                        raise StoreError(
                            f"policy payload {key!r} requires path and digest before mirroring"
                        )
                    rel = Path(str(entry["path"]))
                    if rel.is_absolute() or ".." in rel.parts:
                        raise StoreError(f"unsafe policy payload path: {entry['path']!r}")
                    rel_text = str(rel)
                    if rel_text not in files:
                        raise StoreError(f"policy payload {key!r} missing on disk: {rel_text}")
                    actual = digest_uri(sha256_bytes(files[rel_text]))
                    if actual != entry["digest"]:
                        raise StoreError(
                            f"policy payload {key!r} integrity check failed: "
                            f"declared {entry['digest']}, actual {actual}"
                        )
                    declared.add(rel_text)
                if "DIGEST" in files:
                    declared.add("DIGEST")
                    recorded = files["DIGEST"].decode("utf-8").strip()
                    if recorded != identity:
                        raise StoreError(
                            f"policy DIGEST is stale: recorded {recorded!r}, actual {identity}"
                        )
                extras = sorted(set(files) - declared)
                if extras:
                    raise StoreError(
                        "policy bundle contains files outside its content-addressed manifest: "
                        + ", ".join(extras)
                    )
            else:
                tree = [
                    {"path": name, "digest": digest_uri(sha256_bytes(data))}
                    for name, data in sorted(files.items())
                ]
                identity = digest_uri(sha256_bytes(canonical_json(tree)))
                kind = "directory"
            return MirrorArtifact(identity=identity, kind=kind, files=files)
        data = path.read_bytes()
        identity = digest_uri(sha256_bytes(data))
        return MirrorArtifact(identity=identity, kind="object", files={path.name: data})

    store = LocalStore.find()
    text = str(source)
    digest = text if text.startswith("sha256:") else store.get_ref(text)
    data = store.get_bytes(digest, verify=True)
    return MirrorArtifact(identity=digest_uri(parse_digest(digest)), kind="object", files={"object": data})


def _validate_descriptor(descriptor: dict[str, Any], expected: str | None = None) -> None:
    if descriptor.get("schema") != MIRROR_SCHEMA:
        raise StoreError(f"unsupported mirror descriptor schema: {descriptor.get('schema')!r}")
    identity = descriptor.get("identity")
    if not isinstance(identity, str) or not identity.startswith("sha256:"):
        raise StoreError("mirror descriptor identity must be sha256:…")
    if expected is not None and identity != expected:
        raise StoreError(f"mirror identity mismatch: URI requested {expected}, descriptor has {identity}")
    files = descriptor.get("files")
    if not isinstance(files, list) or not files:
        raise StoreError("mirror descriptor requires files")
    seen: set[str] = set()
    for entry in files:
        rel = Path(str(entry.get("path", "")))
        if not rel.parts or rel.is_absolute() or ".." in rel.parts:
            raise StoreError(f"unsafe mirror path: {entry.get('path')!r}")
        if str(rel) in seen:
            raise StoreError(f"duplicate mirror path: {rel}")
        seen.add(str(rel))
        parse_digest(str(entry.get("digest", "")))


def _artifact_uri(destination: str, identity: str) -> str:
    base, _fragment = urldefrag(destination)
    return f"{base}#{identity}"


def _identity_from_uri(uri: str) -> tuple[str, str]:
    base, fragment = urldefrag(uri)
    if not fragment.startswith("sha256:"):
        raise StoreError("artifact URI must include #sha256:… identity returned by `rlx push`")
    parse_digest(fragment)
    return base, fragment


def _write_restored(
    descriptor: dict[str, Any],
    load_blob: Any,
    out: Path | str,
    *,
    verify: bool,
) -> dict[str, Any]:
    _validate_descriptor(descriptor)
    out_path = Path(out)
    if out_path.exists() and (not out_path.is_dir() or any(out_path.iterdir())):
        raise StoreError(f"pull output must not already contain files: {out_path}")
    out_path.mkdir(parents=True, exist_ok=True)
    restored: list[dict[str, Any]] = []
    for entry in descriptor["files"]:
        data = load_blob(entry["digest"])
        actual = digest_uri(sha256_bytes(data))
        if verify and actual != entry["digest"]:
            raise StoreError(
                f"pulled blob digest mismatch for {entry['path']}: "
                f"expected {entry['digest']}, got {actual}"
            )
        dest = out_path / entry["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        restored.append({"path": str(dest), "digest": actual})
    if verify and descriptor["kind"] == "policy":
        actual_identity = Policy.load(out_path).digest
        if actual_identity != descriptor["identity"]:
            raise StoreError(
                "restored policy identity changed: "
                f"expected {descriptor['identity']}, got {actual_identity}"
            )
    return {
        "identity": descriptor["identity"],
        "kind": descriptor["kind"],
        "out": str(out_path),
        "files": restored,
        "verified": bool(verify),
    }


class FileStoreAdapter:
    scheme = "file"

    @staticmethod
    def _root(uri: str) -> Path:
        parsed = urlparse(uri)
        if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
            raise StoreError("file store URI must be file:///absolute/path")
        root = Path(parsed.path)
        if not root.is_absolute():
            raise StoreError("file store URI path must be absolute")
        return root

    def push(self, artifact: MirrorArtifact, destination: str, *, verify: bool = False) -> str:
        root = self._root(urldefrag(destination)[0])
        descriptor = artifact.descriptor()
        _validate_descriptor(descriptor, artifact.identity)
        for entry in descriptor["files"]:
            data = artifact.files[entry["path"]]
            digest_hex = parse_digest(entry["digest"])
            dest = root / "objects" / digest_hex[:2] / digest_hex[2:]
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and sha256_bytes(dest.read_bytes()) != digest_hex:
                raise StoreError(f"refusing corrupt pre-existing mirror object: {dest}")
            if not dest.exists():
                dest.write_bytes(data)
        identity_hex = parse_digest(artifact.identity)
        descriptor_path = root / "artifacts" / f"{identity_hex}.json"
        descriptor_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor_path.write_bytes(canonical_json(descriptor) + b"\n")
        uri = _artifact_uri(destination, artifact.identity)
        if verify:
            loaded = json.loads(descriptor_path.read_text(encoding="utf-8"))
            _validate_descriptor(loaded, artifact.identity)
            for entry in loaded["files"]:
                blob = root / "objects" / parse_digest(entry["digest"])[:2] / parse_digest(entry["digest"])[2:]
                if digest_uri(sha256_bytes(blob.read_bytes())) != entry["digest"]:
                    raise StoreError(f"push verification failed for {entry['path']}")
        return uri

    def pull(self, source: str, out: Path | str, *, verify: bool = False) -> dict[str, Any]:
        base, identity = _identity_from_uri(source)
        root = self._root(base)
        descriptor_path = root / "artifacts" / f"{parse_digest(identity)}.json"
        if not descriptor_path.exists():
            raise StoreError(f"mirror descriptor not found: {source}")
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
        _validate_descriptor(descriptor, identity)

        def load_blob(digest: str) -> bytes:
            digest_hex = parse_digest(digest)
            path = root / "objects" / digest_hex[:2] / digest_hex[2:]
            if not path.exists():
                raise StoreError(f"mirror object missing: {digest}")
            return path.read_bytes()

        return _write_restored(descriptor, load_blob, out, verify=verify)


def _parse_hf_uri(uri: str) -> tuple[str, str, str, str | None]:
    parsed = urlparse(uri)
    if parsed.scheme != "hf":
        raise StoreError("Hugging Face store URI must start hf://")
    parts = [parsed.netloc, *[part for part in parsed.path.split("/") if part]]
    repo_type = "model"
    if parts and parts[0] in {"models", "datasets", "spaces"}:
        repo_type = {"models": "model", "datasets": "dataset", "spaces": "space"}[parts.pop(0)]
    if len(parts) < 2:
        raise StoreError("hf URI requires owner/repo, e.g. hf://models/lab/rlx-artifacts")
    repo_id = "/".join(parts[:2])
    prefix = "/".join(parts[2:]).strip("/")
    revision = parse_qs(parsed.query).get("revision", [None])[0]
    return repo_id, repo_type, prefix, revision


class HuggingFaceStoreAdapter:
    scheme = "hf"

    @staticmethod
    def _api() -> Any:
        try:
            from huggingface_hub import HfApi
        except ImportError as e:
            raise StoreError(
                "Hugging Face store adapter is optional. Install with: pip install 'rlx[hf]'"
            ) from e
        return HfApi()

    def push(self, artifact: MirrorArtifact, destination: str, *, verify: bool = False) -> str:
        base = urldefrag(destination)[0]
        repo_id, repo_type, prefix, revision = _parse_hf_uri(base)
        api = self._api()
        descriptor = artifact.descriptor()
        _validate_descriptor(descriptor, artifact.identity)

        def remote_path(path: str) -> str:
            return "/".join(part for part in (prefix, path) if part)

        for entry in descriptor["files"]:
            digest_hex = parse_digest(entry["digest"])
            api.upload_file(
                path_or_fileobj=io.BytesIO(artifact.files[entry["path"]]),
                path_in_repo=remote_path(f"objects/{digest_hex[:2]}/{digest_hex[2:]}"),
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                commit_message=f"Mirror RLX object {entry['digest']}",
            )
        api.upload_file(
            path_or_fileobj=io.BytesIO(canonical_json(descriptor) + b"\n"),
            path_in_repo=remote_path(f"artifacts/{parse_digest(artifact.identity)}.json"),
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            commit_message=f"Mirror RLX artifact {artifact.identity}",
        )
        uri = _artifact_uri(destination, artifact.identity)
        if verify:
            loaded = self._download_descriptor(uri)
            _validate_descriptor(loaded, artifact.identity)
            for entry in loaded["files"]:
                actual = digest_uri(sha256_bytes(self._download_blob(uri, entry["digest"])))
                if actual != entry["digest"]:
                    raise StoreError(
                        f"push verification failed for {entry['path']}: "
                        f"expected {entry['digest']}, got {actual}"
                    )
        return uri

    def _download_descriptor(self, source: str) -> dict[str, Any]:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise StoreError("install the HF adapter with: pip install 'rlx[hf]'") from e
        base, identity = _identity_from_uri(source)
        repo_id, repo_type, prefix, revision = _parse_hf_uri(base)
        filename = "/".join(
            part for part in (prefix, f"artifacts/{parse_digest(identity)}.json") if part
        )
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            revision=revision,
        )
        descriptor = json.loads(Path(path).read_text(encoding="utf-8"))
        _validate_descriptor(descriptor, identity)
        return descriptor

    @staticmethod
    def _download_blob(source: str, digest: str) -> bytes:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise StoreError("install the HF adapter with: pip install 'rlx[hf]'") from e
        base, _identity = _identity_from_uri(source)
        repo_id, repo_type, prefix, revision = _parse_hf_uri(base)
        digest_hex = parse_digest(digest)
        filename = "/".join(
            part for part in (prefix, f"objects/{digest_hex[:2]}/{digest_hex[2:]}") if part
        )
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            revision=revision,
        )
        return Path(path).read_bytes()

    def pull(self, source: str, out: Path | str, *, verify: bool = False) -> dict[str, Any]:
        descriptor = self._download_descriptor(source)
        def load_blob(digest: str) -> bytes:
            return self._download_blob(source, digest)

        return _write_restored(descriptor, load_blob, out, verify=verify)


def push_artifact(source: Path | str, destination: str, *, verify: bool = False) -> dict[str, Any]:
    from rlx.core.registry import EXTERNAL_STORES, ensure_plugins_loaded

    artifact = build_mirror_artifact(source)
    scheme = urlparse(destination).scheme
    ensure_plugins_loaded()
    uri = EXTERNAL_STORES.get(scheme).push(artifact, destination, verify=verify)
    return {"identity": artifact.identity, "kind": artifact.kind, "uri": uri, "verified": verify}


def pull_artifact(source: str, out: Path | str, *, verify: bool = False) -> dict[str, Any]:
    from rlx.core.registry import EXTERNAL_STORES, ensure_plugins_loaded

    scheme = urlparse(source).scheme
    ensure_plugins_loaded()
    return EXTERNAL_STORES.get(scheme).pull(source, out, verify=verify)
