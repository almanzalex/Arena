"""Content-preserving artifact mirroring for file:// and hf:// stores."""

from __future__ import annotations

import io
import shutil
import tarfile
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urldefrag, urlencode, urlparse

from arena.core.errors import StoreError, missing_digest
from arena.core.identity import (
    canonical_json,
    digest_uri,
    parse_digest,
    sha256_bytes,
)
from arena.core.io import atomic_create_bytes, atomic_write_bytes, publish_directory
from arena.core.manifests import load_manifest
from arena.core.sdk import Policy
from arena.core.store import LocalStore

MIRROR_SCHEMA = "arena.mirror/v1"
MAX_MIRROR_FILES = 100_000
MAX_MIRROR_BYTES = 10 * 1024 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 200_000


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
            total_size = 0
            for file in sorted(path.rglob("*")):
                if file.is_symlink():
                    raise StoreError(f"refusing symlink in mirrored artifact: {file}")
                if file.is_file():
                    if len(files) >= MAX_MIRROR_FILES:
                        raise StoreError(
                            f"artifact exceeds {MAX_MIRROR_FILES} file limit"
                        )
                    size = file.stat().st_size
                    total_size += size
                    if total_size > MAX_MIRROR_BYTES:
                        raise StoreError(
                            f"artifact exceeds {MAX_MIRROR_BYTES} byte limit"
                        )
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
    parse_digest(identity)
    if expected is not None and identity != expected:
        raise StoreError(f"mirror identity mismatch: URI requested {expected}, descriptor has {identity}")
    kind = descriptor.get("kind")
    if kind not in {"policy", "directory", "object"}:
        raise StoreError(f"unsupported mirror artifact kind: {kind!r}")
    files = descriptor.get("files")
    if not isinstance(files, list) or not files:
        raise StoreError("mirror descriptor requires files")
    if len(files) > MAX_MIRROR_FILES:
        raise StoreError(f"mirror descriptor exceeds {MAX_MIRROR_FILES} file limit")
    seen: set[str] = set()
    portable_seen: set[str] = set()
    total_size = 0
    for entry in files:
        raw_path = str(entry.get("path", ""))
        if (
            not raw_path
            or raw_path.startswith("/")
            or "\\" in raw_path
            or "\x00" in raw_path
            or any(ord(char) < 32 for char in raw_path)
        ):
            raise StoreError(f"unsafe mirror path: {raw_path!r}")
        if unicodedata.normalize("NFC", raw_path) != raw_path:
            raise StoreError(f"mirror path must use Unicode NFC: {raw_path!r}")
        rel = Path(raw_path)
        if (
            not rel.parts
            or rel.is_absolute()
            or ".." in rel.parts
            or "." in rel.parts
            or len(raw_path.encode("utf-8")) > 1024
            or any(len(part.encode("utf-8")) > 255 for part in rel.parts)
        ):
            raise StoreError(f"unsafe mirror path: {raw_path!r}")
        if raw_path in seen:
            raise StoreError(f"duplicate mirror path: {raw_path}")
        portable = raw_path.casefold()
        if portable in portable_seen:
            raise StoreError(f"portable mirror path collision: {raw_path!r}")
        seen.add(raw_path)
        portable_seen.add(portable)
        parse_digest(str(entry.get("digest", "")))
        size = entry.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise StoreError(f"mirror file size must be an integer >= 0: {raw_path!r}")
        total_size += size
        if total_size > MAX_MIRROR_BYTES:
            raise StoreError(f"mirror descriptor exceeds {MAX_MIRROR_BYTES} byte limit")


def _artifact_uri(destination: str, identity: str) -> str:
    base, _fragment = urldefrag(destination)
    return f"{base}#{identity}"


def _identity_from_uri(uri: str) -> tuple[str, str]:
    base, fragment = urldefrag(uri)
    if not fragment.startswith("sha256:"):
        raise missing_digest(
            field="artifact URI #fragment",
            value=fragment or None,
            require_sha256_prefix=True,
        )
    parse_digest(fragment)
    return base, fragment


def _simulation_root(uri: str) -> Path | None:
    values = parse_qs(urlparse(urldefrag(uri)[0]).query).get("simulate")
    if not values:
        return None
    root = Path(values[0])
    if not root.is_absolute():
        raise StoreError("store ?simulate= path must be absolute")
    return root


def _simulate_push(
    artifact: MirrorArtifact, destination: str, *, verify: bool
) -> str | None:
    root = _simulation_root(destination)
    if root is None:
        return None
    FileStoreAdapter().push(artifact, root.as_uri(), verify=verify)
    return _artifact_uri(destination, artifact.identity)


def _simulate_pull(source: str, out: Path | str, *, verify: bool) -> dict[str, Any] | None:
    root = _simulation_root(source)
    if root is None:
        return None
    _base, identity = _identity_from_uri(source)
    return FileStoreAdapter().pull(
        _artifact_uri(root.as_uri(), identity), out, verify=verify
    )


def _write_provider_tree(artifact: MirrorArtifact, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    descriptor = artifact.descriptor()
    _validate_descriptor(descriptor, artifact.identity)
    for entry in descriptor["files"]:
        digest_hex = parse_digest(entry["digest"])
        target = root / "objects" / digest_hex[:2] / digest_hex[2:]
        atomic_write_bytes(target, artifact.files[entry["path"]])
    # Descriptor last: an interrupted tree is unreachable/incomplete, never a
    # descriptor that references objects that were not fully published.
    atomic_write_bytes(root / "descriptor.json", canonical_json(descriptor) + b"\n")


def _read_provider_tree(root: Path, *, expected: str) -> tuple[dict[str, Any], dict[str, bytes]]:
    descriptors = sorted(root.rglob("descriptor.json"))
    if len(descriptors) != 1:
        raise StoreError(
            f"remote mirror must contain exactly one descriptor.json, found {len(descriptors)}"
        )
    tree = descriptors[0].parent
    descriptor = load_manifest(descriptors[0])
    _validate_descriptor(descriptor, expected)
    blobs: dict[str, bytes] = {}
    for entry in descriptor["files"]:
        digest_hex = parse_digest(entry["digest"])
        blob = tree / "objects" / digest_hex[:2] / digest_hex[2:]
        if not blob.is_file():
            raise StoreError(f"remote mirror object missing: {entry['digest']}")
        blobs[entry["digest"]] = blob.read_bytes()
    return descriptor, blobs


def _verify_provider_blobs(
    descriptor: dict[str, Any], blobs: dict[str, bytes], *, backend: str
) -> None:
    for entry in descriptor["files"]:
        actual = digest_uri(sha256_bytes(blobs[entry["digest"]]))
        if len(blobs[entry["digest"]]) != entry["size"]:
            raise StoreError(
                f"{backend} size verification failed for {entry['path']}: "
                f"expected {entry['size']}, got {len(blobs[entry['digest']])}"
            )
        if actual != entry["digest"]:
            raise StoreError(
                f"{backend} verification failed for {entry['path']}: "
                f"expected {entry['digest']}, got {actual}"
            )


def _restore_provider_tree(
    root: Path,
    source: str,
    out: Path | str,
    *,
    verify: bool,
) -> dict[str, Any]:
    _base, identity = _identity_from_uri(source)
    descriptor, blobs = _read_provider_tree(root, expected=identity)
    return _write_restored(descriptor, blobs.__getitem__, out, verify=verify)


def _write_restored(
    descriptor: dict[str, Any],
    load_blob: Any,
    out: Path | str,
    *,
    verify: bool,
) -> dict[str, Any]:
    _validate_descriptor(descriptor)
    out_path = Path(out)
    if out_path.exists():
        raise StoreError(f"pull output already exists: {out_path}")
    restored_digests: list[dict[str, Any]] = []

    def build(stage: Path) -> None:
        for entry in descriptor["files"]:
            data = load_blob(entry["digest"])
            actual = digest_uri(sha256_bytes(data))
            # Content-addressed bytes are always verified. --verify controls the
            # additional artifact-level identity/conformance check below.
            if actual != entry["digest"]:
                raise StoreError(
                    f"pulled blob digest mismatch for {entry['path']}: "
                    f"expected {entry['digest']}, got {actual}"
                )
            if len(data) != entry["size"]:
                raise StoreError(
                    f"pulled blob size mismatch for {entry['path']}: "
                    f"expected {entry['size']}, got {len(data)}"
                )
            destination = stage / entry["path"]
            atomic_write_bytes(destination, data)
            restored_digests.append(
                {"path": entry["path"], "digest": actual}
            )

    def verify_stage(stage: Path) -> None:
        if descriptor["kind"] == "object":
            if len(descriptor["files"]) != 1:
                raise StoreError("object mirror descriptor must contain exactly one file")
            only_path = stage / descriptor["files"][0]["path"]
            actual_identity = digest_uri(sha256_bytes(only_path.read_bytes()))
        else:
            actual_identity = build_mirror_artifact(stage).identity
        if actual_identity != descriptor["identity"]:
            raise StoreError(
                "restored artifact identity changed: "
                f"expected {descriptor['identity']}, got {actual_identity}"
            )

    publish_directory(
        out_path,
        build,
        verify=verify_stage,
    )
    restored = [
        {**entry, "path": str(out_path / entry["path"])}
        for entry in restored_digests
    ]
    return {
        "identity": descriptor["identity"],
        "kind": descriptor["kind"],
        "out": str(out_path),
        "files": restored,
        "verified": bool(verify),
        "content_verified": True,
        "identity_verified": True,
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
            try:
                atomic_create_bytes(dest, data)
            except Exception as exc:
                raise StoreError(
                    f"refusing conflicting pre-existing mirror object: {dest}"
                ) from exc
        identity_hex = parse_digest(artifact.identity)
        descriptor_path = root / "artifacts" / f"{identity_hex}.json"
        try:
            atomic_create_bytes(
                descriptor_path,
                canonical_json(descriptor) + b"\n",
            )
        except Exception as exc:
            raise StoreError(
                f"refusing conflicting mirror descriptor: {descriptor_path}"
            ) from exc
        uri = _artifact_uri(destination, artifact.identity)
        if verify:
            loaded = load_manifest(descriptor_path)
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
        descriptor = load_manifest(descriptor_path)
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
        raise StoreError("hf URI requires owner/repo, e.g. hf://models/lab/arena-artifacts")
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
                "Hugging Face store requires optional extra 'hf'. "
                "Install with: python -m pip install 'arena[hf]'",
                code="CAPABILITY_MISSING",
                cause="optional extra 'hf' is not installed",
                repair=(
                    "Install the missing extra, then retry: python -m pip install 'arena[hf]'. "
                    "Confirm with `arena doctor --capability hf`."
                ),
                context={"extra": "hf", "capability": "hf"},
            ) from e
        return HfApi()

    @staticmethod
    def _with_revision(uri: str, revision: str) -> str:
        base, fragment = urldefrag(uri)
        parsed = urlparse(base)
        query = parse_qs(parsed.query)
        query["revision"] = [revision]
        pinned = parsed._replace(
            query=urlencode({key: values[-1] for key, values in query.items()})
        ).geturl()
        return f"{pinned}#{fragment}" if fragment else pinned

    def _pin_revision(self, source: str) -> str:
        base, _fragment = urldefrag(source)
        repo_id, repo_type, _prefix, revision = _parse_hf_uri(base)
        if (
            isinstance(revision, str)
            and len(revision) == 40
            and all(c in "0123456789abcdef" for c in revision)
        ):
            return source
        try:
            info = self._api().repo_info(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
            )
            commit = str(info.sha)
        except Exception as exc:
            raise StoreError(
                "could not resolve the Hugging Face revision to an immutable commit",
                code="HF_REVISION_UNRESOLVED",
                repair=(
                    "Check repository access and pass a readable ?revision= branch, "
                    "tag, or commit before retrying."
                ),
            ) from exc
        if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
            raise StoreError(f"Hugging Face returned a malformed commit revision: {commit!r}")
        return self._with_revision(source, commit)

    def push(self, artifact: MirrorArtifact, destination: str, *, verify: bool = False) -> str:
        simulated = _simulate_push(artifact, destination, verify=verify)
        if simulated is not None:
            return simulated
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
                commit_message=f"Mirror Arena object {entry['digest']}",
            )
        committed = api.upload_file(
            path_or_fileobj=io.BytesIO(canonical_json(descriptor) + b"\n"),
            path_in_repo=remote_path(f"artifacts/{parse_digest(artifact.identity)}.json"),
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            commit_message=f"Mirror Arena artifact {artifact.identity}",
        )
        commit = getattr(committed, "oid", None)
        if not isinstance(commit, str) or len(commit) != 40:
            pinned_destination = self._pin_revision(destination)
        else:
            pinned_destination = self._with_revision(destination, commit)
        uri = _artifact_uri(pinned_destination, artifact.identity)
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
            raise StoreError(
                "Hugging Face store requires optional extra 'hf'. "
                "Install with: python -m pip install 'arena[hf]'",
                code="CAPABILITY_MISSING",
                cause="optional extra 'hf' is not installed",
                repair=(
                    "Install the missing extra, then retry: python -m pip install 'arena[hf]'. "
                    "Confirm with `arena doctor --capability hf`."
                ),
                context={"extra": "hf", "capability": "hf"},
            ) from e
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
        descriptor = load_manifest(path)
        _validate_descriptor(descriptor, identity)
        return descriptor

    @staticmethod
    def _download_blob(source: str, digest: str) -> bytes:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise StoreError(
                "Hugging Face store requires optional extra 'hf'. "
                "Install with: python -m pip install 'arena[hf]'",
                code="CAPABILITY_MISSING",
                cause="optional extra 'hf' is not installed",
                repair=(
                    "Install the missing extra, then retry: python -m pip install 'arena[hf]'. "
                    "Confirm with `arena doctor --capability hf`."
                ),
                context={"extra": "hf", "capability": "hf"},
            ) from e
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
        simulated = _simulate_pull(source, out, verify=verify)
        if simulated is not None:
            return simulated
        _identity_from_uri(source)
        pinned_source = self._pin_revision(source)
        descriptor = self._download_descriptor(pinned_source)

        def load_blob(digest: str) -> bytes:
            return self._download_blob(pinned_source, digest)

        return _write_restored(descriptor, load_blob, out, verify=verify)


class OCIStoreAdapter:
    """OCI artifact mirror using the standard ORAS CLI and normal registry auth."""

    scheme = "oci"
    artifact_type = "application/vnd.arena.mirror.v1"
    media_type = "application/vnd.arena.mirror.v1+tar"

    @staticmethod
    def _oras() -> str:
        executable = shutil.which("oras")
        if executable is None:
            raise StoreError(
                "OCI store requires the ORAS CLI. Install from https://oras.land/ "
                "and authenticate with `oras login`, or append ?simulate=/absolute/path."
            )
        return executable

    @staticmethod
    def _target(uri: str, identity: str) -> tuple[str, str]:
        parsed = urlparse(urldefrag(uri)[0])
        if parsed.scheme != "oci" or not parsed.netloc or not parsed.path.strip("/"):
            raise StoreError("OCI URI must be oci://registry/repository")
        repository = f"{parsed.netloc}/{parsed.path.strip('/')}"
        tag = parse_qs(parsed.query).get("tag", [f"sha256-{parse_digest(identity)}"])[0]
        return f"{repository}:{tag}", tag

    @staticmethod
    def _archive(artifact: MirrorArtifact, path: Path) -> None:
        with tempfile.TemporaryDirectory(prefix="arena-oci-tree-") as raw:
            tree = Path(raw) / "mirror"
            _write_provider_tree(artifact, tree)
            with tarfile.open(path, "w") as archive:
                archive.add(tree, arcname="mirror", recursive=True)

    @staticmethod
    def _extract_archive(path: Path, out: Path) -> None:
        with tarfile.open(path, "r") as archive:
            members = archive.getmembers()
            if len(members) > MAX_ARCHIVE_MEMBERS:
                raise StoreError(
                    f"OCI archive exceeds {MAX_ARCHIVE_MEMBERS} member limit"
                )
            total_size = 0
            portable_seen: set[str] = set()
            for member in members:
                relative = Path(member.name)
                normalized = unicodedata.normalize("NFC", member.name)
                portable = normalized.casefold()
                if normalized != member.name or portable in portable_seen:
                    raise StoreError(
                        f"ambiguous OCI mirror archive member: {member.name!r}"
                    )
                portable_seen.add(portable)
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or "\\" in member.name
                    or member.issym()
                    or member.islnk()
                ):
                    raise StoreError(f"unsafe OCI mirror archive member: {member.name!r}")
                if member.isdir():
                    (out / relative).mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    raise StoreError(f"unsupported OCI mirror archive member: {member.name!r}")
                total_size += int(member.size)
                if total_size > MAX_MIRROR_BYTES:
                    raise StoreError(
                        f"OCI archive exceeds {MAX_MIRROR_BYTES} expanded byte limit"
                    )
                target = out / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise StoreError(f"cannot extract OCI mirror member: {member.name!r}")
                with target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)

    def _download_tree(self, source: str) -> tuple[Path, tempfile.TemporaryDirectory[str]]:
        _base, identity = _identity_from_uri(source)
        target, _tag = self._target(source, identity)
        holder = tempfile.TemporaryDirectory(prefix="arena-oci-pull-")
        root = Path(holder.name)
        from arena.core.supervisor import run_supervised

        completed = run_supervised(
            [self._oras(), "pull", target, "--output", str(root)],
            timeout_seconds=600,
        )
        if completed.returncode != 0:
            holder.cleanup()
            raise StoreError(
                f"ORAS pull failed with exit {completed.returncode}: "
                f"{completed.stderr[-2000:]}"
            )
        archives = sorted(root.rglob("mirror.tar"))
        if len(archives) != 1:
            holder.cleanup()
            raise StoreError(f"OCI artifact must contain one mirror.tar, found {len(archives)}")
        extracted = root / "extracted"
        extracted.mkdir()
        self._extract_archive(archives[0], extracted)
        return extracted, holder

    def push(self, artifact: MirrorArtifact, destination: str, *, verify: bool = False) -> str:
        simulated = _simulate_push(artifact, destination, verify=verify)
        if simulated is not None:
            return simulated
        target, tag = self._target(destination, artifact.identity)
        with tempfile.TemporaryDirectory(prefix="arena-oci-push-") as raw:
            root = Path(raw)
            archive = root / "mirror.tar"
            self._archive(artifact, archive)
            from arena.core.supervisor import run_supervised

            completed = run_supervised(
                [
                    self._oras(),
                    "push",
                    target,
                    "--artifact-type",
                    self.artifact_type,
                    f"mirror.tar:{self.media_type}",
                ],
                cwd=root,
                timeout_seconds=600,
            )
            if completed.returncode != 0:
                raise StoreError(
                    f"ORAS push failed with exit {completed.returncode}: "
                    f"{completed.stderr[-2000:]}"
                )
        base = urldefrag(destination)[0]
        parsed = urlparse(base)
        query = parse_qs(parsed.query)
        query["tag"] = [tag]
        encoded = urlencode({key: values[-1] for key, values in query.items()})
        uri = parsed._replace(query=encoded).geturl() + f"#{artifact.identity}"
        if verify:
            tree, holder = self._download_tree(uri)
            try:
                descriptor, blobs = _read_provider_tree(tree, expected=artifact.identity)
                _verify_provider_blobs(descriptor, blobs, backend="OCI push")
            finally:
                holder.cleanup()
        return uri

    def pull(self, source: str, out: Path | str, *, verify: bool = False) -> dict[str, Any]:
        simulated = _simulate_pull(source, out, verify=verify)
        if simulated is not None:
            return simulated
        tree, holder = self._download_tree(source)
        try:
            return _restore_provider_tree(tree, source, out, verify=verify)
        finally:
            holder.cleanup()


class WandBStoreAdapter:
    """Weights & Biases artifact adapter; credentials remain owned by W&B."""

    scheme = "wandb"

    @staticmethod
    def _location(uri: str, identity: str) -> tuple[str, str, str, str]:
        parsed = urlparse(urldefrag(uri)[0])
        parts = [part for part in parsed.path.split("/") if part]
        if parsed.scheme != "wandb" or not parsed.netloc or not parts:
            raise StoreError("W&B URI must be wandb://entity/project[/artifact]")
        name = parts[1] if len(parts) > 1 else f"arena-{parse_digest(identity)[:16]}"
        version = parse_qs(parsed.query).get("version", ["latest"])[0]
        return parsed.netloc, parts[0], name, version

    @staticmethod
    def _wandb() -> Any:
        try:
            import wandb
        except ImportError as exc:
            raise StoreError(
                "W&B store requires optional extra 'wandb'. "
                "Install with: python -m pip install 'arena[wandb]'",
                code="CAPABILITY_MISSING",
                cause="optional extra 'wandb' is not installed",
                repair=(
                    "Install with: python -m pip install 'arena[wandb]', then `wandb login`, "
                    "or append ?simulate=/absolute/path. Confirm with `arena doctor --capability wandb`."
                ),
                context={"extra": "wandb", "capability": "wandb"},
            ) from exc
        return wandb

    def push(self, artifact: MirrorArtifact, destination: str, *, verify: bool = False) -> str:
        simulated = _simulate_push(artifact, destination, verify=verify)
        if simulated is not None:
            return simulated
        entity, project, name, _version = self._location(destination, artifact.identity)
        wandb = self._wandb()
        with tempfile.TemporaryDirectory(prefix="arena-wandb-push-") as raw:
            tree = Path(raw) / "mirror"
            _write_provider_tree(artifact, tree)
            run = wandb.init(entity=entity, project=project, job_type="arena-mirror", reinit=True)
            try:
                remote = wandb.Artifact(
                    name=name,
                    type="arena-artifact",
                    metadata={"identity": artifact.identity, "schema": MIRROR_SCHEMA},
                )
                remote.add_dir(str(tree))
                logged = run.log_artifact(remote)
                logged.wait()
                version = getattr(logged, "version", None) or "latest"
            finally:
                run.finish()
        parsed = urlparse(urldefrag(destination)[0])
        uri = parsed._replace(query=urlencode({"version": version})).geturl()
        uri += f"#{artifact.identity}"
        if verify:
            with tempfile.TemporaryDirectory(prefix="arena-wandb-verify-") as raw:
                downloaded = self._download(uri, Path(raw))
                descriptor, blobs = _read_provider_tree(downloaded, expected=artifact.identity)
                _verify_provider_blobs(descriptor, blobs, backend="W&B push")
        return uri

    def _download(self, source: str, root: Path) -> Path:
        _base, identity = _identity_from_uri(source)
        entity, project, name, version = self._location(source, identity)
        artifact = self._wandb().Api().artifact(
            f"{entity}/{project}/{name}:{version}", type="arena-artifact"
        )
        return Path(artifact.download(root=str(root)))

    def pull(self, source: str, out: Path | str, *, verify: bool = False) -> dict[str, Any]:
        simulated = _simulate_pull(source, out, verify=verify)
        if simulated is not None:
            return simulated
        with tempfile.TemporaryDirectory(prefix="arena-wandb-pull-") as raw:
            tree = self._download(source, Path(raw))
            return _restore_provider_tree(tree, source, out, verify=verify)


class MLflowStoreAdapter:
    """MLflow run-artifact adapter with normal tracking-server credentials."""

    scheme = "mlflow"

    @staticmethod
    def _mlflow() -> Any:
        try:
            import mlflow
        except ImportError as exc:
            raise StoreError(
                "MLflow store requires optional extra 'mlflow'. "
                "Install with: python -m pip install 'arena[mlflow]'",
                code="CAPABILITY_MISSING",
                cause="optional extra 'mlflow' is not installed",
                repair=(
                    "Install with: python -m pip install 'arena[mlflow]', "
                    "or append ?simulate=/absolute/path. Confirm with `arena doctor --capability mlflow`."
                ),
                context={"extra": "mlflow", "capability": "mlflow"},
            ) from exc
        return mlflow

    @staticmethod
    def _tracking(uri: str) -> str | None:
        return parse_qs(urlparse(urldefrag(uri)[0]).query).get("tracking_uri", [None])[0]

    def push(self, artifact: MirrorArtifact, destination: str, *, verify: bool = False) -> str:
        simulated = _simulate_push(artifact, destination, verify=verify)
        if simulated is not None:
            return simulated
        parsed = urlparse(urldefrag(destination)[0])
        if parsed.scheme != "mlflow" or not parsed.netloc:
            raise StoreError("MLflow destination must be mlflow://experiment-name")
        experiment = parsed.netloc
        prefix = parsed.path.strip("/") or "arena"
        tracking = self._tracking(destination)
        mlflow = self._mlflow()
        if tracking:
            mlflow.set_tracking_uri(tracking)
        experiment_record = mlflow.set_experiment(experiment)
        with tempfile.TemporaryDirectory(prefix="arena-mlflow-push-") as raw:
            tree = Path(raw) / "mirror"
            _write_provider_tree(artifact, tree)
            with mlflow.start_run(
                experiment_id=experiment_record.experiment_id,
                run_name=f"arena-{parse_digest(artifact.identity)[:12]}",
            ) as run:
                artifact_path = f"{prefix}/{parse_digest(artifact.identity)}"
                mlflow.log_artifacts(str(tree), artifact_path=artifact_path)
                mlflow.set_tags({"arena.identity": artifact.identity, "arena.schema": MIRROR_SCHEMA})
                run_id = run.info.run_id
        query = urlencode({"tracking_uri": tracking}) if tracking else ""
        uri = f"mlflow://{run_id}/{artifact_path}"
        if query:
            uri += f"?{query}"
        uri += f"#{artifact.identity}"
        if verify:
            with tempfile.TemporaryDirectory(prefix="arena-mlflow-verify-") as raw:
                tree = self._download(uri, Path(raw))
                descriptor, blobs = _read_provider_tree(tree, expected=artifact.identity)
                _verify_provider_blobs(descriptor, blobs, backend="MLflow push")
        return uri

    def _download(self, source: str, root: Path) -> Path:
        parsed = urlparse(urldefrag(source)[0])
        if parsed.scheme != "mlflow" or not parsed.netloc or not parsed.path.strip("/"):
            raise StoreError("MLflow artifact URI must include run id and artifact path")
        mlflow = self._mlflow()
        tracking = self._tracking(source)
        if tracking:
            mlflow.set_tracking_uri(tracking)
        downloaded = mlflow.artifacts.download_artifacts(
            run_id=parsed.netloc,
            artifact_path=parsed.path.strip("/"),
            dst_path=str(root),
            tracking_uri=tracking,
        )
        return Path(downloaded)

    def pull(self, source: str, out: Path | str, *, verify: bool = False) -> dict[str, Any]:
        simulated = _simulate_pull(source, out, verify=verify)
        if simulated is not None:
            return simulated
        with tempfile.TemporaryDirectory(prefix="arena-mlflow-pull-") as raw:
            tree = self._download(source, Path(raw))
            return _restore_provider_tree(tree, source, out, verify=verify)


def push_artifact(source: Path | str, destination: str, *, verify: bool = False) -> dict[str, Any]:
    from arena.core.registry import EXTERNAL_STORES, ensure_plugins_loaded

    artifact = build_mirror_artifact(source)
    scheme = urlparse(destination).scheme
    ensure_plugins_loaded()
    uri = EXTERNAL_STORES.get(scheme).push(artifact, destination, verify=verify)
    return {"identity": artifact.identity, "kind": artifact.kind, "uri": uri, "verified": verify}


def pull_artifact(source: str, out: Path | str, *, verify: bool = False) -> dict[str, Any]:
    from arena.core.registry import EXTERNAL_STORES, ensure_plugins_loaded

    scheme = urlparse(source).scheme
    ensure_plugins_loaded()
    return EXTERNAL_STORES.get(scheme).pull(source, out, verify=verify)
