"""Hugging Face artifact-store adapter."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urldefrag, urlencode, urlparse

from arena.core.errors import StoreError
from arena.core.identity import canonical_json, digest_uri, parse_digest, sha256_bytes
from arena.core.manifests import load_manifest
from arena.core.mirror import (
    MirrorArtifact,
    _artifact_uri,
    _identity_from_uri,
    _simulate_pull,
    _simulate_push,
    _validate_descriptor,
    _write_restored,
)


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


