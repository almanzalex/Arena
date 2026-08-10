"""OCI artifact-store adapter."""

from __future__ import annotations

import shutil
import tarfile
import tempfile
import unicodedata
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urldefrag, urlencode, urlparse

import arena.core.mirror as mirror
from arena.core.errors import StoreError
from arena.core.identity import parse_digest
from arena.core.mirror import (
    MirrorArtifact,
    _identity_from_uri,
    _read_provider_tree,
    _restore_provider_tree,
    _simulate_pull,
    _simulate_push,
    _verify_provider_blobs,
    _write_provider_tree,
)


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
            if len(members) > mirror.MAX_ARCHIVE_MEMBERS:
                raise StoreError(
                    f"OCI archive exceeds {mirror.MAX_ARCHIVE_MEMBERS} member limit"
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
                if total_size > mirror.MAX_MIRROR_BYTES:
                    raise StoreError(
                        f"OCI archive exceeds {mirror.MAX_MIRROR_BYTES} expanded byte limit"
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


