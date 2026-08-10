"""Weights & Biases artifact-store adapter."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urldefrag, urlencode, urlparse

from arena.core.errors import StoreError
from arena.core.identity import parse_digest
from arena.core.mirror import (
    MIRROR_SCHEMA,
    MirrorArtifact,
    _identity_from_uri,
    _read_provider_tree,
    _restore_provider_tree,
    _simulate_pull,
    _simulate_push,
    _verify_provider_blobs,
    _write_provider_tree,
)


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


