"""Weights & Biases artifact mirror (preview until live evidence exists)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urldefrag, urlencode, urlparse

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
from arena.core.store_preview import credentials_required_error, wrap_auth_failure


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
        from urllib.parse import parse_qs

        version = parse_qs(parsed.query).get("version", ["latest"])[0]
        return parsed.netloc, parts[0], name, version

    @staticmethod
    def _wandb() -> Any:
        try:
            import wandb
        except ImportError as exc:
            raise credentials_required_error(
                "wandb",
                cause="optional extra 'wandb' is not installed",
                repair=(
                    "Install with: python -m pip install 'arena[wandb]', then "
                    "`wandb login` (or set WANDB_API_KEY). Qualify live with "
                    "`arena store qualify policy.arena 'wandb://entity/project/artifact' "
                    "--out docs/qualifications/wandb/live-qualification.json`. "
                    "For local rehearsal only: append `?simulate=/absolute/path`."
                ),
                context={"extra": "wandb"},
            ) from exc
        return wandb

    @classmethod
    def _require_live_credentials(cls) -> Any:
        wandb = cls._wandb()
        api_key = (os.environ.get("WANDB_API_KEY") or "").strip()
        if api_key:
            return wandb
        # Respect an existing wandb login without reading the secret into Arena state.
        settings_key = getattr(getattr(wandb, "api", None), "api_key", None)
        if callable(settings_key):
            try:
                settings_key = settings_key()
            except Exception:  # noqa: BLE001 - treat probe failure as missing creds
                settings_key = None
        if not settings_key:
            try:
                settings_key = getattr(wandb.Api(), "api_key", None)
            except Exception as exc:  # noqa: BLE001
                raise credentials_required_error(
                    "wandb",
                    cause="W&B API credentials are not configured",
                    repair=(
                        "Run `wandb login` or export WANDB_API_KEY, then re-run "
                        "`arena store qualify` without `?simulate=`. "
                        "Simulation never counts as live. See "
                        "docs/qualifications/wandb/README.md."
                    ),
                ) from exc
        if not settings_key:
            raise credentials_required_error(
                "wandb",
                cause="W&B API credentials are not configured",
                repair=(
                    "Run `wandb login` or export WANDB_API_KEY, then re-run "
                    "`arena store qualify` without `?simulate=`. "
                    "Simulation never counts as live. See "
                    "docs/qualifications/wandb/README.md."
                ),
            )
        return wandb

    def push(self, artifact: MirrorArtifact, destination: str, *, verify: bool = False) -> str:
        simulated = _simulate_push(artifact, destination, verify=verify)
        if simulated is not None:
            return simulated
        entity, project, name, _version = self._location(destination, artifact.identity)
        wandb = self._require_live_credentials()
        with tempfile.TemporaryDirectory(prefix="arena-wandb-push-") as raw:
            tree = Path(raw) / "mirror"
            _write_provider_tree(artifact, tree)
            try:
                run = wandb.init(
                    entity=entity, project=project, job_type="arena-mirror", reinit=True
                )
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
            except Exception as exc:  # noqa: BLE001
                if _looks_like_auth_failure(str(exc)):
                    raise wrap_auth_failure("wandb", exc, detail=str(exc)) from exc
                raise
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
        wandb = self._require_live_credentials()
        try:
            artifact = wandb.Api().artifact(
                f"{entity}/{project}/{name}:{version}", type="arena-artifact"
            )
        except Exception as exc:  # noqa: BLE001
            if _looks_like_auth_failure(str(exc)):
                raise wrap_auth_failure("wandb", exc, detail=str(exc)) from exc
            raise
        return Path(artifact.download(root=str(root)))

    def pull(self, source: str, out: Path | str, *, verify: bool = False) -> dict[str, Any]:
        simulated = _simulate_pull(source, out, verify=verify)
        if simulated is not None:
            return simulated
        with tempfile.TemporaryDirectory(prefix="arena-wandb-pull-") as raw:
            tree = self._download(source, Path(raw))
            return _restore_provider_tree(tree, source, out, verify=verify)


def _looks_like_auth_failure(message: str) -> bool:
    text = message.lower()
    needles = (
        "unauthorized",
        "permission",
        "forbidden",
        "not logged in",
        "api key",
        "authentication",
        "401",
        "403",
    )
    return any(item in text for item in needles)
