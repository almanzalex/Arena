"""MLflow run-artifact mirror (preview until live evidence exists)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urldefrag, urlencode, urlparse

from arena.core.errors import StoreError
from arena.core.identity import parse_digest
from arena.core.mirror import (
    MIRROR_SCHEMA,
    MirrorArtifact,
    _read_provider_tree,
    _restore_provider_tree,
    _simulate_pull,
    _simulate_push,
    _verify_provider_blobs,
    _write_provider_tree,
)
from arena.core.store_preview import credentials_required_error, wrap_auth_failure


class MLflowStoreAdapter:
    """MLflow run-artifact adapter with normal tracking-server credentials."""

    scheme = "mlflow"

    @staticmethod
    def _mlflow() -> Any:
        try:
            import mlflow
        except ImportError as exc:
            raise credentials_required_error(
                "mlflow",
                cause="optional extra 'mlflow' is not installed",
                repair=(
                    "Install with: python -m pip install 'arena[mlflow]', configure "
                    "MLFLOW_TRACKING_URI (or `?tracking_uri=`), then "
                    "`arena store qualify policy.arena "
                    "'mlflow://experiment?tracking_uri=…' "
                    "--out docs/qualifications/mlflow/live-qualification.json`. "
                    "For local rehearsal only: append `?simulate=/absolute/path`."
                ),
                context={"extra": "mlflow"},
            ) from exc
        return mlflow

    @staticmethod
    def _tracking(uri: str) -> str | None:
        query = parse_qs(urlparse(urldefrag(uri)[0]).query).get("tracking_uri", [None])[0]
        if query:
            return query
        env = (os.environ.get("MLFLOW_TRACKING_URI") or "").strip()
        return env or None

    @classmethod
    def _require_live_tracking(cls, destination: str) -> str:
        tracking = cls._tracking(destination)
        if not tracking:
            raise credentials_required_error(
                "mlflow",
                cause="no tracking_uri configured for live MLflow qualification",
                repair=(
                    "Set MLFLOW_TRACKING_URI or pass "
                    "`?tracking_uri=https%3A%2F%2Fmlflow.example` on the destination, "
                    "authenticate to that tracking server, then re-run "
                    "`arena store qualify` without `?simulate=`. "
                    "A default local `./mlruns` directory is not live evidence. "
                    "See docs/qualifications/mlflow/README.md."
                ),
            )
        if tracking.startswith("file:"):
            raise credentials_required_error(
                "mlflow",
                cause="file:// tracking_uri is local rehearsal, not live evidence",
                repair=(
                    "Use a remote MLflow tracking URI for live qualification, or "
                    "append `?simulate=/absolute/path` for local protocol rehearsal. "
                    "See docs/qualifications/mlflow/README.md."
                ),
                context={"tracking_uri_scheme": "file"},
            )
        return tracking

    def push(self, artifact: MirrorArtifact, destination: str, *, verify: bool = False) -> str:
        simulated = _simulate_push(artifact, destination, verify=verify)
        if simulated is not None:
            return simulated
        parsed = urlparse(urldefrag(destination)[0])
        if parsed.scheme != "mlflow" or not parsed.netloc:
            raise StoreError("MLflow destination must be mlflow://experiment-name")
        experiment = parsed.netloc
        prefix = parsed.path.strip("/") or "arena"
        tracking = self._require_live_tracking(destination)
        mlflow = self._mlflow()
        try:
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
                    mlflow.set_tags(
                        {"arena.identity": artifact.identity, "arena.schema": MIRROR_SCHEMA}
                    )
                    run_id = run.info.run_id
        except StoreError:
            raise
        except Exception as exc:  # noqa: BLE001
            if _looks_like_auth_failure(str(exc)):
                raise wrap_auth_failure("mlflow", exc, detail=str(exc)) from exc
            raise
        uri = f"mlflow://{run_id}/{artifact_path}?{urlencode({'tracking_uri': tracking})}"
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
        tracking = self._require_live_tracking(source)
        try:
            mlflow.set_tracking_uri(tracking)
            downloaded = mlflow.artifacts.download_artifacts(
                run_id=parsed.netloc,
                artifact_path=parsed.path.strip("/"),
                dst_path=str(root),
                tracking_uri=tracking,
            )
        except StoreError:
            raise
        except Exception as exc:  # noqa: BLE001
            if _looks_like_auth_failure(str(exc)):
                raise wrap_auth_failure("mlflow", exc, detail=str(exc)) from exc
            raise
        return Path(downloaded)

    def pull(self, source: str, out: Path | str, *, verify: bool = False) -> dict[str, Any]:
        simulated = _simulate_pull(source, out, verify=verify)
        if simulated is not None:
            return simulated
        with tempfile.TemporaryDirectory(prefix="arena-mlflow-pull-") as raw:
            tree = self._download(source, Path(raw))
            return _restore_provider_tree(tree, source, out, verify=verify)


def _looks_like_auth_failure(message: str) -> bool:
    text = message.lower()
    needles = (
        "unauthorized",
        "authentication",
        "permission",
        "forbidden",
        "401",
        "403",
        "invalid credentials",
    )
    return any(item in text for item in needles)
