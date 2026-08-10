"""MLflow artifact-store adapter."""

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
    _read_provider_tree,
    _restore_provider_tree,
    _simulate_pull,
    _simulate_push,
    _verify_provider_blobs,
    _write_provider_tree,
)


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


