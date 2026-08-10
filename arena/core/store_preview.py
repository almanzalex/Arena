"""Shared helpers for preview credentialed stores (OCI, W&B, MLflow).

Simulation (`?simulate=/absolute/path`) rehearses the mirror protocol only.
It never satisfies a live or stable claim.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urldefrag, urlparse

from arena.core.errors import StoreError

PREVIEW_STORES = frozenset({"oci", "wandb", "mlflow"})

_QUAL_DOC = {
    "oci": "docs/qualifications/oci/README.md",
    "wandb": "docs/qualifications/wandb/README.md",
    "mlflow": "docs/qualifications/mlflow/README.md",
}


def is_simulation_uri(uri: str) -> bool:
    values = parse_qs(urlparse(urldefrag(uri)[0]).query).get("simulate")
    return bool(values)


def simulation_root(uri: str) -> Path | None:
    values = parse_qs(urlparse(urldefrag(uri)[0]).query).get("simulate")
    if not values:
        return None
    root = Path(values[0])
    if not root.is_absolute():
        raise StoreError("store ?simulate= path must be absolute")
    return root


def mode_and_live_claim(destination: str, immutable_uri: str) -> tuple[str, bool]:
    """Return (mode, counts_as_live_evidence). Simulation never counts as live."""
    if is_simulation_uri(destination) or is_simulation_uri(immutable_uri):
        return "simulation", False
    return "live", True


def credentials_required_error(
    backend: str,
    *,
    cause: str,
    repair: str,
    context: dict[str, Any] | None = None,
) -> StoreError:
    doc = _QUAL_DOC.get(backend, "docs/qualifications/")
    return StoreError(
        (
            f"{backend} live store qualification requires credentials/tooling "
            f"({cause}). Simulation (`?simulate=/absolute/path`) never counts as "
            f"live evidence. See {doc}."
        ),
        code="STORE_CREDENTIALS_REQUIRED",
        cause=cause,
        repair=repair,
        context={"capability": backend, "counts_as_live_evidence": False, **(context or {})},
    )


def wrap_auth_failure(backend: str, exc: BaseException, *, detail: str) -> StoreError:
    doc = _QUAL_DOC.get(backend, "docs/qualifications/")
    return StoreError(
        (
            f"{backend} live push/pull failed authentication or authorization: {detail}. "
            f"Simulation never counts as live. See {doc}."
        ),
        code="STORE_CREDENTIALS_REQUIRED",
        cause=str(exc) or type(exc).__name__,
        repair=(
            f"Authenticate with the backend's normal login, then re-run "
            f"`arena store qualify …` without `?simulate=`, or rehearse locally with "
            f"`?simulate=/absolute/path`. Docs: {doc}."
        ),
        context={"capability": backend, "counts_as_live_evidence": False},
    )
