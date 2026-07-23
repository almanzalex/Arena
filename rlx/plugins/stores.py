"""External artifact-store registry.

Backends mirror bytes addressed by RLX identities. They never assign or rewrite
artifact digests.
"""

from __future__ import annotations

from typing import Any, Protocol

from rlx.core.registry import EXTERNAL_STORES


class ExternalStoreAdapter(Protocol):
    scheme: str

    def push(self, artifact: Any, destination: str, *, verify: bool = False) -> str: ...

    def pull(self, source: str, out: Any, *, verify: bool = False) -> Any: ...


def register_store_adapter(
    scheme: str, adapter: ExternalStoreAdapter, *, replace: bool = False
) -> ExternalStoreAdapter:
    return EXTERNAL_STORES.register(scheme, adapter, replace=replace)


def register_builtins() -> None:
    from rlx.core.mirror import (
        FileStoreAdapter,
        HuggingFaceStoreAdapter,
        MLflowStoreAdapter,
        OCIStoreAdapter,
        WandBStoreAdapter,
    )

    register_store_adapter("file", FileStoreAdapter(), replace=True)
    register_store_adapter("hf", HuggingFaceStoreAdapter(), replace=True)
    register_store_adapter("oci", OCIStoreAdapter(), replace=True)
    register_store_adapter("wandb", WandBStoreAdapter(), replace=True)
    register_store_adapter("mlflow", MLflowStoreAdapter(), replace=True)
