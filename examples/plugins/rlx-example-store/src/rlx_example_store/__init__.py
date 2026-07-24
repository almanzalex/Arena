"""Example create-only local store plugin for the RLX v1 entry-point contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urldefrag, urlparse


class ExampleStore:
    scheme = "example"

    @staticmethod
    def _file_uri(uri: str) -> str:
        base, fragment = urldefrag(uri)
        parsed = urlparse(base)
        if parsed.scheme != "example" or parsed.netloc not in {"", "localhost"}:
            raise ValueError("example store URI must be example:///absolute/path")
        root = Path(parsed.path)
        if not root.is_absolute():
            raise ValueError("example store path must be absolute")
        translated = root.as_uri()
        return f"{translated}#{fragment}" if fragment else translated

    def push(self, artifact: Any, destination: str, *, verify: bool = False) -> str:
        from rlx.core.mirror import FileStoreAdapter

        result = FileStoreAdapter().push(
            artifact,
            self._file_uri(destination),
            verify=verify,
        )
        return "example:" + result.removeprefix("file:")

    def pull(self, source: str, out: Path | str, *, verify: bool = False) -> dict[str, Any]:
        from rlx.core.mirror import FileStoreAdapter

        return FileStoreAdapter().pull(
            self._file_uri(source),
            out,
            verify=verify,
        )


def register() -> None:
    from rlx.plugins.stores import register_store_adapter

    register_store_adapter("example", ExampleStore())
