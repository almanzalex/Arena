"""Content identity helpers (SHA-256)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_uri(hex_digest: str) -> str:
    return f"sha256:{hex_digest}"


def parse_digest(value: str) -> str:
    if value.startswith("sha256:"):
        return value.split(":", 1)[1]
    return value


def canonical_json(obj: Any) -> bytes:
    """Deterministic JSON encoding for content identity."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def sha256_canonical(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj))
