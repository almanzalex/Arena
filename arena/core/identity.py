"""Content identity helpers (SHA-256)."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from arena.core.errors import SchemaError, missing_digest

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


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
    """Return a validated lowercase SHA-256 hex digest.

    Both the canonical ``sha256:<hex>`` URI and a raw 64-character lowercase
    hexadecimal digest are accepted. No other algorithm, abbreviation, uppercase
    spelling, or non-hexadecimal value is interpreted as content identity.
    """

    text = str(value).strip()
    if ":" in text:
        algorithm, text = text.split(":", 1)
        if algorithm != "sha256":
            raise SchemaError(
                (
                    f"unsupported digest algorithm {algorithm!r}; Arena accepts only sha256. "
                    "Recompute with `sha256sum <file>` or use the digest returned by "
                    "`arena push` / policy export."
                ),
                code="DIGEST_INVALID",
                cause=f"digest algorithm {algorithm!r} is not supported",
                repair=(
                    "Use sha256:<64-lowercase-hex> (or a raw 64-character lowercase hex digest). "
                    "Arena will not reinterpret other hash algorithms as content identity."
                ),
                context={"algorithm": algorithm, "value": value},
            )
    if not _SHA256_RE.fullmatch(text):
        raise missing_digest(field="digest", value=value, require_sha256_prefix=False)
    return text


def _validate_finite(value: Any, *, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise SchemaError(f"non-finite number is not valid canonical JSON at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise SchemaError(
                    f"canonical JSON object keys must be strings at {path}, "
                    f"got {type(key).__name__}"
                )
            _validate_finite(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_finite(item, path=f"{path}[{index}]")


def canonical_json(obj: Any) -> bytes:
    """Deterministic JSON encoding for content identity."""
    _validate_finite(obj)
    try:
        return json.dumps(
            obj,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"value cannot be represented as canonical JSON: {exc}") from exc


def sha256_canonical(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj))
