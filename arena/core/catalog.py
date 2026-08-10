"""Local-only catalog listing for file:// mirror directories.

This is intentionally not a hosted catalog, account service, or control plane.
It lists ``arena.mirror/v1`` descriptors already present under a filesystem
mirror root (``artifacts/*.json``), foreshadowing a future hosted index shape
without network calls or Arena-operated identity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from arena.core.errors import StoreError
from arena.core.identity import parse_digest
from arena.core.manifests import load_manifest
from arena.core.mirror import MIRROR_SCHEMA, _validate_descriptor

CATALOG_LIST_SCHEMA = "arena.catalog-list/v1"
CATALOG_MODE = "local-file-stub"


def _root_from_source(source: str | Path) -> Path:
    text = str(source).strip()
    if not text:
        raise StoreError(
            "catalog local requires a directory path or file:///absolute/path URI",
            repair="Pass a filesystem mirror root, e.g. /tmp/mirror or file:///tmp/mirror.",
        )
    if "://" in text:
        parsed = urlparse(text)
        if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
            raise StoreError(
                "catalog local only accepts file:///absolute/path URIs "
                "(no hosted or remote catalog backends)",
                repair=(
                    "Use a local directory or file:// URI. Hosted accounts/catalog/"
                    "control plane are deferred; see rfcs/012-hosted-control-plane.md."
                ),
                context={"source": text, "scheme": parsed.scheme or None},
            )
        root = Path(parsed.path)
        if not root.is_absolute():
            raise StoreError("file store URI path must be absolute")
        return root
    return Path(text).expanduser().resolve()


def list_local_catalog(source: str | Path) -> dict[str, Any]:
    """List mirrored artifacts from a local file:// mirror directory.

    Expects the layout written by ``FileStoreAdapter.push``:
    ``<root>/artifacts/<identity-hex>.json`` plus content-addressed objects.
    """
    root = _root_from_source(source)
    if not root.exists():
        raise StoreError(
            f"catalog root does not exist: {root}",
            repair="Create the directory and push artifacts with "
            "`arena push <artifact> file:///…`, then retry.",
            context={"root": str(root)},
        )
    if not root.is_dir():
        raise StoreError(
            f"catalog root must be a directory: {root}",
            context={"root": str(root)},
        )

    artifacts_dir = root / "artifacts"
    entries: list[dict[str, Any]] = []
    warnings: list[str] = []

    if artifacts_dir.is_dir():
        for path in sorted(artifacts_dir.glob("*.json")):
            if not path.is_file():
                continue
            try:
                descriptor = load_manifest(path)
                _validate_descriptor(descriptor)
            except Exception as exc:  # noqa: BLE001 — surface per-file, keep listing
                warnings.append(f"skipping invalid descriptor {path.name}: {exc}")
                continue
            if descriptor.get("schema") != MIRROR_SCHEMA:
                warnings.append(
                    f"skipping {path.name}: unsupported schema {descriptor.get('schema')!r}"
                )
                continue
            identity = str(descriptor["identity"])
            identity_hex = parse_digest(identity)
            expected_name = f"{identity_hex}.json"
            if path.name != expected_name:
                warnings.append(
                    f"skipping {path.name}: filename does not match identity {identity}"
                )
                continue
            files = descriptor.get("files") or []
            entries.append(
                {
                    "identity": identity,
                    "kind": descriptor["kind"],
                    "uri": f"{root.as_uri()}#{identity}",
                    "file_count": len(files),
                    "bytes": sum(int(entry.get("size", 0)) for entry in files),
                    "descriptor": str(path),
                }
            )

    return {
        "schema": CATALOG_LIST_SCHEMA,
        "mode": CATALOG_MODE,
        "hosted": False,
        "root": str(root),
        "uri": root.as_uri(),
        "count": len(entries),
        "artifacts": entries,
        "warnings": warnings,
        "note": (
            "Local file:// stub only — not a hosted Arena catalog, account service, "
            "or control plane. See rfcs/012-hosted-control-plane.md."
        ),
    }


def format_catalog_human(report: dict[str, Any]) -> str:
    """Render a local catalog list for terminals."""
    lines = [
        f"arena catalog local ({report.get('mode', CATALOG_MODE)})",
        f"root: {report.get('root')}",
        f"hosted: {report.get('hosted', False)}",
        f"count: {report.get('count', 0)}",
        "",
    ]
    artifacts = report.get("artifacts") or []
    if not artifacts:
        lines.append("(no mirrored artifacts)")
    else:
        for entry in artifacts:
            lines.append(
                f"{entry['identity']}  {entry['kind']}  "
                f"files={entry['file_count']}  bytes={entry['bytes']}"
            )
            lines.append(f"  {entry['uri']}")
    warnings = report.get("warnings") or []
    if warnings:
        lines.append("")
        lines.append("warnings:")
        for warning in warnings:
            lines.append(f"- {warning}")
    note = report.get("note")
    if note:
        lines.append("")
        lines.append(note)
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "CATALOG_LIST_SCHEMA",
    "CATALOG_MODE",
    "format_catalog_human",
    "list_local_catalog",
]
