"""Local content-addressed workspace (.arena/)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from arena.core.errors import SchemaError, StoreError
from arena.core.identity import digest_uri, parse_digest, sha256_bytes, sha256_file
from arena.core.io import atomic_create_bytes, atomic_write_bytes, verify_regular_file

WORKSPACE_DIR = ".arena"
WORKSPACE_TOML = "workspace.toml"


class LocalStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()
        self.arena = self.root / WORKSPACE_DIR
        self.objects = self.arena / "objects"
        self.refs = self.arena / "refs"
        self.runs = self.arena / "runs"
        self.cache = self.arena / "cache"

    @classmethod
    def find(cls, start: Path | str | None = None) -> LocalStore:
        cur = Path(start or Path.cwd()).resolve()
        for candidate in [cur, *cur.parents]:
            if (candidate / WORKSPACE_DIR / WORKSPACE_TOML).exists():
                return cls(candidate)
        raise StoreError(
            f"no {WORKSPACE_DIR}/ workspace found from {cur}; run `arena init` first"
        )

    def init(self, *, force: bool = False) -> Path:
        if self.arena.exists() and not force:
            if (self.arena / WORKSPACE_TOML).exists():
                return self.arena
        for d in (self.objects, self.refs, self.runs, self.cache):
            d.mkdir(parents=True, exist_ok=True)
        config = {
            "version": 1,
            "adapters": {
                "policy": ["custom-pytorch"],
                "task": ["pettingzoo-parallel"],
            },
        }
        atomic_write_bytes(
            self.arena / WORKSPACE_TOML,
            (
                "# Arena local workspace\n" + yaml.safe_dump(config, sort_keys=False)
            ).encode("utf-8"),
        )
        return self.arena

    def _object_path(self, digest_hex: str) -> Path:
        return self.objects / digest_hex[:2] / digest_hex[2:]

    def put_bytes(self, data: bytes) -> str:
        digest_hex = sha256_bytes(data)
        dest = self._object_path(digest_hex)
        if dest.exists():
            verify_regular_file(dest, root=self.objects)
            existing = sha256_bytes(dest.read_bytes())
            if existing != digest_hex:
                raise StoreError(
                    f"refusing corrupt pre-existing object at sha256:{digest_hex}"
                )
            return digest_uri(digest_hex)
        atomic_create_bytes(dest, data)
        return digest_uri(digest_hex)

    def put_file(self, path: Path | str) -> str:
        path = Path(path)
        digest_hex = sha256_file(path)
        dest = self._object_path(digest_hex)
        if dest.exists():
            return digest_uri(digest_hex)
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = path.read_bytes()
        return self.put_bytes(data)

    def get_bytes(self, digest: str, *, verify: bool = True) -> bytes:
        del verify  # Content-addressed bytes are unconditionally verified in 1.0.
        try:
            digest_hex = parse_digest(digest)
        except SchemaError as exc:
            raise StoreError(f"invalid object digest {digest!r}: {exc}") from exc
        path = self._object_path(digest_hex)
        if not path.exists():
            raise StoreError(f"object not found: {digest}")
        verify_regular_file(path, root=self.objects)
        data = path.read_bytes()
        # Content-addressed integrity: a stored object must hash to its own name.
        # Detect on-disk corruption/tampering instead of silently trusting bytes.
        actual = sha256_bytes(data)
        if actual != digest_hex:
            raise StoreError(
                "object integrity check failed (content does not match digest): "
                f"requested sha256:{digest_hex}, on-disk content is sha256:{actual}"
            )
        return data

    def verify_object(self, digest: str) -> bool:
        """Re-hash a stored object and confirm it matches its digest."""
        self.get_bytes(digest, verify=True)
        return True

    def open_path(self, digest: str) -> Path:
        try:
            digest_hex = parse_digest(digest)
        except SchemaError as exc:
            raise StoreError(f"invalid object digest {digest!r}: {exc}") from exc
        path = self._object_path(digest_hex)
        if not path.exists():
            raise StoreError(f"object not found: {digest}")
        verify_regular_file(path, root=self.objects)
        self.get_bytes(digest)
        return path

    def _ref_path(self, name: str) -> Path:
        """Resolve a ref name to a path that is guaranteed to stay inside refs/.

        Ref names are human-readable and may contain ``:`` / ``@`` separators, but
        they must never be able to escape the refs directory (path traversal via
        ``..``, absolute paths, or drive letters). Anything that resolves outside
        ``self.refs`` is rejected loudly.
        """
        if not name or not name.strip():
            raise StoreError("ref name must be a non-empty string")
        safe = name.replace(":", "/").replace("@", "/")
        candidate = Path(safe)
        if candidate.is_absolute() or (candidate.drive or candidate.root):
            raise StoreError(f"invalid ref name (absolute paths are not allowed): {name!r}")
        if ".." in candidate.parts:
            raise StoreError(f"invalid ref name (path traversal not allowed): {name!r}")
        refs_root = self.refs.resolve()
        path = (refs_root / safe).resolve()
        if path != refs_root and refs_root not in path.parents:
            raise StoreError(f"invalid ref name (escapes refs directory): {name!r}")
        return path

    def set_ref(self, name: str, digest: str) -> None:
        try:
            canonical = digest_uri(parse_digest(digest))
        except SchemaError as exc:
            raise StoreError(f"ref value must be a valid SHA-256 digest: {digest!r}") from exc
        path = self._ref_path(name)
        atomic_write_bytes(path, (canonical + "\n").encode("utf-8"))

    def get_ref(self, name: str) -> str:
        path = self._ref_path(name)
        if not path.exists():
            raise StoreError(f"ref not found: {name}")
        return path.read_text(encoding="utf-8").strip()

    def run_dir(self, run_id: str) -> Path:
        path = self.runs / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def read_workspace_config(self) -> dict[str, Any]:
        path = self.arena / WORKSPACE_TOML
        if not path.exists():
            raise StoreError("workspace.toml missing")
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
