"""Task packaging cases and the external-task extension boundary."""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path
from typing import Any, Protocol

from arena.core.errors import SchemaError
from arena.core.identity import digest_uri, sha256_file
from arena.core.registry import TASK_PACKAGERS

ENTRYPOINT_BUNDLE_WARNING = (
    "WARNING: task packaging 'entrypoint_bundle' executes a digest-pinned Python "
    "entrypoint. This is NOT sandboxed. Only opt in with --trust-task-code for "
    "code you already trust. Prefer pettingzoo_wrappers for declared SuperSuit chains."
)


class TaskPackager(Protocol):
    kind: str

    def make_env(self, spec: dict[str, Any], *, trust_task_code: bool = False) -> Any: ...

    def describe_task(self, spec: dict[str, Any]) -> dict[str, Any]: ...


def register_task_packager(
    kind: str, packager: TaskPackager, *, replace: bool = False
) -> TaskPackager:
    return TASK_PACKAGERS.register(kind, packager, replace=replace)


def resolve_packaging_kind(spec: dict[str, Any]) -> str:
    """Default packaging is pettingzoo_wrappers for existing PettingZoo tasks."""
    packaging = spec.get("packaging") or spec.get("task_packaging")
    if packaging is None:
        adapter = str(spec.get("adapter") or "pettingzoo-parallel")
        packaging = {
            "openenv": "openenv",
            "openspiel": "openspiel",
            "pettingzoo-parallel": "pettingzoo_wrappers",
            "pettingzoo-aec": "pettingzoo_wrappers",
        }.get(adapter, adapter)
    if isinstance(packaging, dict):
        kind = packaging.get("kind")
    else:
        kind = packaging
    if not isinstance(kind, str) or not kind:
        raise SchemaError("task packaging kind must be a non-empty string")
    return kind


class PettingZooWrappersPackager:
    """Existing SuperSuit-declared wrapper chain on PettingZoo Parallel envs."""

    kind = "pettingzoo_wrappers"

    def make_env(self, spec: dict[str, Any], *, trust_task_code: bool = False) -> Any:
        del trust_task_code
        from arena.adapters.task_pettingzoo import adapter as pz

        # Delegate to the existing PettingZoo path (wrappers applied inside).
        return pz._make_env_pettingzoo(spec)

    def describe_task(self, spec: dict[str, Any]) -> dict[str, Any]:
        from arena.adapters.task_pettingzoo import adapter as pz

        return pz._describe_pettingzoo_task(spec)


class EntrypointBundlePackager:
    """Narrow trusted task entrypoint with digest pin (opt-in, not sandboxed)."""

    kind = "entrypoint_bundle"

    def make_env(self, spec: dict[str, Any], *, trust_task_code: bool = False) -> Any:
        packaging = spec.get("packaging") if isinstance(spec.get("packaging"), dict) else {}
        if not trust_task_code and not packaging.get("trust_task_code"):
            raise SchemaError(
                "task packaging 'entrypoint_bundle' refused by default. "
                "This path executes digest-pinned Python and is NOT sandboxed. "
                "Pass trust_task_code=True / --trust-task-code after reviewing the "
                "pinned entrypoint digest, or use packaging.kind=pettingzoo_wrappers."
            )
        warnings.warn(ENTRYPOINT_BUNDLE_WARNING, UserWarning, stacklevel=2)

        entry = packaging.get("entrypoint") or spec.get("entrypoint")
        digest = packaging.get("digest") or spec.get("entrypoint_digest")
        root = packaging.get("root") or spec.get("bundle_root")
        if not entry or not digest:
            raise SchemaError(
                "entrypoint_bundle requires packaging.entrypoint (relative .py path or "
                "module:attr) and packaging.digest (sha256:…). Refusing incomplete claim."
            )
        if root is None:
            raise SchemaError(
                "entrypoint_bundle requires packaging.root (bundle directory containing "
                "the pinned entrypoint file)"
            )
        root_path = Path(root)
        # Path-traversal safe resolve when entry is a relative file path.
        if ":" in str(entry) and not str(entry).endswith(".py"):
            # module:attr form is only allowed when digest pins a source file listed in
            # packaging.source — keep the MVP to a single .py file for clarity.
            raise SchemaError(
                "entrypoint_bundle MVP requires packaging.entrypoint to be a relative "
                ".py file path under packaging.root (module:attr deferred)."
            )
        src = _safe_join(root_path, str(entry))
        if not src.is_file() or src.suffix != ".py":
            raise SchemaError(f"entrypoint_bundle entrypoint missing or not a .py file: {src}")
        actual = digest_uri(sha256_file(src))
        if actual != digest:
            raise SchemaError(
                f"entrypoint_bundle digest mismatch (declared {digest}, actual {actual}). "
                "Refusing to import tampered task code."
            )
        factory_attr = packaging.get("factory", "parallel_env")
        factory = _import_pinned_source(src, attr=str(factory_attr))
        config = dict(spec.get("config") or {})
        config.pop("seed", None)
        if not callable(factory):
            raise SchemaError(f"entrypoint factory {factory_attr!r} is not callable")
        try:
            env = factory(**config) if config else factory()
        except TypeError:
            env = factory()
        # Optional declared wrappers still go through the wrapper registry.
        wrappers = packaging.get("wrappers", spec.get("wrappers"))
        if wrappers:
            from arena.adapters.task_pettingzoo.wrappers import apply_wrappers

            env = apply_wrappers(env, wrappers)
        return env

    def describe_task(self, spec: dict[str, Any]) -> dict[str, Any]:
        from arena.adapters.task_pettingzoo import adapter as pz

        env = self.make_env(spec, trust_task_code=bool(spec.get("trust_task_code")))
        return pz.describe_env_contract(
            spec,
            env,
            adapter_name="entrypoint_bundle",
            version=str(spec.get("source_revision") or "digest-pinned"),
        )


def _safe_join(root: Path, rel: str) -> Path:
    if Path(rel).is_absolute():
        raise SchemaError(f"entrypoint path must be relative, got {rel!r}")
    candidate = (root / rel).resolve()
    try:
        resolved_root = root.resolve()
    except OSError as e:
        raise SchemaError(f"cannot resolve entrypoint root {root}") from e
    if resolved_root not in candidate.parents and candidate != resolved_root:
        raise SchemaError(f"entrypoint path escapes bundle root: {rel!r}")
    return candidate


def _import_pinned_source(src_path: Path, *, attr: str) -> Any:
    mod_name = f"arena_entrypoint_{sha256_file(src_path)[:12]}"
    spec = importlib.util.spec_from_file_location(mod_name, src_path)
    if spec is None or spec.loader is None:
        raise SchemaError(f"cannot import entrypoint from {src_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # noqa: BLE001
        sys.modules.pop(mod_name, None)
        raise SchemaError(f"entrypoint_bundle import failed: {e}") from e
    if not hasattr(mod, attr):
        raise SchemaError(f"entrypoint module has no attribute {attr!r}")
    return getattr(mod, attr)


def register_builtins() -> None:
    register_task_packager("pettingzoo_wrappers", PettingZooWrappersPackager(), replace=True)
    register_task_packager("entrypoint_bundle", EntrypointBundlePackager(), replace=True)
    from arena.adapters.task_openenv import OpenEnvPackager
    from arena.adapters.task_openspiel import OpenSpielPackager

    register_task_packager("openenv", OpenEnvPackager(), replace=True)
    register_task_packager("openspiel", OpenSpielPackager(), replace=True)
