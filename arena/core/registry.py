"""Plugin-style case registries for Arena axes.

Stable axes (action type, distribution, preprocess/wrapper ops, actor payload,
task packaging) dispatch exclusively through registries. Unknown kinds fail loud
with an extension recipe — never silently coerce into a neighboring case.
"""

from __future__ import annotations

import contextlib
import sys
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from arena.core.errors import PluginError, SchemaError

T = TypeVar("T")
_LOADED_ENTRY_POINTS: set[tuple[str, str]] = set()
_LOADING_ENTRY_POINTS: set[tuple[str, str]] = set()
_PLUGIN_LOCK = threading.RLock()


@dataclass(frozen=True)
class ExtensionRecipe:
    """What an author must do before claiming support for a new case."""

    axis: str
    kind: str
    interface: str
    register_via: str
    tests: str
    qualify: str = (
        "arena adapter qualify <fixture> — required before claiming the case is supported"
    )

    def format(self, *, known: Iterable[str]) -> str:
        known_list = ", ".join(repr(k) for k in sorted(known)) or "(none)"
        return (
            f"Unknown {self.axis} kind {self.kind!r}. "
            f"Registered cases: [{known_list}]. "
            f"To add support: implement {self.interface}, register via "
            f"{self.register_via}, add tests covering {self.tests}, and run "
            f"`{self.qualify}` on a fixture that exercises the new case before "
            "claiming support. Arena will not silently coerce, flatten, pad, or "
            "approximate an unregistered case."
        )


class UnknownKindError(SchemaError):
    """Raised when ``Registry.get`` cannot resolve a kind."""

    def __init__(self, recipe: ExtensionRecipe, *, known: Iterable[str]) -> None:
        self.recipe = recipe
        self.known = frozenset(known)
        super().__init__(recipe.format(known=self.known))


class Registry(Generic[T]):
    """Named case registry with fail-loud lookup and extension recipes."""

    def __init__(
        self,
        axis: str,
        *,
        interface: str,
        register_via: str,
        tests: str,
        qualify: str | None = None,
    ) -> None:
        self.axis = axis
        self.interface = interface
        self.register_via = register_via
        self.tests = tests
        self.qualify = (
            qualify
            if qualify is not None
            else ExtensionRecipe.__dataclass_fields__["qualify"].default
        )
        self._cases: dict[str, T] = {}

    def register(self, kind: str, case: T, *, replace: bool = False) -> T:
        if not kind or not isinstance(kind, str):
            raise SchemaError(f"{self.axis} kind must be a non-empty string")
        if kind in self._cases and not replace:
            raise SchemaError(
                f"{self.axis} kind {kind!r} is already registered; "
                "pass replace=True only for deliberate test overrides"
            )
        self._cases[kind] = case
        return case

    def unregister(self, kind: str) -> None:
        self._cases.pop(kind, None)

    def get(self, kind: str) -> T:
        if kind in self._cases:
            return self._cases[kind]
        ensure_plugins_loaded()
        load_entry_points_for(self.axis, str(kind))
        if kind in self._cases:
            return self._cases[kind]
        raise UnknownKindError(
            ExtensionRecipe(
                axis=self.axis,
                kind=str(kind),
                interface=self.interface,
                register_via=self.register_via,
                tests=self.tests,
                qualify=self.qualify,
            ),
            known=self._cases.keys(),
        )

    def known(self) -> frozenset[str]:
        return frozenset(self._cases.keys())

    def items(self) -> Iterator[tuple[str, T]]:
        yield from sorted(self._cases.items(), key=lambda kv: kv[0])

    def __contains__(self, kind: object) -> bool:
        return isinstance(kind, str) and kind in self._cases

    def __len__(self) -> int:
        return len(self._cases)


def require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{label} must be a mapping, got {type(value).__name__}")
    return value


# ---------------------------------------------------------------------------
# Axis registries (populated by arena.plugins.builtins)
# ---------------------------------------------------------------------------

ACTION_CASES: Registry[Any] = Registry(
    "action",
    interface="arena.plugins.actions.ActionCase",
    register_via="arena.plugins.actions.register_action_case(kind, case)",
    tests="incomplete-claim fail-loud + complete end-to-end export/verify/act",
)

DISTRIBUTIONS: Registry[Any] = Registry(
    "distribution",
    interface="arena.plugins.distributions.DistributionCase",
    register_via="arena.plugins.distributions.register_distribution(kind, case)",
    tests="stochastic completeness (param_layout/transform/rng) + seed-match",
)

PREPROCESS_OPS: Registry[Any] = Registry(
    "preprocess",
    interface="arena.plugins.preprocess_ops.PreprocessOp",
    register_via="arena.plugins.preprocess_ops.register_preprocess_op(kind, op)",
    tests="shape-safe apply + unknown-op fail-loud",
)

WRAPPER_OPS: Registry[Any] = Registry(
    "wrapper",
    interface="arena.plugins.wrappers.WrapperOp",
    register_via="arena.plugins.wrappers.register_wrapper_op(kind, op)",
    tests="normalize + SuperSuit apply + unknown-op fail-loud",
)

PAYLOAD_LOADERS: Registry[Any] = Registry(
    "payload",
    interface="arena.plugins.payloads.PayloadCase",
    register_via="arena.plugins.payloads.register_payload_case(kind, case)",
    tests="load integrity + trust defaults (trusted_source refused without opt-in)",
)

TASK_PACKAGERS: Registry[Any] = Registry(
    "task_packaging",
    interface="arena.plugins.tasks.TaskPackager",
    register_via="arena.plugins.tasks.register_task_packager(kind, packager)",
    tests="make_env dispatch + trust defaults for entrypoint_bundle",
)

EVAL_PROVIDERS: Registry[Any] = Registry(
    "eval_provider",
    interface="arena.plugins.eval_providers.EvalProvider",
    register_via="arena.plugins.eval_providers.register_eval_provider(kind, provider)",
    tests="provider config identity + complete lineage + native-offline regression",
)

EXTERNAL_STORES: Registry[Any] = Registry(
    "external_store",
    interface="arena.plugins.stores.ExternalStoreAdapter",
    register_via="arena.plugins.stores.register_store_adapter(scheme, adapter)",
    tests="byte-identical push/pull --verify + tamper rejection + offline-core regression",
    qualify=(
        "arena store qualify <artifact> <destination> — required before claiming "
        "the store case is supported"
    ),
)

TRAINERS: Registry[Any] = Registry(
    "trainer",
    interface="arena.plugins.trainers.TrainingCase",
    register_via="arena.plugins.trainers.register_trainer(kind, trainer)",
    tests="recipe validation + seeded reproducibility + portable policy verify/reuse",
)

LIFECYCLE_RESOLVERS: Registry[Any] = Registry(
    "lifecycle_resolver",
    interface="arena.plugins.lifecycle.LifecycleResolver",
    register_via="arena.plugins.lifecycle.register_lifecycle_resolver(kind, resolver)",
    tests="assignment preflight + join eligibility + rejoin segment provenance",
)


def ensure_plugins_loaded() -> None:
    """Idempotently register built-ins without importing optional plugins."""

    from arena.plugins import builtins as _builtins

    _builtins.ensure_registered()


def capability_matrix() -> dict[str, list[str]]:
    """Return currently registered kinds per axis (qualified support is separate)."""
    ensure_plugins_loaded()
    return {
        "action": sorted(ACTION_CASES.known()),
        "distribution": sorted(DISTRIBUTIONS.known()),
        "preprocess": sorted(PREPROCESS_OPS.known()),
        "wrapper": sorted(WRAPPER_OPS.known()),
        "payload": sorted(PAYLOAD_LOADERS.known()),
        "task_packaging": sorted(TASK_PACKAGERS.known()),
        "eval_provider": sorted(EVAL_PROVIDERS.known()),
        "external_store": sorted(EXTERNAL_STORES.known()),
        "trainer": sorted(TRAINERS.known()),
        "lifecycle_resolver": sorted(LIFECYCLE_RESOLVERS.known()),
    }


def register_entry_points(group: str = "arena.plugins.v1") -> list[str]:
    """Load optional third-party plugins via importlib entry points.

    Returns the names of loaded entry points. Failures raise SchemaError so a
    broken plugin cannot silently disable a case.
    """
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover
        return []
    loaded: list[str] = []
    eps = entry_points()
    selected = eps.select(group=group) if hasattr(eps, "select") else eps.get(group, [])
    for ep in selected:
        try:
            loader: Callable[[], Any] = ep.load()
            if not callable(loader):
                raise TypeError("entry point did not resolve to a callable")
            # Plugins may report progress, but JSON stdout belongs exclusively to
            # Arena's versioned result envelope.
            with contextlib.redirect_stdout(sys.stderr):
                loader()
        except Exception as e:  # noqa: BLE001
            distribution = getattr(getattr(ep, "dist", None), "name", "unknown")
            version = getattr(getattr(ep, "dist", None), "version", "unknown")
            raise PluginError(
                f"failed loading entry-point plugin {ep.name!r} from "
                f"{distribution}=={version}: {e}",
                code="PLUGIN_LOAD_FAILED",
                repair=(
                    f"Uninstall or repair {distribution}=={version}; Arena core remains "
                    "usable when the plugin is not referenced."
                ),
                context={
                    "entry_point": ep.name,
                    "group": group,
                    "distribution": distribution,
                    "version": version,
                },
            ) from e
        loaded.append(ep.name)
    return loaded


def load_entry_points_for(axis: str, kind: str) -> list[str]:
    """Load only the v1 plugin named ``<axis>:<kind>`` on a registry miss.

    The old unversioned group is retained for 1.x compatibility and is consulted
    only after a miss. New plugins must use the explicit name so unrelated or
    broken distributions cannot affect core startup or another registry kind.
    """
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover
        return []
    target = f"{axis}:{kind}"
    loaded: list[str] = []
    with _PLUGIN_LOCK:
        key = ("arena.plugins.v1", target)
        if key in _LOADED_ENTRY_POINTS or key in _LOADING_ENTRY_POINTS:
            return []
        eps = entry_points()
        selected = (
            eps.select(group="arena.plugins.v1")
            if hasattr(eps, "select")
            else eps.get("arena.plugins.v1", [])
        )
        matches = [ep for ep in selected if ep.name == target]
        if matches:
            if len(matches) > 1:
                raise PluginError(
                    f"multiple plugins claim {target!r}",
                    code="PLUGIN_NAME_COLLISION",
                )
            _LOADING_ENTRY_POINTS.add(key)
            try:
                _load_entry_point(matches[0], group="arena.plugins.v1")
                _LOADED_ENTRY_POINTS.add(key)
                loaded.append(target)
            finally:
                _LOADING_ENTRY_POINTS.discard(key)
            return loaded

        # Legacy 0.5 entry points had no naming convention. Import them only on
        # a registry miss and only once apiece.
        legacy = (
            eps.select(group="arena.plugins")
            if hasattr(eps, "select")
            else eps.get("arena.plugins", [])
        )
        for ep in legacy:
            legacy_key = ("arena.plugins", ep.name)
            if legacy_key in _LOADED_ENTRY_POINTS:
                continue
            _load_entry_point(ep, group="arena.plugins")
            _LOADED_ENTRY_POINTS.add(legacy_key)
            loaded.append(ep.name)
    return loaded


def _load_entry_point(ep: Any, *, group: str) -> None:
    try:
        loader: Callable[[], Any] = ep.load()
        if not callable(loader):
            raise TypeError("entry point did not resolve to a callable")
        with contextlib.redirect_stdout(sys.stderr):
            loader()
    except Exception as e:  # noqa: BLE001
        distribution = getattr(getattr(ep, "dist", None), "name", "unknown")
        version = getattr(getattr(ep, "dist", None), "version", "unknown")
        raise PluginError(
            f"failed loading entry-point plugin {ep.name!r} from "
            f"{distribution}=={version}: {e}",
            code="PLUGIN_LOAD_FAILED",
            repair=(
                f"Uninstall or repair {distribution}=={version}; Arena core remains "
                "usable when the plugin is not referenced."
            ),
            context={
                "entry_point": ep.name,
                "group": group,
                "distribution": distribution,
                "version": version,
            },
        ) from e
