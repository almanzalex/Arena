"""Plugin-style case registries for RLX axes.

Stable axes (action type, distribution, preprocess/wrapper ops, actor payload,
task packaging) dispatch exclusively through registries. Unknown kinds fail loud
with an extension recipe — never silently coerce into a neighboring case.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from rlx.core.errors import SchemaError

T = TypeVar("T")


@dataclass(frozen=True)
class ExtensionRecipe:
    """What an author must do before claiming support for a new case."""

    axis: str
    kind: str
    interface: str
    register_via: str
    tests: str
    qualify: str = (
        "rlx adapter qualify <fixture> — required before claiming the case is supported"
    )

    def format(self, *, known: Iterable[str]) -> str:
        known_list = ", ".join(repr(k) for k in sorted(known)) or "(none)"
        return (
            f"Unknown {self.axis} kind {self.kind!r}. "
            f"Registered cases: [{known_list}]. "
            f"To add support: implement {self.interface}, register via "
            f"{self.register_via}, add tests covering {self.tests}, and run "
            f"`{self.qualify}` on a fixture that exercises the new case before "
            "claiming support. RLX will not silently coerce, flatten, pad, or "
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
    ) -> None:
        self.axis = axis
        self.interface = interface
        self.register_via = register_via
        self.tests = tests
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
        raise UnknownKindError(
            ExtensionRecipe(
                axis=self.axis,
                kind=str(kind),
                interface=self.interface,
                register_via=self.register_via,
                tests=self.tests,
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
# Axis registries (populated by rlx.plugins.builtins)
# ---------------------------------------------------------------------------

ACTION_CASES: Registry[Any] = Registry(
    "action",
    interface="rlx.plugins.actions.ActionCase",
    register_via="rlx.plugins.actions.register_action_case(kind, case)",
    tests="incomplete-claim fail-loud + complete end-to-end export/verify/act",
)

DISTRIBUTIONS: Registry[Any] = Registry(
    "distribution",
    interface="rlx.plugins.distributions.DistributionCase",
    register_via="rlx.plugins.distributions.register_distribution(kind, case)",
    tests="stochastic completeness (param_layout/transform/rng) + seed-match",
)

PREPROCESS_OPS: Registry[Any] = Registry(
    "preprocess",
    interface="rlx.plugins.preprocess_ops.PreprocessOp",
    register_via="rlx.plugins.preprocess_ops.register_preprocess_op(kind, op)",
    tests="shape-safe apply + unknown-op fail-loud",
)

WRAPPER_OPS: Registry[Any] = Registry(
    "wrapper",
    interface="rlx.plugins.wrappers.WrapperOp",
    register_via="rlx.plugins.wrappers.register_wrapper_op(kind, op)",
    tests="normalize + SuperSuit apply + unknown-op fail-loud",
)

PAYLOAD_LOADERS: Registry[Any] = Registry(
    "payload",
    interface="rlx.plugins.payloads.PayloadCase",
    register_via="rlx.plugins.payloads.register_payload_case(kind, case)",
    tests="load integrity + trust defaults (trusted_source refused without opt-in)",
)

TASK_PACKAGERS: Registry[Any] = Registry(
    "task_packaging",
    interface="rlx.plugins.tasks.TaskPackager",
    register_via="rlx.plugins.tasks.register_task_packager(kind, packager)",
    tests="make_env dispatch + trust defaults for entrypoint_bundle",
)

EVAL_PROVIDERS: Registry[Any] = Registry(
    "eval_provider",
    interface="rlx.plugins.eval_providers.EvalProvider",
    register_via="rlx.plugins.eval_providers.register_eval_provider(kind, provider)",
    tests="provider config identity + complete lineage + native-offline regression",
)

EXTERNAL_STORES: Registry[Any] = Registry(
    "external_store",
    interface="rlx.plugins.stores.ExternalStoreAdapter",
    register_via="rlx.plugins.stores.register_store_adapter(scheme, adapter)",
    tests="byte-identical push/pull --verify + tamper rejection + offline-core regression",
)

TRAINERS: Registry[Any] = Registry(
    "trainer",
    interface="rlx.plugins.trainers.TrainingCase",
    register_via="rlx.plugins.trainers.register_trainer(kind, trainer)",
    tests="recipe validation + seeded reproducibility + portable policy verify/reuse",
)

LIFECYCLE_RESOLVERS: Registry[Any] = Registry(
    "lifecycle_resolver",
    interface="rlx.plugins.lifecycle.LifecycleResolver",
    register_via="rlx.plugins.lifecycle.register_lifecycle_resolver(kind, resolver)",
    tests="assignment preflight + join eligibility + rejoin segment provenance",
)


def ensure_plugins_loaded() -> None:
    """Idempotently register built-in cases (import side-effect safe)."""
    from rlx.plugins import builtins as _builtins

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


def register_entry_points(group: str = "rlx.plugins") -> list[str]:
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
            loader()
        except Exception as e:  # noqa: BLE001
            raise SchemaError(
                f"failed loading entry-point plugin {ep.name!r} from group "
                f"{group!r}: {e}"
            ) from e
        loaded.append(ep.name)
    return loaded
