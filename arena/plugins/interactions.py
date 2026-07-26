"""Interaction-mode cases (Parallel / AEC) for match and evaluation runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from arena.core.registry import Registry


class InteractionCase(Protocol):
    kind: str

    def run_match(self, **kwargs: Any) -> dict[str, Any]:
        ...


@dataclass
class ParallelInteraction:
    kind: str = "parallel"

    def run_match(self, **kwargs: Any) -> dict[str, Any]:
        from arena.runtime.match import run_match

        return run_match(**kwargs)


@dataclass
class AECInteraction:
    kind: str = "aec"

    def run_match(self, **kwargs: Any) -> dict[str, Any]:
        from arena.runtime.aec_match import run_aec_match

        return run_aec_match(**kwargs)


@dataclass
class DynamicAECInteraction:
    kind: str = "dynamic_aec"

    def run_match(self, **kwargs: Any) -> dict[str, Any]:
        from arena.runtime.dynamic_aec_match import run_dynamic_aec_match

        return run_dynamic_aec_match(**kwargs)


INTERACTIONS: Registry[InteractionCase] = Registry(
    "interaction",
    interface="InteractionCase",
    register_via="arena.plugins.interactions.register_interaction",
    tests="parallel atomicity + AEC selection/reward/reset fixtures (F5/F6)",
)


def register_interaction(
    kind: str, case: InteractionCase, *, replace: bool = False
) -> InteractionCase:
    return INTERACTIONS.register(kind, case, replace=replace)


def register_builtins() -> None:
    register_interaction("parallel", ParallelInteraction(), replace=True)
    register_interaction("aec", AECInteraction(), replace=True)
    register_interaction("dynamic_aec", DynamicAECInteraction(), replace=True)


def get_interaction(kind: str) -> InteractionCase:
    register_builtins()
    return INTERACTIONS.get(kind)


def require_interaction_kind(kind: str) -> str:
    """Validate kind via registry (fail loud with extension recipe)."""
    get_interaction(kind)
    return kind
