"""Dynamic-agent policy-resolution cases.

Resolvers turn immutable assignment inputs into an agent-to-policy plan before
the run exists. They never choose an undeclared policy at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Protocol

from arena.core.errors import CompatibilityError, SchemaError
from arena.core.registry import LIFECYCLE_RESOLVERS


@dataclass(frozen=True)
class LifecycleBinding:
    agent: str
    assignment_key: str
    policy_role: str
    policy: Any


@dataclass(frozen=True)
class LifecyclePlan:
    kind: str
    bindings: Mapping[str, LifecycleBinding]
    join_eligibility: Mapping[str, frozenset[str]]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "bindings",
            MappingProxyType(dict(self.bindings)),
        )
        object.__setattr__(
            self,
            "join_eligibility",
            MappingProxyType(
                {
                    agent: frozenset(digests)
                    for agent, digests in self.join_eligibility.items()
                }
            ),
        )

    def binding_for(self, agent: str) -> LifecycleBinding:
        if agent not in self.bindings:
            raise CompatibilityError(
                f"dynamic agent {agent!r} has no declared lifecycle binding"
            )
        return self.bindings[agent]

    def verify_join(self, agent: str) -> LifecycleBinding:
        binding = self.binding_for(agent)
        eligible = self.join_eligibility.get(agent)
        if eligible is not None and binding.policy.digest not in eligible:
            raise CompatibilityError(
                f"join eligibility for {agent!r} does not include assigned policy "
                f"{binding.policy.digest}"
            )
        return binding


class LifecycleResolver(Protocol):
    kind: str

    def prepare(
        self,
        *,
        task_spec: dict[str, Any],
        task_info: dict[str, Any],
        assignments: dict[str, Any],
    ) -> LifecyclePlan: ...


@dataclass(frozen=True)
class ExplicitAgentResolver:
    """Back-compatible exact `agent_id -> policy` resolver."""

    kind: str = "explicit"

    def prepare(
        self,
        *,
        task_spec: dict[str, Any],
        task_info: dict[str, Any],
        assignments: dict[str, Any],
    ) -> LifecyclePlan:
        possible = list(task_info.get("possible_agents") or task_info["roles"])
        missing = sorted(set(possible) - set(assignments))
        extra = sorted(set(assignments) - set(possible))
        if missing or extra:
            raise CompatibilityError(
                "dynamic assignments must cover possible_agents exactly; "
                f"missing={missing}, extra={extra}"
            )
        initial = set(task_info.get("agents") or [])
        births = set(possible) - initial
        lifecycle = dict(task_spec.get("lifecycle") or {})
        raw_eligibility = dict(lifecycle.get("birth_eligibility") or {})
        eligibility: dict[str, frozenset[str]] = {}
        bindings: dict[str, LifecycleBinding] = {}
        for agent in possible:
            policy = assignments[agent]
            bindings[agent] = LifecycleBinding(
                agent=agent,
                assignment_key=agent,
                policy_role=agent,
                policy=policy,
            )
            if agent in births:
                values = raw_eligibility.get(agent)
                if not isinstance(values, list) or policy.digest not in values:
                    raise CompatibilityError(
                        f"birth_eligibility.{agent} must explicitly include assigned "
                        f"policy digest {policy.digest}"
                    )
                eligibility[agent] = frozenset(str(item) for item in values)
        return LifecyclePlan(
            kind=self.kind,
            bindings=bindings,
            join_eligibility=eligibility,
        )


@dataclass(frozen=True)
class RoleResolver:
    """Map many concrete agent IDs onto stable policy roles."""

    kind: str = "role"

    def prepare(
        self,
        *,
        task_spec: dict[str, Any],
        task_info: dict[str, Any],
        assignments: dict[str, Any],
    ) -> LifecyclePlan:
        lifecycle = dict(task_spec.get("lifecycle") or {})
        resolver = dict(lifecycle.get("resolver") or {})
        agent_roles = dict(resolver.get("agent_roles") or lifecycle.get("agent_roles") or {})
        possible = list(task_info.get("possible_agents") or task_info["roles"])
        missing_agents = sorted(set(possible) - set(agent_roles))
        extra_agents = sorted(set(agent_roles) - set(possible))
        if missing_agents or extra_agents:
            raise CompatibilityError(
                "role lifecycle resolver must map possible_agents exactly; "
                f"missing={missing_agents}, extra={extra_agents}"
            )
        used_roles = {str(agent_roles[agent]) for agent in possible}
        missing_roles = sorted(used_roles - set(assignments))
        extra_roles = sorted(set(assignments) - used_roles)
        if missing_roles or extra_roles:
            raise CompatibilityError(
                "role lifecycle assignments must cover declared roles exactly; "
                f"missing={missing_roles}, extra={extra_roles}"
            )
        raw_join = dict(
            resolver.get("join_eligibility")
            or lifecycle.get("join_eligibility")
            or lifecycle.get("birth_eligibility")
            or {}
        )
        if not raw_join:
            raise SchemaError(
                "role lifecycle resolver requires join_eligibility by policy role"
            )
        bindings: dict[str, LifecycleBinding] = {}
        eligibility: dict[str, frozenset[str]] = {}
        for agent in possible:
            role = str(agent_roles[agent])
            policy = assignments[role]
            values = raw_join.get(role)
            if not isinstance(values, list) or policy.digest not in values:
                raise CompatibilityError(
                    f"join_eligibility.{role} must explicitly include assigned policy "
                    f"digest {policy.digest}"
                )
            bindings[agent] = LifecycleBinding(
                agent=agent,
                assignment_key=role,
                policy_role=role,
                policy=policy,
            )
            eligibility[agent] = frozenset(str(item) for item in values)
        return LifecyclePlan(
            kind=self.kind,
            bindings=bindings,
            join_eligibility=eligibility,
        )


def register_lifecycle_resolver(
    kind: str,
    resolver: LifecycleResolver,
    *,
    replace: bool = False,
) -> LifecycleResolver:
    return LIFECYCLE_RESOLVERS.register(kind, resolver, replace=replace)


def register_builtins() -> None:
    register_lifecycle_resolver("explicit", ExplicitAgentResolver(), replace=True)
    register_lifecycle_resolver("role", RoleResolver(), replace=True)


def resolve_lifecycle_plan(
    task_spec: dict[str, Any],
    task_info: dict[str, Any],
    assignments: dict[str, Any],
) -> LifecyclePlan:
    from arena.core.registry import ensure_plugins_loaded

    ensure_plugins_loaded()
    lifecycle = dict(task_spec.get("lifecycle") or {})
    resolver = dict(lifecycle.get("resolver") or {})
    kind = str(resolver.get("kind", "explicit"))
    return LIFECYCLE_RESOLVERS.get(kind).prepare(
        task_spec=task_spec,
        task_info=task_info,
        assignments=assignments,
    )
