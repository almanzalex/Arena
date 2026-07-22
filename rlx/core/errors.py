"""Structured RLX errors and compatibility reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class RlxError(Exception):
    """Base error for RLX."""


class SchemaError(RlxError):
    """Manifest or payload failed schema validation."""


class StoreError(RlxError):
    """Local workspace / object store error."""


class CompatibilityError(RlxError):
    """Artifacts cannot be composed safely."""


class ConformanceError(RlxError):
    """Source-versus-exported behavior mismatch."""


class RuntimeFailure(RlxError):
    """Match or policy runtime failure (never silently dropped)."""

    def __init__(
        self,
        message: str,
        *,
        kind: str,
        episode_index: int | None = None,
        agent: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.episode_index = episode_index
        self.agent = agent
        self.details = details or {}


class TaskRuntimeError(RlxError):
    """Failure reported by an external task transport before runtime accounting.

    Task adapters use this narrow error to preserve disconnect, remote crash, and
    transport timeout semantics. Match runners convert it into ``RuntimeFailure``
    records instead of collapsing every external failure into ``crash``.
    """

    def __init__(
        self,
        message: str,
        *,
        kind: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        if kind not in {"disconnect", "container_crash", "timeout", "protocol_error"}:
            raise ValueError(f"invalid task runtime failure kind: {kind!r}")
        super().__init__(message)
        self.kind = kind
        self.details = details or {}


@dataclass
class CompatibilityIssue:
    code: str
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)
    repairs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "evidence": self.evidence,
            "repairs": self.repairs,
        }


@dataclass
class CompatibilityReport:
    ok: bool
    issues: list[CompatibilityIssue] = field(default_factory=list)

    def raise_for_errors(self) -> None:
        if not self.ok:
            parts = [f"{i.code}: {i.message}" for i in self.issues]
            raise CompatibilityError("; ".join(parts))

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "issues": [i.to_dict() for i in self.issues]}

    def format_human(self) -> str:
        if self.ok:
            return "COMPATIBLE: all checked contracts match."
        lines: list[str] = []
        for issue in self.issues:
            lines.append(f"INCOMPATIBLE: {issue.message}")
            if issue.evidence:
                lines.append("Evidence:")
                for k, v in issue.evidence.items():
                    lines.append(f"  {k} = {v!r}")
            if issue.repairs:
                lines.append("Suggested repairs:")
                for r in issue.repairs:
                    lines.append(f"  - {r}")
            lines.append("")
        return "\n".join(lines).rstrip()
