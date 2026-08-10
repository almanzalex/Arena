"""Structured Arena errors and compatibility reports."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


class ArenaError(Exception):
    """Base error for expected Arena failures.

    Existing call sites may continue to pass only a message. New boundaries can
    attach stable diagnostic metadata without creating adapter-specific envelopes.
    """

    default_code = "ARENA_ERROR"
    category = "internal"
    exit_code = 70

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        cause: str | None = None,
        repair: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code or self.default_code
        self.cause = cause
        self.repair = repair
        self.context = context or {}


class SchemaError(ArenaError):
    """Manifest or payload failed schema validation."""

    default_code = "SCHEMA_INVALID"
    category = "schema_compatibility"
    exit_code = 3


class StoreError(ArenaError):
    """Local workspace / object store error."""

    default_code = "STORE_FAILURE"
    category = "external"
    exit_code = 5


class CompatibilityError(ArenaError):
    """Artifacts cannot be composed safely."""

    default_code = "COMPATIBILITY_FAILED"
    category = "schema_compatibility"
    exit_code = 3


class ConformanceError(ArenaError):
    """Source-versus-exported behavior mismatch."""

    default_code = "CONFORMANCE_FAILED"
    category = "integrity_authenticity"
    exit_code = 4


class IntegrityError(ArenaError):
    """Known-hash bytes or artifact structure failed integrity verification."""

    default_code = "INTEGRITY_FAILED"
    category = "integrity_authenticity"
    exit_code = 4


class ExternalUnavailableError(ArenaError):
    """An external service or executable was unavailable."""

    default_code = "EXTERNAL_UNAVAILABLE"
    category = "external"
    exit_code = 5


class IncompleteExecutionError(ArenaError):
    """Execution was cancelled, timed out, or produced incomplete evidence."""

    default_code = "EXECUTION_INCOMPLETE"
    category = "incomplete_cancelled"
    exit_code = 6


class PluginError(ArenaError):
    """A third-party plugin could not be discovered or registered safely."""

    default_code = "PLUGIN_FAILURE"
    category = "schema_compatibility"
    exit_code = 3


class CliUsageError(ArenaError):
    """Command-line grammar or argument validation failed."""

    default_code = "USAGE_INVALID"
    category = "usage"
    exit_code = 2


class RuntimeFailure(ArenaError):
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
        super().__init__(
            message,
            code=f"RUNTIME_{kind.upper()}",
            cause=kind.replace("_", " "),
            context={
                "episode_index": episode_index,
                "agent": agent,
                **(details or {}),
            },
        )
        self.kind = kind
        self.episode_index = episode_index
        self.agent = agent
        self.details = details or {}


class TaskRuntimeError(ArenaError):
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
        super().__init__(
            message,
            code=f"TASK_{kind.upper()}",
            cause=kind.replace("_", " "),
            context=details,
        )
        self.kind = kind
        self.details = details or {}


_SECRET_KEY = re.compile(
    r"(authorization|credential|password|private|secret|token|api[_-]?key|cookie)",
    re.IGNORECASE,
)
_BEARER = re.compile(r"(?i)\b(bearer|token)\s+[A-Za-z0-9._~+/=-]+")
_QUERY_SECRET = re.compile(
    r"(?i)([?&](?:access[_-]?token|api[_-]?key|password|secret|token)=)[^&#\s]+"
)
_ASSIGNED_SECRET = re.compile(
    r"(?i)\b((?:access[_-]?token|api[_-]?key|password|secret|token)\s*=\s*)"
    r"[^\s,;]+"
)


def redact(value: Any, *, key: str = "") -> Any:
    """Redact credential-shaped diagnostic context recursively."""

    if _SECRET_KEY.search(key):
        return "<redacted>"
    if isinstance(value, dict):
        return {str(k): redact(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return [redact(item) for item in value]
    if isinstance(value, str):
        redacted = _BEARER.sub(lambda m: f"{m.group(1)} <redacted>", value)
        redacted = _QUERY_SECRET.sub(lambda m: f"{m.group(1)}<redacted>", redacted)
        return _ASSIGNED_SECRET.sub(lambda m: f"{m.group(1)}<redacted>", redacted)
    return value



def missing_extra(
    extra: str,
    *,
    feature: str,
    install: str | None = None,
    capability: str | None = None,
) -> SchemaError:
    """Optional dependency missing — fail loud with install recipe."""
    cap = capability or extra
    cmd = install or f"python -m pip install 'arena[{extra}]'"
    return SchemaError(
        f"{feature} requires optional extra {extra!r}. Install with: {cmd}",
        code="CAPABILITY_MISSING",
        cause=f"optional extra {extra!r} is not installed",
        repair=(
            f"Install the missing extra, then retry: {cmd}. "
            f"Confirm with `arena doctor --capability {cap}`."
        ),
        context={"extra": extra, "capability": cap},
    )


def wrong_schema(
    *,
    expected: str | set[str] | tuple[str, ...],
    got: object,
    kind: str | None = None,
) -> SchemaError:
    """Wrong/missing schema id — list supported ids and migration next step."""
    if isinstance(expected, str):
        supported = [expected]
    else:
        supported = sorted(expected)
    supported_text = " or ".join(supported)
    kind_bit = f" for {kind}" if kind else ""
    return SchemaError(
        (
            f"unsupported schema{kind_bit}: got {got!r}; "
            f"supported: {supported_text}. "
            f"Set schema to one of the supported ids (creating a new artifact if migrating). "
            f"List schemas with `arena schema list`; Arena will not silently coerce versions."
        ),
        code="SCHEMA_VERSION_UNSUPPORTED",
        cause="manifest schema version is not supported by this Arena release",
        repair=(
            f"Update the `schema` field to {supported_text}. "
            "If migrating from another version, rewrite/export a new artifact "
            "(identity migration creates a new digest). Run `arena schema list` / `arena doctor` "
            "to see supported schemas."
        ),
        context={"expected": supported, "got": got, "kind": kind},
    )


def missing_digest(
    *,
    field: str,
    value: object = None,
    require_sha256_prefix: bool = True,
) -> SchemaError:
    got = f", got {value!r}" if value is not None else ""
    prefix = "sha256:<64-hex>" if require_sha256_prefix else "sha256:<64-hex> or raw 64-hex"
    return SchemaError(
        (
            f"{field} requires a content digest ({prefix}){got}. "
            "Compute with `sha256sum <file>` or use the digest returned by `arena push` / policy export. "
            "Arena will not invent or skip digests."
        ),
        code="DIGEST_MISSING",
        cause=f"{field} is missing or not a sha256 digest",
        repair=(
            f"Set {field} to a sha256 digest ({prefix}). "
            "For mirrored artifacts, copy the `#sha256:…` fragment from the URI returned by `arena push`."
        ),
        context={"field": field, "value": value},
    )


def bad_uri(
    message: str,
    *,
    scheme: str | None = None,
    example: str | None = None,
    repair: str | None = None,
) -> SchemaError:
    example_bit = f" Example: {example}." if example else ""
    default_repair = repair or (
        "Correct the URI grammar for the scheme, include required path/digest fragments, "
        "and retry. See `arena doctor` for store/task capability status."
    )
    return SchemaError(
        f"{message.rstrip('.')}." + example_bit,
        code="URI_INVALID",
        cause="URI failed Arena grammar checks",
        repair=default_repair,
        context={"scheme": scheme} if scheme else {},
    )


def diagnostic_from_exception(
    exc: BaseException,
    *,
    command: str | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    expected = isinstance(exc, ArenaError)
    code = exc.code if expected else "INTERNAL"
    category = exc.category if expected else "internal"
    message = str(exc) or type(exc).__name__
    cause = exc.cause if expected else type(exc).__name__
    default_repairs = {
        "usage": "Run the command with --help and use the documented argument grammar.",
        "schema_compatibility": (
            "Correct the reported field or use a schema/version listed by `arena doctor`."
        ),
        "integrity_authenticity": (
            "Do not use the artifact; obtain trusted bytes or the correct independent key."
        ),
        "external": (
            "Run `arena doctor --capability NAME`, repair availability, and retry safely."
        ),
        "incomplete_cancelled": (
            "Inspect the incomplete attempt ledger, then retry the whole operation."
        ),
        "internal": (
            "Re-run with --debug and report the redacted traceback with the command and Arena version."
        ),
    }
    repair = (
        exc.repair if expected and exc.repair else default_repairs.get(category)
    )
    context = exc.context if expected else {"exception_type": type(exc).__name__}
    result: dict[str, Any] = {
        "schema": "arena.diagnostic/v1",
        "ok": False,
        "code": code,
        "category": category,
        "message": redact(message),
        "cause": redact(cause),
        "repair": redact(repair),
        "docs_url": f"https://github.com/almanzalex/Arena/blob/main/docs/errors.md#{str(code).lower()}",
        "context": redact(context),
        "command": command,
    }
    if debug:
        result["debug"] = {"exception_type": type(exc).__name__}
    return result


def exit_code_for_exception(exc: BaseException) -> int:
    return exc.exit_code if isinstance(exc, ArenaError) else 70


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
