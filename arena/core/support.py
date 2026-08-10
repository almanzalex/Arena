"""Machine-readable release support and local capability preflight."""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from importlib import resources
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from arena.core.errors import SchemaError
from arena.core.identity import canonical_json, digest_uri, sha256_bytes
from arena.core.supervisor import run_supervised

SUPPORT_SCHEMA = "arena.support-matrix/v1"
DOCTOR_SCHEMA = "arena.doctor/v1"
DOCTOR_TOP_LEVEL_KEYS = (
    "schema",
    "release",
    "schema_registry_digest",
    "platform",
    "platform_status",
    "ok",
    "summary",
    "capabilities",
)
DOCTOR_CAPABILITY_KEYS = (
    "name",
    "release_status",
    "target_status",
    "required_for_1_0",
    "local_status",
    "usable_today",
    "platform_status",
    "platform",
    "extra",
    "extra_installed",
    "installed_distributions",
    "missing_distributions",
    "installed_executables",
    "missing_executables",
    "missing",
    "isolated_python_env",
    "isolated_python",
    "isolated_probe",
    "credentials_required",
    "authentication_attempted",
    "evidence",
    "stable_requires",
    "qualify",
    "repair",
)
DOCTOR_SUMMARY_KEYS = (
    "usable_today_stable",
    "usable_today_preview",
    "locally_unqualified",
    "preview_missing_for_stable",
)


def load_support_matrix() -> dict[str, Any]:
    payload = resources.files("arena").joinpath("support-matrix.json").read_text(
        encoding="utf-8"
    )
    data = json.loads(payload)
    if data.get("schema") != SUPPORT_SCHEMA:
        raise SchemaError("installed Arena support matrix has an unsupported schema")
    return data


def load_schema_registry() -> dict[str, Any]:
    payload = resources.files("arena").joinpath("schema-registry.json").read_text(
        encoding="utf-8"
    )
    data = json.loads(payload)
    if data.get("schema") != "arena.schema-registry/v1":
        raise SchemaError("installed Arena schema registry has an unsupported schema")
    schemas = data.get("schemas")
    if not isinstance(schemas, list) or not schemas:
        raise SchemaError("installed Arena schema registry is empty")
    ids = [item.get("id") for item in schemas if isinstance(item, dict)]
    if len(ids) != len(schemas) or len(set(ids)) != len(ids):
        raise SchemaError("installed Arena schema registry has duplicate or invalid ids")
    return data


def _distribution_available(name: str) -> bool:
    try:
        return find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _platform_status(matrix: dict[str, Any]) -> tuple[str, dict[str, str]]:
    current = {
        "os": sys.platform.lower(),
        "arch": platform.machine().lower(),
        "python": ".".join(map(str, sys.version_info[:3])),
    }
    os_name = "darwin" if current["os"] == "darwin" else (
        "linux" if current["os"].startswith("linux") else current["os"]
    )
    current["os"] = os_name
    supported_python = sys.version_info[:2] in {(3, 12), (3, 13)}
    if not supported_python:
        return "unqualified", current
    # Prefer a stable claim; experimental CI scaffolding never upgrades to stable.
    for status in ("stable", "experimental"):
        matched = any(
            row.get("os") == os_name
            and str(row.get("arch", "")).lower() == current["arch"]
            and row.get("status") == status
            for row in matrix.get("platforms", [])
        )
        if matched:
            return status, current
    return "unqualified", current


def _probe_isolated_python(
    python: Path,
    *,
    distributions: list[str],
    release: str,
) -> dict[str, Any]:
    if not python.is_file() or not os.access(python, os.X_OK):
        return {"status": "unavailable", "versions": {}, "missing": distributions}
    probe = (
        "import json\n"
        "from importlib.metadata import PackageNotFoundError, version\n"
        f"names={distributions!r}\n"
        "versions={}\n"
        "missing=[]\n"
        "for name in names:\n"
        "  try: versions[name]=version(name)\n"
        "  except PackageNotFoundError: missing.append(name)\n"
        "print(json.dumps({'versions':versions,'missing':missing},sort_keys=True))\n"
    )
    try:
        completed = run_supervised(
            [str(python), "-I", "-c", probe],
            timeout_seconds=5,
            max_stdout_bytes=65_536,
            max_stderr_bytes=65_536,
        )
        payload = json.loads(completed.stdout) if completed.returncode == 0 else {}
    except Exception as exc:  # noqa: BLE001 - doctor converts probe faults to status
        return {
            "status": "failed",
            "versions": {},
            "missing": distributions,
            "reason": type(exc).__name__,
        }
    versions = payload.get("versions") if isinstance(payload, dict) else {}
    missing = payload.get("missing") if isinstance(payload, dict) else distributions
    if not isinstance(versions, dict) or not isinstance(missing, list):
        return {
            "status": "failed",
            "versions": {},
            "missing": distributions,
            "reason": "invalid probe response",
        }
    version_match = versions.get("arena") == release
    return {
        "status": "ready" if not missing and version_match else "incompatible",
        "versions": versions,
        "missing": missing,
        "arena_version_match": version_match,
    }


def _usable_today(*, release_status: str, local_status: str) -> str:
    """Answer 'can I use this today?' without conflating local deps with stable claims."""
    if local_status != "ready":
        return "no"
    if release_status == "stable":
        return "stable"
    if release_status == "preview":
        return "preview"
    return "no"


def _extra_installed(spec: dict[str, Any]) -> bool | None:
    """True when every declared distribution/exec for an extra is present.

    Returns None when the capability has no pip extra (core/file/oci).
    Isolated extras are resolved after the worker probe in capability_report.
    """
    extra = spec.get("extra")
    if extra is None:
        return None
    if spec.get("isolated_python_env"):
        return None
    dists = [str(item) for item in spec.get("distributions", [])]
    exes = [str(item) for item in spec.get("executables", [])]
    if not dists and not exes:
        return True
    return all(_distribution_available(name) for name in dists) and all(
        shutil.which(name) is not None for name in exes
    )


def capability_report(name: str, *, matrix: dict[str, Any] | None = None) -> dict[str, Any]:
    matrix = matrix or load_support_matrix()
    capabilities = matrix.get("capabilities") or {}
    if name not in capabilities:
        raise SchemaError(
            f"unknown capability {name!r}; known: {', '.join(sorted(capabilities))}",
            code="CAPABILITY_UNKNOWN",
            repair="Run `arena doctor` to list installed and supported capabilities.",
        )
    spec = dict(capabilities[name])
    required_distributions = [str(item) for item in spec.get("distributions", [])]
    required_executables = [str(item) for item in spec.get("executables", [])]
    installed_distributions = [
        item for item in required_distributions if _distribution_available(item)
    ]
    missing_distributions = [
        item for item in required_distributions if item not in installed_distributions
    ]
    installed_executables = [
        item for item in required_executables if shutil.which(item) is not None
    ]
    missing_executables = [
        item for item in required_executables if item not in installed_executables
    ]
    isolated_env = spec.get("isolated_python_env")
    isolated_python = os.environ.get(str(isolated_env), "") if isolated_env else ""
    isolated_ready = True
    isolated_probe = None
    if isolated_env:
        isolated_probe = (
            _probe_isolated_python(
                Path(isolated_python),
                distributions=[
                    str(item) for item in spec.get("isolated_distributions", [])
                ],
                release=str(matrix.get("release", "")),
            )
            if isolated_python
            else {
                "status": "unavailable",
                "versions": {},
                "missing": list(spec.get("isolated_distributions", [])),
            }
        )
        isolated_ready = isolated_probe["status"] == "ready"
    platform_state, current = _platform_status(matrix)
    ready = (
        platform_state == "stable"
        and not missing_distributions
        and not missing_executables
        and isolated_ready
    )
    local_status = "ready" if ready else "locally-unqualified"
    release_status = str(spec.get("status", "unsupported"))
    usable_today = _usable_today(
        release_status=release_status, local_status=local_status
    )
    extra = spec.get("extra")
    if isolated_env:
        extra_installed = isolated_ready if extra is not None else None
    else:
        extra_installed = _extra_installed(spec)

    missing: list[str] = []
    missing.extend(f"distribution:{item}" for item in missing_distributions)
    missing.extend(f"executable:{item}" for item in missing_executables)
    if isolated_env and not isolated_ready:
        probe_missing = (
            isolated_probe.get("missing") if isinstance(isolated_probe, dict) else None
        )
        if isinstance(probe_missing, list) and probe_missing:
            missing.extend(f"isolated:{item}" for item in probe_missing)
        else:
            missing.append(f"env:{isolated_env}")

    repair = None
    if not ready:
        repair = spec.get("install")
        if platform_state != "stable":
            repair = (
                f"Arena has no stable 1.0 claim for {current['os']}/{current['arch']} "
                f"on Python {current['python']} (platform status: {platform_state})."
            )
        elif isolated_env and not isolated_ready:
            repair = (
                f"Create the isolated worker and set {isolated_env} to its absolute "
                "Python executable. " + str(spec.get("install") or "")
            )

    return {
        "name": name,
        "release_status": release_status,
        "target_status": spec.get("target_status"),
        "required_for_1_0": bool(spec.get("required_for_1_0", False)),
        "local_status": local_status,
        "usable_today": usable_today,
        "platform_status": platform_state,
        "platform": current,
        "extra": extra,
        "extra_installed": extra_installed,
        "installed_distributions": installed_distributions,
        "missing_distributions": missing_distributions,
        "installed_executables": installed_executables,
        "missing_executables": missing_executables,
        "missing": missing,
        "isolated_python_env": isolated_env,
        "isolated_python": isolated_python or None,
        "isolated_probe": isolated_probe,
        "credentials_required": bool(spec.get("credentials", False)),
        "authentication_attempted": False,
        "evidence": spec.get("evidence"),
        "stable_requires": spec.get("stable_requires"),
        "qualify": spec.get("qualify"),
        "repair": repair,
    }


def doctor_report(capability: str | None = None) -> dict[str, Any]:
    matrix = load_support_matrix()
    registry = load_schema_registry()
    platform_state, current = _platform_status(matrix)
    names = [capability] if capability else sorted(matrix["capabilities"])
    reports = [capability_report(name, matrix=matrix) for name in names]
    summary = {
        "usable_today_stable": [
            item["name"] for item in reports if item["usable_today"] == "stable"
        ],
        "usable_today_preview": [
            item["name"] for item in reports if item["usable_today"] == "preview"
        ],
        "locally_unqualified": [
            item["name"] for item in reports if item["local_status"] != "ready"
        ],
        "preview_missing_for_stable": [
            {
                "name": item["name"],
                "stable_requires": item.get("stable_requires"),
                "qualify": item.get("qualify"),
                "missing": item.get("missing") or [],
                "credentials_required": item.get("credentials_required"),
                "local_status": item.get("local_status"),
            }
            for item in reports
            if item["release_status"] == "preview"
        ],
    }
    return {
        "schema": DOCTOR_SCHEMA,
        "release": matrix.get("release"),
        "schema_registry_digest": digest_uri(
            sha256_bytes(canonical_json(registry))
        ),
        "platform": current,
        "platform_status": platform_state,
        "ok": all(item["local_status"] == "ready" for item in reports),
        "summary": summary,
        "capabilities": reports,
    }


def format_doctor_human(report: dict[str, Any]) -> str:
    """Readable doctor summary for labs deciding what they can use today."""
    lines: list[str] = []
    platform_info = report.get("platform") or {}
    lines.append(f"Arena doctor {report.get('release')}")
    lines.append(
        "Platform: "
        f"{platform_info.get('os')}/{platform_info.get('arch')} "
        f"Python {platform_info.get('python')} "
        f"({report.get('platform_status')})"
    )
    lines.append(f"Schema registry: {report.get('schema_registry_digest')}")
    lines.append("")

    summary = report.get("summary") or {}
    stable = summary.get("usable_today_stable") or []
    preview = summary.get("usable_today_preview") or []
    blocked = summary.get("locally_unqualified") or []

    lines.append("Usable today (stable claim):")
    if stable:
        for name in stable:
            lines.append(f"  + {name}")
    else:
        lines.append("  (none in this report)")

    lines.append("")
    lines.append("Usable today (preview only — not a 1.0 stable claim):")
    if preview:
        caps = {item["name"]: item for item in report.get("capabilities") or []}
        for name in preview:
            item = caps.get(name, {})
            bits = ["deps ready"]
            if item.get("credentials_required"):
                bits.append("credentials required for live use")
            if item.get("extra"):
                state = "installed" if item.get("extra_installed") else "incomplete"
                bits.append(f"extra[{item['extra']}]={state}")
            lines.append(f"  ~ {name}  ({'; '.join(bits)})")
    else:
        lines.append("  (none in this report)")

    lines.append("")
    lines.append("Not ready locally:")
    if blocked:
        caps = {item["name"]: item for item in report.get("capabilities") or []}
        for name in blocked:
            item = caps.get(name, {})
            detail = (
                item.get("repair")
                or ", ".join(item.get("missing") or [])
                or "unqualified"
            )
            lines.append(f"  x {name}")
            lines.append(f"      {detail}")
    else:
        lines.append("  (none in this report)")

    preview_gaps = summary.get("preview_missing_for_stable") or []
    if preview_gaps:
        lines.append("")
        lines.append("What is missing to promote preview → stable:")
        for gap in preview_gaps:
            lines.append(f"  {gap['name']}:")
            if gap.get("missing"):
                lines.append(f"    local gaps: {', '.join(gap['missing'])}")
            elif gap.get("local_status") == "ready":
                lines.append(
                    "    local deps: ready (still preview until release evidence)"
                )
            if gap.get("stable_requires"):
                lines.append(f"    stable requires: {gap['stable_requires']}")
            if gap.get("qualify"):
                lines.append(f"    how to qualify: {gap['qualify']}")
            if gap.get("credentials_required"):
                lines.append(
                    "    credentials: required for live qualification "
                    "(doctor does not authenticate)"
                )

    lines.append("")
    lines.append(
        "Legend: local_status=ready means dependencies are present; "
        "usable_today=stable is the only release-stable claim. "
        "Doctor never authenticates."
    )
    return "\n".join(lines) + "\n"
