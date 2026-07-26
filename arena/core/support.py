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
    supported_python = sys.version_info[:2] in {(3, 12), (3, 13)}
    matched = any(
        row.get("os") == os_name
        and str(row.get("arch", "")).lower() == current["arch"]
        and row.get("status") == "stable"
        for row in matrix.get("platforms", [])
    )
    return ("stable" if matched and supported_python else "unqualified"), current


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
    missing_distributions = [
        item for item in spec.get("distributions", []) if not _distribution_available(item)
    ]
    missing_executables = [
        item for item in spec.get("executables", []) if shutil.which(item) is None
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
    repair = None
    if not ready:
        repair = spec.get("install")
        if platform_state != "stable":
            repair = (
                f"Arena has no stable 1.0 claim for {current['os']}/{current['arch']} "
                f"on Python {current['python']}."
            )
        elif isolated_env and not isolated_ready:
            repair = (
                f"Create the isolated worker and set {isolated_env} to its absolute "
                "Python executable. " + str(spec.get("install") or "")
            )
    return {
        "name": name,
        "release_status": spec.get("status", "unsupported"),
        "target_status": spec.get("target_status"),
        "required_for_1_0": bool(spec.get("required_for_1_0", False)),
        "local_status": local_status,
        "platform_status": platform_state,
        "platform": current,
        "missing_distributions": missing_distributions,
        "missing_executables": missing_executables,
        "isolated_python_env": isolated_env,
        "isolated_python": isolated_python or None,
        "isolated_probe": isolated_probe,
        "credentials_required": bool(spec.get("credentials", False)),
        "authentication_attempted": False,
        "repair": repair,
    }


def doctor_report(capability: str | None = None) -> dict[str, Any]:
    matrix = load_support_matrix()
    registry = load_schema_registry()
    names = [capability] if capability else sorted(matrix["capabilities"])
    reports = [capability_report(name, matrix=matrix) for name in names]
    return {
        "schema": "arena.doctor/v1",
        "release": matrix.get("release"),
        "schema_registry_digest": digest_uri(
            sha256_bytes(canonical_json(registry))
        ),
        "ok": all(item["local_status"] == "ready" for item in reports),
        "capabilities": reports,
    }
