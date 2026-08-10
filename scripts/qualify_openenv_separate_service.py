#!/usr/bin/env python3
"""Qualify Arena against a separately operated OpenEnv service (R-05).

Requires ``ARENA_OPENENV_BASE_URL`` (or ``--base-url``) pointing at a healthy
OpenEnv pilot. Does not start the service. Writes machine-readable evidence under
``docs/qualifications/openenv/`` by default.
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arena.adapters.task_openenv.service_recipe import (  # noqa: E402
    OPENENV_BASE_URL_ENV,
    SEPARATE_SERVICE_RECIPE,
    require_openenv_separate_service,
)
from arena.conformance.qualification import qualify_task_fixture  # noqa: E402
from arena.core.identity import digest_uri, sha256_file  # noqa: E402
from arena.core.manifests import dump_json, load_manifest  # noqa: E402
from arena.core.tasks import import_openenv_task, verify_task_equivalence  # noqa: E402


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _host_port(base_url: str) -> tuple[str, int | None]:
    parsed = urlparse(base_url)
    return parsed.hostname or "", parsed.port


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=None,
        help=f"OpenEnv service base URL (default: ${OPENENV_BASE_URL_ENV})",
    )
    parser.add_argument(
        "--env-path",
        default="arena/competitive_rps_v0",
        help="OpenEnv env path after the authority (default: arena/competitive_rps_v0)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs" / "qualifications" / "openenv",
        help="Directory for imported task YAML + qualification evidence JSON",
    )
    parser.add_argument(
        "--source-revision",
        default="openenv-0.4.1",
        help="Pinned OpenEnv revision recorded on the imported task",
    )
    parser.add_argument(
        "--transport",
        choices=("http", "https"),
        default=None,
        help="Override URI transport (default: scheme of --base-url)",
    )
    args = parser.parse_args(argv)

    try:
        base_url = require_openenv_separate_service(args.base_url)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    host, port = _host_port(base_url)
    if not host:
        print(f"invalid OpenEnv base URL: {base_url!r}", file=sys.stderr)
        print(SEPARATE_SERVICE_RECIPE, file=sys.stderr)
        return 2
    parsed = urlparse(base_url)
    transport = args.transport or parsed.scheme or "http"
    authority = host if port is None else f"{host}:{port}"
    source = f"openenv://{authority}/{args.env_path.lstrip('/')}?transport={transport}"

    imported_path = out_dir / "task-rps-openenv-separate.yaml"
    started = _utc_now()
    imported = import_openenv_task(
        source,
        name="task:rps-openenv-separate@1.0",
        out=imported_path,
        source_revision=args.source_revision,
    )
    packaging = imported.get("packaging") or {}
    pinned = str(packaging.get("base_url") or "").rstrip("/")
    if pinned != base_url.rstrip("/"):
        print(
            f"imported packaging.base_url {pinned!r} does not match service {base_url!r}",
            file=sys.stderr,
        )
        return 1

    native = load_manifest(ROOT / "examples/tasks/native-rps.yaml")
    suite = load_manifest(ROOT / "examples/tasks/rps-equivalence.yaml")
    equivalence = verify_task_equivalence(native, imported, suite)
    if not equivalence.get("ok"):
        dump_json(
            {
                "schema": "arena.openenv-separate-qualification/v1",
                "ok": False,
                "gate": "R-05",
                "error": "trace equivalence failed",
                "equivalence": equivalence,
                "started_at": started,
                "finished_at": _utc_now(),
            },
            out_dir / "R-05-openenv-separate-service.json",
        )
        print(json.dumps(equivalence, indent=2, sort_keys=True))
        return 1

    qualification_path = out_dir / "openenv-separate-qualification.json"
    qualification = qualify_task_fixture(
        imported_path,
        peer=ROOT / "examples/tasks/native-rps.yaml",
        trace_suite=ROOT / "examples/tasks/rps-equivalence.yaml",
        report_path=qualification_path,
    )
    if not qualification.get("ok"):
        print(json.dumps(qualification, indent=2, sort_keys=True))
        return 1

    evidence = {
        "schema": "arena.openenv-separate-qualification/v1",
        "gate": "R-05",
        "ok": True,
        "mode": "separate-service",
        "started_at": started,
        "finished_at": _utc_now(),
        "service": {
            "base_url": base_url,
            "host": host,
            "port": port,
            "env_path": args.env_path,
            "source_uri": source,
            "operator_hostname": socket.gethostname(),
            "platform": {
                "os": platform.system().lower(),
                "arch": platform.machine().lower(),
                "python": platform.python_version(),
            },
        },
        "client": {
            "connects_via": "packaging.base_url",
            "does_not_spawn_service": True,
            "env_var": OPENENV_BASE_URL_ENV,
        },
        "imported_task": str(imported_path),
        "schema_digest": packaging.get("schema_digest"),
        "shared_task_intent_digest": equivalence.get("shared_task_intent_digest"),
        "equivalence_ok": True,
        "qualification": {
            "ok": True,
            "path": str(qualification_path),
            "digest": digest_uri(sha256_file(qualification_path)),
            "adapter": qualification.get("adapter"),
            "kind": qualification.get("kind"),
        },
        "stable_claim": {
            "support_matrix_flipped": False,
            "note": (
                "Evidence only. support-matrix.json remains preview until release "
                "owners attach this report and promote the capability."
            ),
        },
        "recipe": SEPARATE_SERVICE_RECIPE.strip().splitlines(),
    }
    evidence_path = out_dir / "R-05-openenv-separate-service.json"
    dump_json(evidence, evidence_path)
    print(
        json.dumps(
            {
                "ok": True,
                "evidence": str(evidence_path),
                "qualification": str(qualification_path),
                "base_url": base_url,
                "shared_task_intent_digest": evidence["shared_task_intent_digest"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
