"""Strict JSON worker for one hard-budget evaluation cell."""

from __future__ import annotations

import sys
from pathlib import Path

from arena import __version__
from arena.core.identity import canonical_json, digest_uri, sha256_bytes
from arena.core.io import atomic_write_bytes
from arena.core.manifests import load_manifest
from arena.runtime.evaluation import _execute_evaluation_cell


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        raise SystemExit("usage: python -m arena.runtime.eval_worker REQUEST RESPONSE")
    request_path, response_path = map(Path, args)
    request = load_manifest(request_path, max_bytes=64 * 1024 * 1024)
    if request.get("schema") != "arena.eval-cell-request/v1":
        raise SystemExit("unsupported eval-cell request schema")
    for field in ("request_id", "request_digest"):
        if not isinstance(request.get(field), str) or not request[field]:
            raise SystemExit(f"eval-cell request missing {field}")
    digest_input = {
        key: value for key, value in request.items() if key != "request_digest"
    }
    actual_request_digest = digest_uri(
        sha256_bytes(canonical_json(digest_input))
    )
    if request["request_digest"] != actual_request_digest:
        raise SystemExit("eval-cell request digest mismatch")
    result = _execute_evaluation_cell(
        cell=dict(request["cell"]),
        suite=dict(request["suite"]),
        task_spec=dict(request["task_spec"]),
        task_info=dict(request["task_info"]),
        task_digest=str(request["task_digest"]),
        provider_lineage=dict(request["provider_lineage"]),
        policy_index={
            str(key): Path(value)
            for key, value in dict(request.get("policy_index") or {}).items()
        },
        run_root=Path(request["run_root"]),
        record=bool(request.get("record", True)),
    )
    response = {
        "schema": "arena.eval-cell-response/v1",
        "ok": True,
        "request_id": request["request_id"],
        "request_digest": request["request_digest"],
        "worker": {
            "arena_version": __version__,
            "python": sys.version.split()[0],
        },
        "result": result,
    }
    atomic_write_bytes(response_path, canonical_json(response) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
