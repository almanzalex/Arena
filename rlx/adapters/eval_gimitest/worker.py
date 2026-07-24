"""JSON worker for dependency-isolated Gimitest evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

from rlx import __version__
from rlx.adapters.eval_gimitest import GimitestEvalProvider
from rlx.core.identity import canonical_json
from rlx.core.io import atomic_write_bytes
from rlx.core.manifests import load_manifest


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        raise SystemExit("usage: python -m rlx.adapters.eval_gimitest.worker REQUEST RESPONSE")
    request_path, response_path = map(Path, args)
    request = load_manifest(request_path, max_bytes=16 * 1024 * 1024)
    if request.get("schema") != "rlx.eval-provider-request/v1":
        raise SystemExit("unsupported eval-provider request schema")
    for field in ("request_id", "request_digest"):
        if not isinstance(request.get(field), str) or not request[field]:
            raise SystemExit(f"eval-provider request missing {field}")
    digest_input = {key: value for key, value in request.items() if key != "request_digest"}
    from rlx.core.identity import digest_uri, sha256_bytes

    if request["request_digest"] != digest_uri(
        sha256_bytes(canonical_json(digest_input))
    ):
        raise SystemExit("eval-provider request digest mismatch")
    suite = dict(request["suite"])
    result = GimitestEvalProvider()._run_in_process(
        suite,
        _worker_lineage={
            "protocol": "rlx.eval-provider/v1",
            "rlx_version": __version__,
            "python": {
                "executable": str(Path(sys.executable).resolve()),
                "version": sys.version.split()[0],
            },
        },
        identity_suite=request["identity_suite"],
        policy_index={
            str(key): Path(value)
            for key, value in dict(request.get("policy_index") or {}).items()
        },
        populations=request.get("populations"),
        store=None,
        out_dir=Path(request["out_dir"]),
        workers=int(request.get("workers", 1)),
        record=bool(request.get("record", True)),
    )
    response = {
        "schema": "rlx.eval-provider-response/v1",
        "ok": True,
        "request_id": request["request_id"],
        "request_digest": request["request_digest"],
        "rlx_version": __version__,
        "python": {
            "executable": str(Path(sys.executable).resolve()),
            "version": sys.version.split()[0],
        },
        "result": result,
    }
    atomic_write_bytes(response_path, canonical_json(response) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
