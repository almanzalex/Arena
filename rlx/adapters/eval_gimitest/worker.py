"""JSON worker for dependency-isolated Gimitest evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rlx.adapters.eval_gimitest import GimitestEvalProvider
from rlx.core.identity import canonical_json


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        raise SystemExit("usage: python -m rlx.adapters.eval_gimitest.worker REQUEST RESPONSE")
    request_path, response_path = map(Path, args)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("schema") != "rlx.eval-provider-request/v1":
        raise SystemExit("unsupported eval-provider request schema")
    suite = dict(request["suite"])
    result = GimitestEvalProvider()._run_in_process(
        suite,
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
    response_path.write_bytes(canonical_json(result) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
