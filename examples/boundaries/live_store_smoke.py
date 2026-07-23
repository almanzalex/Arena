"""Opt-in authenticated smoke for HF, OCI, W&B, or MLflow stores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from rlx.conformance.qualification import qualify_store


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Push and pull one policy through a real credentialed backend"
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", help="hf://, oci://, wandb://, or mlflow:// URI")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Qualification JSON (default: <out>.qualification.json)",
    )
    args = parser.parse_args()
    parsed = urlparse(args.destination)
    if parsed.scheme not in {"hf", "oci", "wandb", "mlflow"}:
        parser.error("destination must use hf, oci, wandb, or mlflow")
    if "simulate" in parse_qs(parsed.query):
        parser.error("live smoke refuses ?simulate=; use run_demo.py for local simulation")

    report_path = args.report or args.out.with_name(
        args.out.name + ".qualification.json"
    )
    report = qualify_store(
        args.source,
        destination=args.destination,
        report_path=report_path,
        restored_out=args.out,
    )
    if report["mode"] != "live":
        raise RuntimeError("live store smoke produced non-live evidence")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
