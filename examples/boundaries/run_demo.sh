#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
python3 examples/boundaries/run_demo.py --out examples/boundaries/.demo
