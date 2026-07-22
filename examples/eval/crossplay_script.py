#!/usr/bin/env python3
"""DEPRECATED — replaced by RLX 0.2 population + eval workflow (U-02).

Historically this script hand-rolled a 3-opponent RPS cross-play matrix.
Do not extend it. Use:

  rlx population create examples/eval/population.yaml --ref populations/rps-opp
  rlx eval run examples/eval/evaluation.yaml \\
    --policy candidate=<path> --population <digest>=<pop.yaml> \\
    --out ./eval-runs/crossplay
  rlx eval report ./eval-runs/crossplay --json
  rlx eval bundle ./eval-runs/crossplay --out ./bundles/crossplay

See docs/evaluation.md and examples/eval/README.md.
"""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "examples/eval/crossplay_script.py is retired (U-02).\n"
        "Use `rlx population create` + `rlx eval run|report|bundle` instead.\n"
        "See docs/evaluation.md.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
