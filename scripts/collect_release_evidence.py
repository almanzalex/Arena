#!/usr/bin/env python3
"""CLI entry for the R-gates release-evidence collector.

Prefer importing ``r_gates`` from ``scripts/`` in tests. Operators can run:

    python scripts/collect_release_evidence.py
    python scripts/collect_release_evidence.py --attach R-01=/path/ci.json
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from r_gates.collect_release_evidence import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
