"""R-01…R-14 release-gate evidence collection (stream D).

This package collects *local* proof and assembles a skeleton index. It never
marks live Hugging Face, separately deployed OpenEnv, or release-CI Gimitest as
passed without real attached evidence files.
"""

from __future__ import annotations

from .collect_release_evidence import collect_release_evidence
from .gates import GATE_SPECS, MANDATORY_GATE_IDS

__all__ = [
    "GATE_SPECS",
    "MANDATORY_GATE_IDS",
    "collect_release_evidence",
]
