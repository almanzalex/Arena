"""Evaluation metrics facade (re-exports registry cases)."""

from __future__ import annotations

from arena.plugins.metrics import (
    METRICS,
    MeanReturnMetric,
    PayoffMatrixMetric,
    WinRateMetric,
    detect_nontransitivity,
    register_builtins,
    register_metric,
)

__all__ = [
    "METRICS",
    "MeanReturnMetric",
    "PayoffMatrixMetric",
    "WinRateMetric",
    "detect_nontransitivity",
    "register_builtins",
    "register_metric",
]
