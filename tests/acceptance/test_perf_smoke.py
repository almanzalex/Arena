"""Lightweight performance smoke for export + verify + match.

This is intentionally not the R-10 / T-1200 performance gate. It only fails on
catastrophic regressions (default >10× checked-in baseline) or absurd absolute
ceilings, so default CI stays green under normal runner variance.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import pytest

pytest.importorskip("torch")
pytest.importorskip("pettingzoo")

from arena.adapters.policy_custom_torch import verify_bundle_self  # noqa: E402
from arena.conformance.fixtures import (  # noqa: E402
    build_f1_deterministic,
    build_rps_policy,
)
from arena.core.sdk import Match, Policy, Task  # noqa: E402

_BASELINE_PATH = Path(__file__).resolve().parents[1] / "baselines" / "perf_smoke.json"
_PILOT = "arena/competitive_rps_v0"
_REPEATS = 3


def _timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def _median_seconds(samples: list[float]) -> float:
    return float(statistics.median(samples))


def _load_or_bootstrap_baseline(measured: dict[str, float]) -> dict[str, Any]:
    """Load checked-in baseline, or write one from measured medians when missing.

    Maintainers can force a rewrite with ARENA_PERF_SMOKE_UPDATE_BASELINE=1.
    Rewrites keep the existing loose ceilings / regression factor when present.
    """
    update = os.environ.get("ARENA_PERF_SMOKE_UPDATE_BASELINE", "").strip() in {
        "1",
        "true",
        "yes",
    }
    if _BASELINE_PATH.exists() and not update:
        return json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))

    prior: dict[str, Any] = {}
    if _BASELINE_PATH.exists():
        prior = json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))

    # Seed generous baselines (~50× local medians on a healthy laptop) so CI
    # machines never trip the 10× relative check under ordinary load.
    seeded = {name: max(0.5, value * 50.0) for name, value in measured.items()}
    payload = {
        "schema": "arena.perf-smoke-baseline/v1",
        "description": prior.get(
            "description",
            "Lightweight export/verify/match smoke baselines. Not R-10.",
        ),
        "regression_factor": float(prior.get("regression_factor", 10.0)),
        "absolute_ceilings_seconds": prior.get(
            "absolute_ceilings_seconds",
            {"export": 60.0, "verify": 30.0, "match": 120.0},
        ),
        "baselines_seconds": seeded,
        "fixture": prior.get(
            "fixture",
            {
                "export_verify": "build_f1_deterministic (tiny mlp)",
                "match": "build_rps_policy + Match 2 seeds max_cycles=2",
            },
        ),
    }
    _BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _BASELINE_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _assert_within_bounds(
    name: str,
    measured: float,
    *,
    baseline: float,
    ceiling: float,
    factor: float,
) -> None:
    relative_limit = baseline * factor
    limit = min(relative_limit, ceiling)
    assert measured <= limit, (
        f"perf-smoke {name} catastrophic regression: "
        f"measured={measured:.4f}s limit={limit:.4f}s "
        f"(baseline={baseline:.4f}s ×{factor:g}, absolute_ceiling={ceiling:.4f}s). "
        "This is a smoke check, not R-10; investigate only if the slowdown is real."
    )


@pytest.mark.acceptance
@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_export_verify_match_perf_smoke(tmp_path: Path) -> None:
    """Time tiny export, verify, and match; fail only on catastrophic regressions."""
    warm = tmp_path / "warm"
    warm.mkdir()
    warm_bundle = build_f1_deterministic(warm / "f1")
    assert verify_bundle_self(warm_bundle, allow_self_consistency=True)["ok"]
    build_rps_policy(warm / "p0", role="player_0", seed=1)
    build_rps_policy(warm / "p1", role="player_1", seed=2)

    export_samples: list[float] = []
    verify_samples: list[float] = []
    match_samples: list[float] = []

    for i in range(_REPEATS):
        run = tmp_path / f"run{i}"
        run.mkdir()

        bundle, export_s = _timed(lambda i=i, run=run: build_f1_deterministic(run / "f1", seed=i))
        export_samples.append(export_s)

        result, verify_s = _timed(
            lambda bundle=bundle: verify_bundle_self(bundle, allow_self_consistency=True)
        )
        assert result["ok"], result
        verify_samples.append(verify_s)

        p0 = build_rps_policy(run / "p0", role="player_0", seed=1)
        p1 = build_rps_policy(run / "p1", role="player_1", seed=2)
        match = Match(
            task=Task.load(
                {
                    "adapter": "pettingzoo-parallel",
                    "env": _PILOT,
                    "config": {"max_cycles": 2},
                }
            ),
            assignments={
                "player_0": Policy.load(p0),
                "player_1": Policy.load(p1),
            },
            action_mode="deterministic",
            failure_policy={
                "timeout_seconds": 30,
                "retain_incomplete": True,
                "retry": 0,
            },
        )
        out = run / "out"
        record, match_s = _timed(lambda match=match, out=out: match.run(seeds=[0, 1], record=False, out=out))
        assert record["outcome"]["episodes_completed"] == 2
        match_samples.append(match_s)

    measured = {
        "export": _median_seconds(export_samples),
        "verify": _median_seconds(verify_samples),
        "match": _median_seconds(match_samples),
    }
    baseline_doc = _load_or_bootstrap_baseline(measured)
    factor = float(baseline_doc.get("regression_factor", 10.0))
    baselines = baseline_doc["baselines_seconds"]
    ceilings = baseline_doc["absolute_ceilings_seconds"]

    for name, value in measured.items():
        _assert_within_bounds(
            name,
            value,
            baseline=float(baselines[name]),
            ceiling=float(ceilings[name]),
            factor=factor,
        )
