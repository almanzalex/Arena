"""Evaluation metrics as registry cases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from arena.core.registry import Registry


class EvalMetric(Protocol):
    kind: str

    def compute(self, cells: list[dict[str, Any]], *, role: str | None = None) -> dict[str, Any]:
        ...


def _returns_by_role(cell: dict[str, Any], role: str) -> list[float]:
    returns = []
    for ep in cell.get("episodes", []):
        rewards = ep.get("returns", {})
        if role in rewards:
            returns.append(float(rewards[role]))
    return returns


def _wilson_interval(wins: int, n: int, z: float = 1.96) -> dict[str, float]:
    if n <= 0:
        return {"low": 0.0, "high": 0.0, "n": 0}
    p = wins / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return {"low": float(center - margin), "high": float(center + margin), "n": n}


def _bootstrap_ci(
    values: list[float], *, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05
) -> dict[str, float]:
    """Percentile bootstrap CI for the mean (MET uncertainty)."""
    if not values:
        return {"low": 0.0, "high": 0.0, "n": 0, "n_boot": 0}
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = arr.size
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        means[i] = float(sample.mean())
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"low": float(low), "high": float(high), "n": int(n), "n_boot": n_boot}


@dataclass
class MeanReturnMetric:
    kind: str = "mean_return"

    def compute(self, cells: list[dict[str, Any]], *, role: str | None = None) -> dict[str, Any]:
        role = role or _primary_role(cells)
        values = []
        failures = 0
        evidence = []
        for cell in cells:
            failures += int(cell.get("failures", 0))
            values.extend(_returns_by_role(cell, role))
            evidence.extend(cell.get("evidence_refs", []))
        arr = np.asarray(values, dtype=np.float64)
        n = int(arr.size)
        mean = float(arr.mean()) if n else 0.0
        se = float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        return {
            "kind": self.kind,
            "role": role,
            "mean": mean,
            "se": se,
            "n": n,
            "failures": failures,
            "ci": _bootstrap_ci(values, seed=0),
            "evidence_refs": evidence,
        }


@dataclass
class WinRateMetric:
    kind: str = "win_rate"

    def compute(self, cells: list[dict[str, Any]], *, role: str | None = None) -> dict[str, Any]:
        role = role or _primary_role(cells)
        wins = 0
        n = 0
        failures = 0
        evidence = []
        for cell in cells:
            failures += int(cell.get("failures", 0))
            evidence.extend(cell.get("evidence_refs", []))
            for ep in cell.get("episodes", []):
                outcome = ep.get("outcomes", {})
                if role in outcome:
                    n += 1
                    if outcome[role] == "win":
                        wins += 1
                else:
                    # Infer from returns when explicit outcome missing.
                    returns = ep.get("returns", {})
                    if role in returns and returns:
                        n += 1
                        if float(returns[role]) > 0:
                            wins += 1
        return {
            "kind": self.kind,
            "role": role,
            "win_rate": (wins / n) if n else 0.0,
            "wins": wins,
            "n": n,
            "failures": failures,
            "ci": _wilson_interval(wins, n),
            "evidence_refs": evidence,
        }


@dataclass
class PayoffMatrixMetric:
    kind: str = "payoff_matrix"

    def compute(self, cells: list[dict[str, Any]], *, role: str | None = None) -> dict[str, Any]:
        del role
        # Rows = candidate policy, cols = opponent policy, value = mean return for candidate role.
        row_keys: list[str] = []
        col_keys: list[str] = []
        buckets: dict[tuple[str, str], list[float]] = {}
        evidence: dict[str, list[str]] = {}
        failures = 0
        for cell in cells:
            cand = cell.get("candidate_policy") or cell.get("assignments", {}).get("player_0")
            opp = cell.get("opponent_policy") or cell.get("assignments", {}).get("player_1")
            if not cand or not opp:
                continue
            cand_s, opp_s = str(cand), str(opp)
            if cand_s not in row_keys:
                row_keys.append(cand_s)
            if opp_s not in col_keys:
                col_keys.append(opp_s)
            key = (cand_s, opp_s)
            buckets.setdefault(key, [])
            evidence.setdefault(f"{cand_s}|{opp_s}", [])
            evidence[f"{cand_s}|{opp_s}"].extend(cell.get("evidence_refs", []))
            failures += int(cell.get("failures", 0))
            for ep in cell.get("episodes", []):
                rets = ep.get("returns", {})
                # Prefer player_0 as candidate when present.
                if "player_0" in rets:
                    buckets[key].append(float(rets["player_0"]))
                elif rets:
                    buckets[key].append(float(next(iter(rets.values()))))
        matrix = []
        for r in row_keys:
            row = []
            for c in col_keys:
                vals = buckets.get((r, c), [])
                row.append(float(np.mean(vals)) if vals else None)
            matrix.append(row)
        warning = detect_nontransitivity(matrix, row_keys, col_keys)
        return {
            "kind": self.kind,
            "rows": row_keys,
            "cols": col_keys,
            "matrix": matrix,
            "failures": failures,
            "evidence_refs": evidence,
            "nontransitivity_warning": warning,
            # Never emit a silent scalar ranking as the primary result.
            "ranking": None if warning else _trivial_ranking(matrix, row_keys),
        }


def detect_nontransitivity(
    matrix: list[list[float | None]],
    rows: list[str],
    cols: list[str] | None = None,
) -> str | None:
    """Detect a directed cycle among pairwise advantages (rock-paper-scissors).

    ``matrix[i][j]`` is the mean return for row policy ``rows[i]`` against
    column policy ``cols[j]``. When ``cols`` is omitted, the matrix is treated
    as square over ``rows``.
    """
    cols = list(cols) if cols is not None else list(rows)
    labels = list(dict.fromkeys([*rows, *cols]))
    n = len(labels)
    if n < 3:
        return None
    index = {lab: i for i, lab in enumerate(labels)}
    edges = {i: set() for i in range(n)}
    for i, ri in enumerate(rows):
        for j, cj in enumerate(cols):
            if ri == cj:
                continue
            v = matrix[i][j] if i < len(matrix) and j < len(matrix[i]) else None
            if v is not None and v > 0:
                edges[index[ri]].add(index[cj])
    # DFS cycle detection
    color = [0] * n

    def dfs(u: int, stack: list[int]) -> list[int] | None:
        color[u] = 1
        stack.append(u)
        for v in edges[u]:
            if color[v] == 0:
                cyc = dfs(v, stack)
                if cyc:
                    return cyc
            elif color[v] == 1:
                return stack[stack.index(v) :] + [v]
        stack.pop()
        color[u] = 2
        return None

    for i in range(n):
        if color[i] == 0:
            cyc = dfs(i, [])
            if cyc:
                names = [labels[k] for k in cyc]
                return (
                    "Non-transitive payoff cycle detected: "
                    + " -> ".join(names)
                    + ". Matrix retained as primary; refusing silent single ranking."
                )
    return None


def _trivial_ranking(matrix: list[list[float | None]], labels: list[str]) -> list[str] | None:
    scores = []
    for i, lab in enumerate(labels):
        vals = [v for v in (matrix[i] if i < len(matrix) else []) if v is not None]
        scores.append((float(np.mean(vals)) if vals else 0.0, lab))
    scores.sort(reverse=True)
    return [lab for _, lab in scores]


def _primary_role(cells: list[dict[str, Any]]) -> str:
    for cell in cells:
        for ep in cell.get("episodes", []):
            rets = ep.get("returns", {})
            if rets:
                return str(next(iter(rets.keys())))
    return "player_0"


METRICS: Registry[EvalMetric] = Registry(
    "eval_metric",
    interface="EvalMetric",
    register_via="arena.plugins.metrics.register_metric",
    tests="mean_return/win_rate/payoff_matrix + nontransitivity fixture",
)


def register_metric(kind: str, metric: EvalMetric, *, replace: bool = False) -> EvalMetric:
    return METRICS.register(kind, metric, replace=replace)


def register_builtins() -> None:
    register_metric("mean_return", MeanReturnMetric(), replace=True)
    register_metric("win_rate", WinRateMetric(), replace=True)
    register_metric("payoff_matrix", PayoffMatrixMetric(), replace=True)
