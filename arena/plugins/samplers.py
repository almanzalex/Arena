"""Population sampling strategies for evaluation expansion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from arena.core.errors import SchemaError
from arena.core.registry import Registry


class PopulationSampler(Protocol):
    kind: str

    def sample(
        self,
        members: list[dict[str, Any]],
        *,
        seed: int,
        stream: str,
        n: int = 1,
    ) -> list[dict[str, Any]]:
        """Return ledger entries: {policy, weight, index, seed, stream}."""


@dataclass
class UniformSampler:
    kind: str = "uniform"

    def sample(
        self,
        members: list[dict[str, Any]],
        *,
        seed: int,
        stream: str,
        n: int = 1,
    ) -> list[dict[str, Any]]:
        rng = np.random.default_rng(seed)
        out = []
        for i in range(n):
            idx = int(rng.integers(0, len(members)))
            m = members[idx]
            out.append(
                {
                    "policy": m["policy"],
                    "weight": float(m.get("weight", 1.0)),
                    "index": idx,
                    "seed": seed,
                    "stream": f"{stream}:{i}",
                    "sampler": self.kind,
                }
            )
        return out


@dataclass
class WeightedSampler:
    kind: str = "weighted"

    def sample(
        self,
        members: list[dict[str, Any]],
        *,
        seed: int,
        stream: str,
        n: int = 1,
    ) -> list[dict[str, Any]]:
        weights = np.asarray([float(m.get("weight", 1.0)) for m in members], dtype=np.float64)
        if np.any(weights < 0) or float(weights.sum()) <= 0:
            raise SchemaError("weighted sampler requires non-negative weights with positive sum")
        probs = weights / weights.sum()
        rng = np.random.default_rng(seed)
        out = []
        for i in range(n):
            idx = int(rng.choice(len(members), p=probs))
            m = members[idx]
            out.append(
                {
                    "policy": m["policy"],
                    "weight": float(m.get("weight", 1.0)),
                    "index": idx,
                    "seed": seed,
                    "stream": f"{stream}:{i}",
                    "sampler": self.kind,
                }
            )
        return out


@dataclass
class EnumeratedCrossplaySampler:
    """Return every member once (cartesian cell expansion uses this)."""

    kind: str = "enumerated_crossplay"

    def sample(
        self,
        members: list[dict[str, Any]],
        *,
        seed: int,
        stream: str,
        n: int = 1,
    ) -> list[dict[str, Any]]:
        del n  # enumerate all
        out = []
        for idx, m in enumerate(members):
            out.append(
                {
                    "policy": m["policy"],
                    "weight": float(m.get("weight", 1.0)),
                    "index": idx,
                    "seed": seed,
                    "stream": f"{stream}:{idx}",
                    "sampler": self.kind,
                }
            )
        return out


SAMPLERS: Registry[PopulationSampler] = Registry(
    "population_sampler",
    interface="PopulationSampler",
    register_via="arena.plugins.samplers.register_sampler",
    tests="sampling ledger replay + weight/uniform/crossplay fixtures",
)


def register_sampler(kind: str, sampler: PopulationSampler, *, replace: bool = False) -> PopulationSampler:
    return SAMPLERS.register(kind, sampler, replace=replace)


def register_builtins() -> None:
    register_sampler("uniform", UniformSampler(), replace=True)
    register_sampler("weighted", WeightedSampler(), replace=True)
    register_sampler("enumerated_crossplay", EnumeratedCrossplaySampler(), replace=True)
