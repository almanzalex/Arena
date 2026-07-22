"""Measure RLX native versus OpenEnv loopback reset+step overhead (S-01)."""

from __future__ import annotations

import argparse
import json
import statistics
import time

from rlx.adapters.task_openenv.adapter import PILOT_CONTRACT, OpenEnvParallelEnv
from rlx.adapters.task_pettingzoo.pilot_env import CompetitiveRPSParallel


def _sample(env, iterations: int) -> list[float]:
    samples: list[float] = []
    for i in range(iterations):
        start = time.perf_counter()
        env.reset(seed=i)
        env.step({"player_0": 0, "player_1": 1})
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def _summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, int(len(ordered) * 0.95))
    return {
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": ordered[p95_index],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    native = CompetitiveRPSParallel(max_cycles=1)
    external = OpenEnvParallelEnv(
        {
            "adapter": "openenv",
            "env": "openenv://rlx/competitive_rps_v0",
            "interaction": "parallel",
            "packaging": {"kind": "openenv", "base_url": args.base_url},
            "contract": PILOT_CONTRACT,
        }
    )
    try:
        native_samples = _sample(native, args.iterations)
        external_samples = _sample(external, args.iterations)
    finally:
        native.close()
        external.close()
    native_summary = _summary(native_samples)
    external_summary = _summary(external_samples)
    print(
        json.dumps(
            {
                "iterations": args.iterations,
                "operation": "reset_plus_one_joint_step",
                "native": native_summary,
                "openenv_loopback": external_summary,
                "mean_overhead_ms": external_summary["mean_ms"]
                - native_summary["mean_ms"],
                "mean_ratio": external_summary["mean_ms"]
                / native_summary["mean_ms"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
