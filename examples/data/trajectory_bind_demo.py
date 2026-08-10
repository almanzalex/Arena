#!/usr/bin/env python3
"""Example: bind a trajectory slice to policy+task identity (fail-loud).

Creates two synthetic match runs, selects episodes for one policy digest with
provenance binding, verifies the binding, then shows unbind + mismatch failure.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from arena.core.errors import ConformanceError
from arena.core.identity import digest_uri, sha256_bytes
from arena.dataset import (
    bind_dataset_provenance,
    select_bound_episodes,
    unbind_dataset_provenance,
    verify_dataset_provenance,
)

POLICY_ROCK = digest_uri("1" * 64)
POLICY_PAPER = digest_uri("2" * 64)
TASK = {
    "adapter": "pettingzoo-parallel",
    "env": "arena/competitive_rps_v0",
    "version": "example+pettingzoo",
}


def _episode(path: Path, *, p0: str, p1: str, seed: int) -> None:
    payload = {
        "schema": "arena.trajectory/v0alpha1",
        "seed": seed,
        "episode_index": 0,
        "status": "completed",
        "action_mode": "deterministic",
        "task": TASK,
        "agents": ["player_0", "player_1"],
        "role_map": {"player_0": "player_0", "player_1": "player_1"},
        "policies": {"player_0": p0, "player_1": p1},
        "returns": {"player_0": -1.0, "player_1": 1.0},
        "steps": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="arena-traj-bind-") as tmp:
        root = Path(tmp)
        rock_run = root / "rock-vs-paper"
        paper_run = root / "paper-vs-rock"
        _episode(
            rock_run / "trajectories" / "episode_0000.json",
            p0=POLICY_ROCK,
            p1=POLICY_PAPER,
            seed=0,
        )
        _episode(
            paper_run / "trajectories" / "episode_0000.json",
            p0=POLICY_PAPER,
            p1=POLICY_ROCK,
            seed=1,
        )

        bound = select_bound_episodes(
            source_runs=[rock_run, paper_run],
            policy_digest=POLICY_ROCK,
            task=TASK,
            role="player_0",
            name="rock-as-player-0",
            out_dir=root / "datasets" / "rock",
        )
        print("bound episodes:", len(bound["episodes"]))
        print("provenance:", json.dumps(bound["provenance"], indent=2, sort_keys=True))
        print("dataset digest:", bound["digest"])
        verify_dataset_provenance(bound, expect_policy=POLICY_ROCK, expect_task=TASK)
        print("verify: ok")

        unbound = unbind_dataset_provenance(bound)
        print("unbound digest:", unbound["digest"])
        assert "provenance" not in unbound

        # Fail-loud: claim the rock-bound slice is for the paper policy.
        try:
            verify_dataset_provenance(bound, expect_policy=POLICY_PAPER)
        except ConformanceError as exc:
            print("expected mismatch:", exc.code, "-", exc)
        else:
            print("error: expected policy digest mismatch was not raised", file=sys.stderr)
            return 1

        # Fail-loud bind against foreign digest on a hand-built slice.
        ep = rock_run / "trajectories" / "episode_0000.json"
        hand = {
            "schema": "arena.dataset/v0alpha1",
            "name": "hand",
            "source_runs": [str(rock_run)],
            "episodes": [
                {
                    "path": str(ep),
                    "digest": digest_uri(sha256_bytes(ep.read_bytes())),
                    "seed": 0,
                }
            ],
            "query": {},
            "lineage": {},
        }
        try:
            bind_dataset_provenance(hand, policy_digest=digest_uri("9" * 64), task=TASK)
        except ConformanceError as exc:
            print("expected bind mismatch:", exc.code, "-", exc)
        else:
            print("error: expected bind mismatch was not raised", file=sys.stderr)
            return 1

    print("trajectory provenance bind example: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
