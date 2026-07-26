"""Joint trajectory bundle writer and inspector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from arena.core.manifests import TRAJECTORY_SCHEMA, dump_json, dump_yaml


class TrajectoryWriter:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.episodes: list[dict[str, Any]] = []

    def write_episode(self, episode: dict[str, Any]) -> None:
        idx = episode.get("episode_index", len(self.episodes))
        path = self.root / f"episode_{int(idx):04d}.json"
        dump_json(episode, path)
        self.episodes.append(
            {
                "episode_index": idx,
                "seed": episode.get("seed"),
                "status": episode.get("status"),
                "steps": len(episode.get("steps", [])),
                "path": str(path.name),
                "returns": episode.get("returns"),
            }
        )

    def finalize(
        self,
        *,
        task_info: dict[str, Any],
        assignments: dict[str, str],
        seeds: list[int],
        action_mode: str,
        failures: list[dict[str, Any]],
    ) -> Path:
        meta = {
            "schema": TRAJECTORY_SCHEMA,
            "task": {
                "env": task_info.get("env"),
                "adapter": task_info.get("adapter"),
                "version": task_info.get("version"),
            },
            "policies": assignments,
            "seeds": seeds,
            "action_mode": action_mode,
            "episodes": self.episodes,
            "failures": failures,
            "episode_count": len(self.episodes),
        }
        dump_yaml(meta, self.root / "bundle.yaml")
        dump_json(meta, self.root / "bundle.json")
        return self.root / "bundle.yaml"


def inspect_trajectory(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    if path.is_dir():
        bundle = path / "bundle.yaml"
        if not bundle.exists():
            bundle = path / "bundle.json"
        path = bundle
    if path.suffix in {".yaml", ".yml"}:
        import yaml

        meta = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        meta = json.loads(path.read_text(encoding="utf-8"))

    # Validate D-01 fields on a sample episode if present
    root = path.parent
    completeness = {"checked": 0, "ok": True, "missing": []}
    required_step = {
        "observations",
        "actions",
        "rewards",
        "terminations",
        "truncations",
    }
    required_ep = {"seed", "task", "agents", "role_map", "policies", "steps"}
    for ep in meta.get("episodes", [])[:5]:
        ep_path = root / ep["path"]
        if not ep_path.exists():
            continue
        episode = json.loads(ep_path.read_text(encoding="utf-8"))
        completeness["checked"] += 1
        for key in required_ep:
            if key not in episode:
                completeness["ok"] = False
                completeness["missing"].append(f"episode.{key}")
        for step in episode.get("steps", []):
            for key in required_step:
                if key not in step:
                    completeness["ok"] = False
                    completeness["missing"].append(f"step.{key}")
            # provenance: each agent in actions should appear in obs/rewards
            for agent in step.get("actions", {}):
                if agent not in step.get("observations", {}):
                    completeness["ok"] = False
                    completeness["missing"].append(f"missing obs for {agent}")
                # Non-Discrete actions must round-trip as JSON-native structures
                # (int vectors / float vectors / dict trees) — never missing/null.
                act = step["actions"].get(agent)
                if act is None:
                    completeness["ok"] = False
                    completeness["missing"].append(f"null action for {agent}")
    meta["completeness"] = completeness
    return meta
