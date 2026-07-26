"""Shared builders for 0.2 eval / qualify fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from arena.adapters.policy_custom_torch import (
    _embed_reference_cases,
    generate_reference_cases,
    load_runtime,
)
from arena.conformance.fixtures import build_fixed_action_rps_policy
from arena.core.manifests import dump_yaml
from arena.core.sdk import Policy


def _with_source_cases(bundle: Path) -> Path:
    runtime = load_runtime(bundle)
    _embed_reference_cases(
        bundle,
        generate_reference_cases(
            runtime,
            observation=runtime.manifest["observation"],
            action=runtime.manifest["action"],
        ),
        provenance="source-conformance",
    )
    return bundle


def build_cyclic_rps_eval_fixture(root: Path, *, interaction: str = "parallel") -> dict[str, Any]:
    """Write rock/paper/scissors bundles + population + evaluation YAMLs under ``root``."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    roles = ["player_0", "player_1"]
    rock = _with_source_cases(
        build_fixed_action_rps_policy(root / "rock.arena", role=roles, action=0, name="rock")
    )
    paper = _with_source_cases(
        build_fixed_action_rps_policy(root / "paper.arena", role=roles, action=1, name="paper")
    )
    scissors = _with_source_cases(
        build_fixed_action_rps_policy(root / "scissors.arena", role=roles, action=2, name="scissors")
    )
    env = (
        "arena/competitive_rps_aec_v0"
        if interaction == "aec"
        else "arena/competitive_rps_v0"
    )
    population = {
        "schema": "arena.population/v0alpha1",
        "name": "cyclic-rps",
        "members": [
            {"policy": "./rock.arena", "weight": 1.0, "tags": ["rock"]},
            {"policy": "./paper.arena", "weight": 1.0, "tags": ["paper"]},
            {"policy": "./scissors.arena", "weight": 1.0, "tags": ["scissors"]},
        ],
    }
    pop_path = root / "population.yaml"
    dump_yaml(population, pop_path)
    evaluation = {
        "schema": "arena.evaluation/v0alpha1",
        "name": "cyclic-matrix",
        "interaction": interaction,
        "task": {
            "adapter": "pettingzoo-parallel",
            "env": env,
            "interaction": interaction,
            "config": {"max_cycles": 1},
        },
        "assignments": {
            "player_0": {"kind": "crossplay", "population": "./population.yaml"},
            "player_1": {"kind": "crossplay", "population": "./population.yaml"},
        },
        "seeds": {"start": 0, "count": 1},
        "action_mode": "deterministic",
        "metrics": ["payoff_matrix", "mean_return", "win_rate"],
    }
    eval_path = root / "evaluation.yaml"
    dump_yaml(evaluation, eval_path)
    digests = {
        "rock": Policy.load(rock).digest,
        "paper": Policy.load(paper).digest,
        "scissors": Policy.load(scissors).digest,
    }
    return {
        "root": root,
        "evaluation": eval_path,
        "population": pop_path,
        "bundles": {"rock": rock, "paper": paper, "scissors": scissors},
        "digests": digests,
    }
