"""Schema validation for 0.2 population / evaluation manifests."""

from __future__ import annotations

import pytest

from arena.core.errors import SchemaError
from arena.core.manifests import (
    EVALUATION_SCHEMA,
    POPULATION_SCHEMA,
    evaluation_content_digest,
    population_content_digest,
    validate_evaluation_manifest,
    validate_population_manifest,
)


def test_population_requires_digest_members() -> None:
    with pytest.raises(SchemaError, match="sha256"):
        validate_population_manifest(
            {
                "schema": POPULATION_SCHEMA,
                "name": "x",
                "members": [{"policy": "./local.arena"}],
            }
        )


def test_population_digest_ignores_name_and_path_aliases() -> None:
    base_members = [
        {"policy": "sha256:" + "a" * 64, "weight": 1.0, "tags": ["t"]},
        {"policy": "sha256:" + "b" * 64, "weight": 2.0},
    ]
    a = validate_population_manifest(
        {"schema": POPULATION_SCHEMA, "name": "one", "members": base_members}
    )
    b = validate_population_manifest(
        {"schema": POPULATION_SCHEMA, "name": "two", "members": list(reversed(base_members))}
    )
    assert population_content_digest(a) == population_content_digest(b)


def test_evaluation_requires_transform_on_role_swap() -> None:
    with pytest.raises(SchemaError, match="transform"):
        validate_evaluation_manifest(
            {
                "schema": EVALUATION_SCHEMA,
                "name": "e",
                "task": {"adapter": "pettingzoo-parallel", "env": "arena/competitive_rps_v0"},
                "assignments": {"player_0": "sha256:" + "c" * 64},
                "seeds": [0],
                "action_mode": "deterministic",
                "metrics": ["payoff_matrix"],
                "role_swaps": [{"map": {"player_0": "player_1"}}],
            }
        )


def test_evaluation_digest_stable() -> None:
    suite = validate_evaluation_manifest(
        {
            "schema": EVALUATION_SCHEMA,
            "name": "e",
            "task": {"adapter": "pettingzoo-parallel", "env": "arena/competitive_rps_v0"},
            "assignments": {
                "player_0": {"kind": "policy", "policy": "sha256:" + "c" * 64},
                "player_1": {"kind": "crossplay", "population": "sha256:" + "d" * 64},
            },
            "seeds": {"start": 0, "count": 2},
            "action_mode": "deterministic",
            "metrics": ["payoff_matrix", "mean_return"],
        }
    )
    d1 = evaluation_content_digest(suite)
    suite["name"] = "renamed"
    d2 = evaluation_content_digest(suite)
    assert d1 == d2
