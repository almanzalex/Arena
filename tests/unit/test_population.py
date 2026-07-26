"""Population create/inspect, digest stability, role constraints (Phase 2)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("pettingzoo")

from arena.conformance.fixtures import build_fixed_action_rps_policy, build_rps_policy
from arena.core.errors import CompatibilityError, SchemaError
from arena.core.manifests import population_content_digest
from arena.core.population import (
    create_population,
    load_population,
    write_population_yaml,
)
from arena.core.sdk import Policy, Population
from arena.core.store import LocalStore


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_population_digest_stable_across_path_aliases(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    store = LocalStore(tmp_path).init()
    store = LocalStore(tmp_path)
    rock = build_fixed_action_rps_policy(tmp_path / "rock", role="player_1", action=0, name="rock")
    paper = build_fixed_action_rps_policy(tmp_path / "paper", role="player_1", action=1, name="paper")
    scissors = build_fixed_action_rps_policy(
        tmp_path / "scissors", role="player_1", action=2, name="scissors"
    )
    d_rock = Policy.load(rock).digest
    members = [
        {"policy": str(rock), "weight": 1.0, "tags": ["rock"], "roles": {"allowed": ["player_1"]}},
        {"policy": str(paper), "weight": 1.0, "tags": ["paper"], "roles": {"allowed": ["player_1"]}},
        {
            "policy": str(scissors),
            "weight": 1.0,
            "tags": ["scissors"],
            "roles": {"allowed": ["player_1"]},
        },
    ]
    pop1 = create_population(name="rps-opp", members=members, store=store, ref="populations/rps-opp")
    # Alias via digests instead of paths — content digest must match.
    members_alias = [
        {"policy": d_rock, "weight": 1.0, "tags": ["rock"], "roles": {"allowed": ["player_1"]}},
        {
            "policy": Policy.load(paper).digest,
            "weight": 1.0,
            "tags": ["paper"],
            "roles": {"allowed": ["player_1"]},
        },
        {
            "policy": Policy.load(scissors).digest,
            "weight": 1.0,
            "tags": ["scissors"],
            "roles": {"allowed": ["player_1"]},
        },
    ]
    pop2 = create_population(name="rps-opp-2", members=members_alias, store=store)
    assert pop1["digest"] == pop2["digest"]
    assert population_content_digest(pop1) == pop1["digest"]

    loaded = load_population("populations/rps-opp", store=store)
    assert len(loaded["members"]) == 3
    assert all(m["policy"].startswith("sha256:") for m in loaded["members"])

    out = tmp_path / "pop.yaml"
    write_population_yaml(pop1, out)
    again = Population.load(out)
    assert again.digest == pop1["digest"]


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_population_rejects_role_incompatible_members(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    LocalStore(tmp_path).init()
    store = LocalStore(tmp_path)
    p = build_rps_policy(tmp_path / "p0", role="player_0", seed=1)
    pop = create_population(
        name="bad",
        members=[{"policy": str(p), "roles": {"allowed": ["player_0"]}}],
        store=store,
    )
    from arena.core.population import assert_members_compatible_with_role

    with pytest.raises(CompatibilityError, match="incompatible with role"):
        assert_members_compatible_with_role(pop, "player_1")


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_population_object_immutable_when_ref_moves(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    LocalStore(tmp_path).init()
    store = LocalStore(tmp_path)
    p = build_fixed_action_rps_policy(tmp_path / "rock", role="player_1", action=0)
    pop = create_population(
        name="v1",
        members=[{"policy": str(p), "weight": 1.0}],
        store=store,
        ref="populations/live",
    )
    object_digest = pop["object_digest"]
    # Move ref to a different object; original blob must still round-trip.
    p2 = build_fixed_action_rps_policy(tmp_path / "paper", role="player_1", action=1)
    create_population(
        name="v2",
        members=[{"policy": str(p2), "weight": 1.0}],
        store=store,
        ref="populations/live",
    )
    original = load_population(object_digest, store=store)
    assert original["digest"] == pop["digest"]
    assert original["name"] == "v1"


@pytest.mark.requires_torch
def test_population_schema_rejects_empty_members() -> None:
    from arena.core.manifests import validate_population_manifest

    with pytest.raises(SchemaError):
        validate_population_manifest(
            {"schema": "arena.population/v0alpha1", "name": "x", "members": []}
        )
