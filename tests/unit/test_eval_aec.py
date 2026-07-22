"""Evaluation suite with interaction=aec (Phase 5 via eval runner)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("pettingzoo")

from _eval_fixtures import build_cyclic_rps_eval_fixture

from rlx.core.population import create_population_from_yaml
from rlx.core.registry import UnknownKindError
from rlx.core.sdk import Policy
from rlx.core.store import LocalStore
from rlx.plugins.interactions import INTERACTIONS, get_interaction
from rlx.runtime.evaluation import run_evaluation


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_eval_run_interaction_aec(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    LocalStore(tmp_path).init()
    store = LocalStore(tmp_path)
    fx = build_cyclic_rps_eval_fixture(tmp_path / "fx", interaction="aec")
    pop = create_population_from_yaml(fx["population"], store=store)
    policy_index = {Policy.load(p).digest: Path(p) for p in fx["bundles"].values()}
    suite = {
        "schema": "rlx.evaluation/v0alpha1",
        "name": "aec-cyclic",
        "interaction": "aec",
        "task": {
            "adapter": "pettingzoo-parallel",
            "env": "rlx/competitive_rps_aec_v0",
            "interaction": "aec",
            "config": {"max_cycles": 1},
        },
        "assignments": {
            "player_0": {"kind": "crossplay", "population": pop["digest"]},
            "player_1": {"kind": "crossplay", "population": pop["digest"]},
        },
        "seeds": {"start": 0, "count": 1},
        "action_mode": "deterministic",
        "metrics": ["payoff_matrix"],
    }
    result = run_evaluation(
        suite,
        policy_index=policy_index,
        populations={pop["digest"]: pop},
        out_dir=tmp_path / "eval-aec",
    )
    assert result["interaction"] == "aec"
    assert len(result["cells"]) == 9
    assert (tmp_path / "eval-aec" / "cell-0" / "trajectories").exists()


@pytest.mark.requires_torch
def test_unknown_interaction_fails_with_recipe() -> None:
    with pytest.raises(UnknownKindError, match="Unknown interaction"):
        get_interaction("turn-based-fantasy")
    assert "parallel" in INTERACTIONS.known()
    assert "aec" in INTERACTIONS.known()
