"""Stronger OpenSpiel Match / CLI smoke for stable frozen games."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

pytest.importorskip("pyspiel")
torch = pytest.importorskip("torch")

from arena.adapters.policy_custom_torch import (  # noqa: E402  (after importorskip)
    build_module,
    export_policy,
)
from arena.cli.main import main  # noqa: E402
from arena.conformance.qualification import qualify_task_fixture  # noqa: E402
from arena.core.sdk import Match, Policy, Task  # noqa: E402


def _masked_policy(
    out: Path,
    *,
    name: str,
    observation_dim: int,
    action_n: int,
) -> Path:
    architecture = {
        "type": "mlp_categorical",
        "observation_dim": observation_dim,
        "hidden_dims": [16],
        "action_n": action_n,
    }
    module = build_module(architecture)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()
    return export_policy(
        out_dir=out,
        name=name,
        roles=["player_0", "player_1"],
        observation={
            "type": "Box",
            "shape": [observation_dim],
            "dtype": "float32",
            "low": 0.0,
            "high": 1.0,
        },
        action={
            "type": "Discrete",
            "n": action_n,
            "dtype": "int64",
            "masks": "required",
        },
        architecture=architecture,
        state_dict=module.state_dict(),
        preprocessing={"id": "normalize_v0", "mean": 0.0, "std": 1.0},
        modes=["deterministic", "stochastic"],
        lineage={"fixture": "env-smoke-openspiel", "name": name},
    )


@pytest.mark.requires_openspiel
@pytest.mark.parametrize(
    ("game", "file_stem", "observation_dim", "action_n", "seeds"),
    [
        ("tic_tac_toe", "tic-tac-toe", 27, 9, [0, 1, 2]),
        ("connect_four", "connect-four", 126, 7, [0, 1]),
        ("kuhn_poker", "kuhn-poker", 11, 2, [0, 1]),
        ("matrix_rps", "matrix-rps", 1, 3, [0]),
    ],
)
def test_openspiel_frozen_games_multi_seed_match_and_qualification(
    tmp_path: Path,
    game: str,
    file_stem: str,
    observation_dim: int,
    action_n: int,
    seeds: list[int],
) -> None:
    bundle = _masked_policy(
        tmp_path / f"{game}.arena",
        name=f"{game}-smoke",
        observation_dim=observation_dim,
        action_n=action_n,
    )
    policy = Policy.load(bundle)
    task_path = Path(f"examples/tasks/openspiel-{file_stem}.yaml")
    result = Match(
        task=Task.load(task_path),
        assignments={"player_0": policy, "player_1": policy},
    ).run(seeds=seeds, out=tmp_path / f"{game}-match")
    assert result["outcome"] == {
        "episodes_requested": len(seeds),
        "episodes_completed": len(seeds),
        "failure_count": 0,
    }
    qualification = qualify_task_fixture(
        task_path,
        peer=None,
        trace_suite=f"examples/tasks/openspiel-{file_stem}-trace.yaml",
        report_path=tmp_path / f"{game}-qualification.json",
    )
    assert qualification["ok"] is True
    assert qualification["adapter"] == "openspiel"


@pytest.mark.requires_openspiel
def test_openspiel_cli_match_run_smoke(tmp_path: Path) -> None:
    left = _masked_policy(
        tmp_path / "left.arena",
        name="cli-left",
        observation_dim=27,
        action_n=9,
    )
    right = _masked_policy(
        tmp_path / "right.arena",
        name="cli-right",
        observation_dim=27,
        action_n=9,
    )
    match_path = tmp_path / "match.yaml"
    match_path.write_text(
        yaml.safe_dump(
            {
                "schema": "arena.match/v0alpha1",
                "task": str(
                    (Path("examples/tasks/openspiel-tic-tac-toe.yaml")).resolve()
                ),
                "assignments": {
                    "player_0": str(left),
                    "player_1": str(right),
                },
                "seeds": {"start": 0, "count": 2},
                "action_mode": "deterministic",
                "record": {"trajectories": "all"},
                "failure_policy": {
                    "timeout_seconds": 60,
                    "retain_incomplete": True,
                    "retry": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "cli-run"
    assert main(["match", "run", str(match_path), "--out", str(out)]) == 0
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["task"]["adapter"] == "openspiel"
    assert run["outcome"] == {
        "episodes_requested": 2,
        "episodes_completed": 2,
        "failure_count": 0,
    }
