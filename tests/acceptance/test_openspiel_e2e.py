from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlx.adapters.policy_custom_torch import build_module, export_policy
from rlx.cli.main import main
from rlx.conformance.qualification import qualify_task_fixture
from rlx.core.sdk import Policy
from rlx.runtime.aec_match import run_aec_match
from rlx.runtime.evaluation import run_evaluation

pytest.importorskip("pyspiel")
torch = pytest.importorskip("torch")


def _masked_policy(out: Path, *, name: str) -> Path:
    architecture = {
        "type": "mlp_categorical",
        "observation_dim": 27,
        "hidden_dims": [16],
        "action_n": 9,
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
            "shape": [27],
            "dtype": "float32",
            "low": 0.0,
            "high": 1.0,
        },
        action={"type": "Discrete", "n": 9, "dtype": "int64", "masks": "required"},
        architecture=architecture,
        state_dict=module.state_dict(),
        preprocessing={"id": "normalize_v0", "mean": 0.0, "std": 1.0},
        modes=["deterministic", "stochastic"],
        lineage={"fixture": "OS-01", "game": "tic_tac_toe"},
    )


@pytest.mark.acceptance
@pytest.mark.requires_openspiel
def test_openspiel_tic_tac_toe_match_and_eval(tmp_path: Path) -> None:
    left = _masked_policy(tmp_path / "left.rlx", name="first-legal-left")
    right = _masked_policy(tmp_path / "right.rlx", name="first-legal-right")
    task = {
        "adapter": "openspiel",
        "env": "openspiel://tic_tac_toe",
        "interaction": "aec",
        "packaging": {"kind": "openspiel"},
    }
    assignments = {"player_0": Policy.load(left), "player_1": Policy.load(right)}
    match = run_aec_match(
        task_spec=task,
        assignments=assignments,
        seeds=[0],
        out_dir=tmp_path / "match",
    )
    assert match["outcome"] == {
        "episodes_requested": 1,
        "episodes_completed": 1,
        "failure_count": 0,
    }
    episode = json.loads(
        (tmp_path / "match" / "trajectories" / "episode_0000.json").read_text()
    )
    assert episode["task"]["adapter"] == "openspiel"
    assert episode["policies"]["player_0"] == assignments["player_0"].digest
    assert episode["status"] == "completed"

    suite = {
        "schema": "rlx.evaluation/v0alpha1",
        "name": "openspiel-tic-tac-toe",
        "provider": "native",
        "interaction": "aec",
        "task": task,
        "assignments": {"player_0": str(left), "player_1": str(right)},
        "seeds": [0],
        "action_mode": "deterministic",
        "metrics": ["mean_return"],
    }
    evaluation = run_evaluation(suite, policy_index={}, out_dir=tmp_path / "eval")
    assert evaluation["provider"]["kind"] == "native"
    assert evaluation["cells"][0]["failures"] == 0
    assert evaluation["cells"][0]["lineage"]["task_digest"] == evaluation["task_digest"]

    imported = tmp_path / "openspiel-task.yaml"
    assert main(
        [
            "task",
            "import",
            "openspiel://tic_tac_toe",
            "--name",
            "task:tic-tac-toe@0.3",
            "--out",
            str(imported),
        ]
    ) == 0
    assert main(
        [
            "task",
            "verify-equivalence",
            str(imported),
            "--trace-suite",
            "examples/tasks/openspiel-tic-tac-toe-trace.yaml",
        ]
    ) == 0
    qualification = qualify_task_fixture(
        imported,
        peer=None,
        trace_suite="examples/tasks/openspiel-tic-tac-toe-trace.yaml",
        report_path=tmp_path / "openspiel-qualification.json",
    )
    assert qualification["ok"] is True
    assert qualification["adapter"] == "openspiel"
