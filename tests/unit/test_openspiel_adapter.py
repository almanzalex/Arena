from __future__ import annotations

import pytest

pytest.importorskip("pyspiel")

from rlx.adapters.task_openspiel import OpenSpielPackager
from rlx.core.errors import SchemaError
from rlx.core.tasks import capture_task_trace


def _task(game: str = "tic_tac_toe") -> dict:
    return {
        "adapter": "openspiel",
        "env": f"openspiel://{game}",
        "interaction": "aec",
        "packaging": {"kind": "openspiel"},
    }


def test_openspiel_frozen_game_contract_and_reference_trace() -> None:
    info = OpenSpielPackager().describe_task(_task())
    assert info["adapter"] == "openspiel"
    assert info["interaction"] == "aec"
    assert info["provides_masks"] is True
    assert info["roles"]["player_0"]["observation"]["shape"] == [27]
    assert info["roles"]["player_0"]["action"]["n"] == 9

    suite = {
        "schema": "rlx.trace-suite/v1",
        "interaction": "aec",
        "episodes": [
            {
                "seed": 0,
                "actions": [
                    {"agent": "player_0", "action": 0},
                    {"agent": "player_1", "action": 3},
                    {"agent": "player_0", "action": 1},
                    {"agent": "player_1", "action": 4},
                    {"agent": "player_0", "action": 2},
                ],
            }
        ],
    }
    trace = capture_task_trace(_task(), suite)
    terminal = trace[0]["events"][-1]
    assert terminal["terminations"] == {"player_0": True, "player_1": True}
    assert terminal["rewards"] == {"player_0": 1.0, "player_1": -1.0}


def test_openspiel_unknown_game_fails_loud() -> None:
    with pytest.raises(SchemaError, match="freezes exactly"):
        OpenSpielPackager().make_env(_task("kuhn_poker"))
