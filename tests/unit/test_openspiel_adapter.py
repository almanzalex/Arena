from __future__ import annotations

import pytest

pytest.importorskip("pyspiel")

from arena.adapters.task_openspiel import OpenSpielPackager, interaction_for_game
from arena.core.errors import SchemaError
from arena.core.tasks import capture_task_trace


def _task(game: str = "tic_tac_toe") -> dict:
    return {
        "adapter": "openspiel",
        "env": f"openspiel://{game}",
        "interaction": interaction_for_game(game),
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
        "schema": "arena.trace-suite/v1",
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
    with pytest.raises(SchemaError, match="qualified catalog.*unqualified"):
        OpenSpielPackager().make_env(_task("go"))


def test_openspiel_connect_four_contract_and_reference_trace() -> None:
    info = OpenSpielPackager().describe_task(_task("connect_four"))
    assert info["roles"]["player_0"]["observation"]["shape"] == [126]
    assert info["roles"]["player_0"]["action"]["n"] == 7
    suite = {
        "schema": "arena.trace-suite/v1",
        "interaction": "aec",
        "episodes": [
            {
                "seed": 0,
                "actions": [
                    {"agent": "player_0", "action": 0},
                    {"agent": "player_1", "action": 1},
                    {"agent": "player_0", "action": 0},
                    {"agent": "player_1", "action": 1},
                    {"agent": "player_0", "action": 0},
                    {"agent": "player_1", "action": 1},
                    {"agent": "player_0", "action": 0},
                ],
            }
        ],
    }
    terminal = capture_task_trace(_task("connect_four"), suite)[0]["events"][-1]
    assert terminal["terminations"] == {"player_0": True, "player_1": True}
    assert terminal["rewards"] == {"player_0": 1.0, "player_1": -1.0}


def test_openspiel_chance_imperfect_information_family_is_seeded() -> None:
    info = OpenSpielPackager().describe_task(_task("kuhn_poker"))
    assert info["interaction"] == "aec"
    assert info["game_semantics"] == {
        "dynamics": "SEQUENTIAL",
        "chance_mode": "EXPLICIT_STOCHASTIC",
        "information": "IMPERFECT_INFORMATION",
        "observation_kind": "information_state",
    }
    assert info["chance_rng"] == "numpy_generator"
    assert info["roles"]["player_0"]["observation"]["shape"] == [11]
    suite = {
        "schema": "arena.trace-suite/v1",
        "interaction": "aec",
        "episodes": [
            {
                "seed": 0,
                "actions": [
                    {"agent": "player_0", "action": 0},
                    {"agent": "player_1", "action": 0},
                ],
            }
        ],
    }
    left = capture_task_trace(_task("kuhn_poker"), suite)
    right = capture_task_trace(_task("kuhn_poker"), suite)
    assert left == right
    assert left[0]["events"][-1]["terminations"] == {
        "player_0": True,
        "player_1": True,
    }


def test_openspiel_leduc_poker_expands_chance_imperfect_family() -> None:
    info = OpenSpielPackager().describe_task(_task("leduc_poker"))
    assert info["interaction"] == "aec"
    assert info["game_semantics"] == {
        "dynamics": "SEQUENTIAL",
        "chance_mode": "EXPLICIT_STOCHASTIC",
        "information": "IMPERFECT_INFORMATION",
        "observation_kind": "information_state",
    }
    assert info["chance_rng"] == "numpy_generator"
    assert info["roles"]["player_0"]["observation"]["shape"] == [30]
    assert info["roles"]["player_0"]["action"]["n"] == 3
    suite = {
        "schema": "arena.trace-suite/v1",
        "interaction": "aec",
        "episodes": [
            {
                "seed": 0,
                "actions": [
                    {"agent": "player_0", "action": 1},
                    {"agent": "player_1", "action": 1},
                    {"agent": "player_0", "action": 1},
                    {"agent": "player_1", "action": 1},
                ],
            }
        ],
    }
    left = capture_task_trace(_task("leduc_poker"), suite)
    right = capture_task_trace(_task("leduc_poker"), suite)
    assert left == right
    terminal = left[0]["events"][-1]
    assert terminal["terminations"] == {"player_0": True, "player_1": True}
    assert terminal["rewards"] == {"player_0": -1.0, "player_1": 1.0}


def test_openspiel_simultaneous_family_uses_parallel_joint_action() -> None:
    info = OpenSpielPackager().describe_task(_task("matrix_rps"))
    assert info["interaction"] == "parallel"
    assert info["game_semantics"]["dynamics"] == "SIMULTANEOUS"
    assert info["roles"]["player_0"]["observation"]["shape"] == [1]
    suite = {
        "schema": "arena.trace-suite/v1",
        "interaction": "parallel",
        "episodes": [
            {
                "seed": 0,
                "actions": [{"player_0": 0, "player_1": 1}],
            }
        ],
    }
    terminal = capture_task_trace(_task("matrix_rps"), suite)[0]["events"][-1]
    assert terminal["terminations"] == {"player_0": True, "player_1": True}
    assert terminal["rewards"] == {"player_0": -1.0, "player_1": 1.0}
