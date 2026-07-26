"""Compatibility check unit tests."""

from arena.core.compatibility import compose_check


def _policy(**overrides):
    base = {
        "name": "p",
        "roles": {"allowed": ["player_0"]},
        "observation": {"type": "Discrete", "n": 4, "dtype": "int64"},
        "action": {"type": "Discrete", "n": 3, "dtype": "int64", "masks": "none"},
        "state": {"recurrent": False, "reset_on": []},
        "inference": {"modes": ["deterministic", "stochastic"]},
        "preprocessing": {"included": True, "id": "normalize_v0"},
    }
    base.update(overrides)
    return base


def test_role_mismatch() -> None:
    report = compose_check(policy=_policy(), role="player_1")
    assert not report.ok
    assert any(i.code == "ROLE_MISMATCH" for i in report.issues)


def test_observation_mismatch() -> None:
    report = compose_check(
        policy=_policy(),
        role="player_0",
        expected_obs={"type": "Discrete", "n": 5, "dtype": "int64"},
    )
    assert not report.ok
    assert any(i.code == "OBSERVATION_MISMATCH" for i in report.issues)


def test_mask_required_without_task_masks() -> None:
    report = compose_check(
        policy=_policy(action={"type": "Discrete", "n": 3, "masks": "required"}),
        role="player_0",
        task_provides_masks=False,
    )
    assert not report.ok
    assert any(i.code == "MASK_REQUIRED" for i in report.issues)


def test_compatible() -> None:
    report = compose_check(
        policy=_policy(),
        role="player_0",
        expected_obs={"type": "Discrete", "n": 4, "dtype": "int64"},
        expected_act={"type": "Discrete", "n": 3, "dtype": "int64"},
        action_mode="deterministic",
        task_provides_masks=False,
    )
    assert report.ok
