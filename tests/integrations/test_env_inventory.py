"""Inventory assertions for Gymnasium / PettingZoo / OpenSpiel / OpenEnv / Gimitest."""

from __future__ import annotations

from arena.core.support import load_support_matrix


def test_support_matrix_lists_real_environment_capabilities() -> None:
    matrix = load_support_matrix()
    caps = matrix["capabilities"]
    assert caps["openspiel"]["status"] == "stable"
    assert "open_spiel" in caps["openspiel"]["distributions"]
    assert "gymnasium" in caps["openspiel"]["distributions"]
    assert caps["openspiel"]["credentials"] is False

    assert caps["openenv"]["status"] == "preview"
    assert caps["openenv"]["required_for_1_0"] is True
    assert "openenv" in caps["openenv"]["distributions"]
    assert "pettingzoo" in caps["openenv"]["distributions"]
    assert "gymnasium" in caps["openenv"]["distributions"]
    assert caps["openenv"]["credentials"] is False

    assert caps["gimitest"]["status"] == "preview"
    assert caps["gimitest"]["isolated_python_env"] == "ARENA_GIMITEST_PYTHON"
    assert caps["gimitest"]["credentials"] is False

    # PettingZoo is the native task path (extra), not a top-level capability row.
    # Gymnasium appears as a distribution dependency of pettingzoo/openenv/openspiel.
    for name in ("hf", "oci", "wandb", "mlflow"):
        assert caps[name]["credentials"] is True


def test_task_packagers_register_external_env_surfaces() -> None:
    from arena.core.registry import EVAL_PROVIDERS, TASK_PACKAGERS, ensure_plugins_loaded

    ensure_plugins_loaded()
    assert "pettingzoo_wrappers" in TASK_PACKAGERS
    assert "entrypoint_bundle" in TASK_PACKAGERS
    assert "openenv" in TASK_PACKAGERS
    assert "openspiel" in TASK_PACKAGERS
    assert EVAL_PROVIDERS.get("gimitest").kind == "gimitest"
