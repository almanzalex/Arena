from __future__ import annotations

import pytest

from rlx.core.errors import SchemaError
from rlx.core.manifests import (
    evaluation_content_digest,
    load_manifest,
    validate_evaluation_manifest,
    validate_task_manifest,
)
from rlx.core.registry import (
    EVAL_PROVIDERS,
    EXTERNAL_STORES,
    TASK_PACKAGERS,
    UnknownKindError,
    ensure_plugins_loaded,
)


def test_v03_builtin_registry_axes() -> None:
    ensure_plugins_loaded()
    assert {"openenv", "openspiel"} <= TASK_PACKAGERS.known()
    assert {"native", "gimitest"} <= EVAL_PROVIDERS.known()
    assert {"file", "hf"} <= EXTERNAL_STORES.known()


@pytest.mark.parametrize(
    ("registry", "kind", "axis"),
    [
        (EVAL_PROVIDERS, "lab-evaluator", "eval_provider"),
        (EXTERNAL_STORES, "s3", "external_store"),
        (TASK_PACKAGERS, "ray", "task_packaging"),
    ],
)
def test_v03_unknown_cases_have_extension_recipe(registry, kind: str, axis: str) -> None:
    ensure_plugins_loaded()
    with pytest.raises(UnknownKindError) as exc:
        registry.get(kind)
    message = str(exc.value)
    assert axis in message
    assert "register" in message
    assert "rlx adapter qualify" in message


def test_native_provider_default_preserves_v02_evaluation_identity() -> None:
    suite = {
        "schema": "rlx.evaluation/v0alpha1",
        "name": "legacy-native-suite",
        "task": {"adapter": "pettingzoo-parallel", "env": "rlx/competitive_rps_v0"},
        "assignments": {"player_0": "sha256:a", "player_1": "sha256:b"},
        "seeds": [0],
        "action_mode": "deterministic",
        "metrics": ["mean_return"],
    }
    legacy_digest = evaluation_content_digest(suite)
    normalized = validate_evaluation_manifest(suite)
    assert normalized["provider"] == "native"
    assert normalized["provider_config"] == {}
    assert evaluation_content_digest(normalized) == legacy_digest


def test_task_manifest_digest_refuses_semantic_mutation() -> None:
    task = load_manifest("examples/tasks/native-rps.yaml")
    validate_task_manifest(task)
    task["config"]["max_cycles"] = 2
    with pytest.raises(SchemaError, match="task digest mismatch"):
        validate_task_manifest(task)
