"""Training-algorithm registry.

Arena owns recipe identity, dataset integrity, output policy conformance, and run
lineage. A training case owns algorithm-specific validation and execution.
Third-party cases can implement a completely different loop without patching
the CLI or pretending to be behavior cloning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from arena.core.errors import SchemaError
from arena.core.registry import TRAINERS


class TrainingCase(Protocol):
    kind: str

    def validate(self, recipe: dict[str, Any]) -> None: ...

    def run(
        self,
        recipe: dict[str, Any],
        *,
        recipe_path: Path,
        out_dir: Path,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class BehaviorCloningTrainer:
    kind: str = "behavior_cloning"

    def validate(self, recipe: dict[str, Any]) -> None:
        _validate_categorical_recipe(recipe)

    def run(
        self,
        recipe: dict[str, Any],
        *,
        recipe_path: Path,
        out_dir: Path,
    ) -> dict[str, Any]:
        from arena.runtime.training import _run_categorical_recipe

        return _run_categorical_recipe(
            recipe,
            recipe_path=recipe_path,
            out_dir=out_dir,
            weighting="uniform",
        )


@dataclass(frozen=True)
class ReturnWeightedRegressionTrainer:
    """Offline reference case that favors samples from higher-return episodes."""

    kind: str = "return_weighted_regression"

    def validate(self, recipe: dict[str, Any]) -> None:
        _validate_categorical_recipe(recipe)
        config = dict(recipe.get("algorithm_config") or {})
        temperature = float(config.get("temperature", 1.0))
        max_weight = float(config.get("max_weight", 20.0))
        if not math.isfinite(temperature) or temperature <= 0:
            raise SchemaError("return_weighted_regression temperature must be positive")
        if not math.isfinite(max_weight) or max_weight < 1:
            raise SchemaError("return_weighted_regression max_weight must be >= 1")

    def run(
        self,
        recipe: dict[str, Any],
        *,
        recipe_path: Path,
        out_dir: Path,
    ) -> dict[str, Any]:
        from arena.runtime.training import _run_categorical_recipe

        return _run_categorical_recipe(
            recipe,
            recipe_path=recipe_path,
            out_dir=out_dir,
            weighting="episode_return",
        )


def _validate_categorical_recipe(recipe: dict[str, Any]) -> None:
    action = recipe["action"]
    if not isinstance(action, dict):
        raise SchemaError("training recipe action must be a mapping")
    if action.get("type") != "Discrete" or int(action.get("n", 0)) < 1:
        raise SchemaError(
            f"{recipe['algorithm']} currently requires a declared Discrete action"
        )
    observation = recipe["observation"]
    if not isinstance(observation, dict):
        raise SchemaError("training recipe observation must be a mapping")
    if observation.get("type") not in {"Discrete", "Box"}:
        raise SchemaError(
            f"{recipe['algorithm']} observations currently support Discrete or Box"
        )


def register_trainer(
    kind: str, trainer: TrainingCase, *, replace: bool = False
) -> TrainingCase:
    return TRAINERS.register(kind, trainer, replace=replace)


def register_builtins() -> None:
    register_trainer("behavior_cloning", BehaviorCloningTrainer(), replace=True)
    register_trainer(
        "return_weighted_regression",
        ReturnWeightedRegressionTrainer(),
        replace=True,
    )
