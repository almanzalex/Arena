"""Action-type cases registered on the action axis."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from rlx.core.registry import ACTION_CASES


class ActionCase(Protocol):
    kind: str

    def validate(
        self,
        action: dict[str, Any],
        *,
        architecture: dict[str, Any] | None = None,
        adapter: str | None = None,
        require_byo_layout: bool = False,
    ) -> None: ...

    def decode(
        self,
        params: np.ndarray,
        *,
        action: dict[str, Any],
        mode: str,
        rng: np.random.Generator | None = None,
        action_mask: Any = None,
    ) -> Any: ...

    def actions_equal(self, got: Any, expected: Any, *, action: dict[str, Any]) -> bool: ...

    def validate_expected(
        self, case: dict[str, Any], *, action: dict[str, Any], index: int | None = None
    ) -> None: ...

    def validate_runtime(
        self, value: Any, *, action: dict[str, Any], agent: str | None = None
    ) -> None: ...


def register_action_case(kind: str, case: ActionCase, *, replace: bool = False) -> ActionCase:
    return ACTION_CASES.register(kind, case, replace=replace)


def register_builtins() -> None:
    from rlx.plugins import _action_impl as impl

    register_action_case("Discrete", impl.DiscreteActionCase(), replace=True)
    register_action_case("MultiDiscrete", impl.MultiDiscreteActionCase(), replace=True)
    register_action_case("Box", impl.BoxActionCase(), replace=True)
    register_action_case("Dict", impl.DictActionCase(), replace=True)
