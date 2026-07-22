"""Evaluation-provider registry (native and optional robustness providers)."""

from __future__ import annotations

from typing import Any, Protocol

from rlx.core.registry import EVAL_PROVIDERS


class EvalProvider(Protocol):
    kind: str

    def run(self, suite: dict[str, Any], **kwargs: Any) -> dict[str, Any]: ...


def register_eval_provider(
    kind: str, provider: EvalProvider, *, replace: bool = False
) -> EvalProvider:
    return EVAL_PROVIDERS.register(kind, provider, replace=replace)


class NativeEvalProvider:
    kind = "native"

    def run(self, suite: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        from rlx.runtime.evaluation import _run_native_evaluation

        return _run_native_evaluation(suite, **kwargs)


def register_builtins() -> None:
    register_eval_provider("native", NativeEvalProvider(), replace=True)
    # The optional implementation imports gimitest only when selected.
    from rlx.adapters.eval_gimitest import GimitestEvalProvider

    register_eval_provider("gimitest", GimitestEvalProvider(), replace=True)
