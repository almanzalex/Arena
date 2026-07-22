"""Idempotent registration of built-in axis cases."""

from __future__ import annotations

_REGISTERED = False


def ensure_registered() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    from rlx.plugins import (
        actions,
        distributions,
        eval_providers,
        interactions,
        metrics,
        payloads,
        preprocess_ops,
        samplers,
        stores,
        tasks,
        wrappers,
    )

    actions.register_builtins()
    distributions.register_builtins()
    preprocess_ops.register_builtins()
    wrappers.register_builtins()
    payloads.register_builtins()
    tasks.register_builtins()
    eval_providers.register_builtins()
    stores.register_builtins()
    samplers.register_builtins()
    metrics.register_builtins()
    interactions.register_builtins()
    _REGISTERED = True
