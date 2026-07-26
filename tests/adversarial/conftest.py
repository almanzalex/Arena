"""Shared fixtures for adversarial tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def patch_task_env(monkeypatch):
    """Return a helper that installs a custom env factory into both make_env sites.

    ``describe_task`` (adapter module) and ``_run_episode`` (match module) hold
    independent references to ``make_env``; both must be patched for a custom env to
    flow through pre-run compatibility introspection *and* execution.
    """
    from arena.adapters.task_pettingzoo import adapter as adapter_mod
    from arena.runtime import match as match_mod

    def _install(env_cls, **env_kwargs):
        def factory(spec, **_kwargs):
            cfg = dict(spec.get("config") or {})
            kwargs = dict(env_kwargs)
            if "max_cycles" in cfg:
                kwargs["max_cycles"] = cfg["max_cycles"]
            return env_cls(**kwargs)

        monkeypatch.setattr(adapter_mod, "make_env", factory)
        monkeypatch.setattr(match_mod, "make_env", factory)
        return factory

    return _install
