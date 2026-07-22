"""Gimitest evaluation provider (RLX 0.3)."""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from rlx.core.errors import RlxError, SchemaError
from rlx.core.identity import canonical_json, digest_uri, sha256_bytes


def _resolve_test_class(ref: str, *, allow_external: bool) -> type[Any]:
    if ":" not in ref:
        raise SchemaError("gimitest provider_config.test_class must be module:Class")
    module_name, attr = ref.split(":", 1)
    if not allow_external and not (
        module_name == "gimitest" or module_name.startswith("gimitest.")
    ):
        raise SchemaError(
            "external Gimitest test classes execute Python. Set "
            "provider_config.allow_external_test_class=true only after reviewing the module, "
            "or select a class shipped by gimitest."
        )
    module = importlib.import_module(module_name)
    cls = getattr(module, attr, None)
    if not isinstance(cls, type):
        raise SchemaError(f"Gimitest test class {ref!r} did not resolve to a class")
    return cls


def decorate_env(env: Any, config: dict[str, Any]) -> Any:
    """Attach upstream Gimitest hooks to an RLX-created task environment."""
    try:
        from gimitest.env_decorator import EnvDecorator
    except ImportError as e:
        raise RlxError(
            "Gimitest provider is optional. Current Gimitest 1.0 pins an obsolete "
            "Gymnasium version; install RLX/PettingZoo first, then use "
            "`pip install --no-deps gimitest==1.0` and run the provider qualification gate."
        ) from e
    test_ref = str(config.get("test_class") or "gimitest.gtest:GTest")
    cls = _resolve_test_class(
        test_ref,
        allow_external=bool(config.get("allow_external_test_class", False)),
    )
    parameters = config.get("parameters") or {}
    if not isinstance(parameters, dict):
        raise SchemaError("gimitest provider_config.parameters must be a mapping")
    try:
        test = cls(env, parameters=parameters)
    except TypeError:
        test = cls(env)
    decorated = EnvDecorator.decorate(env, test)
    setattr(decorated, "_rlx_gimitest", test)
    return decorated


class GimitestEvalProvider:
    kind = "gimitest"

    def run(self, suite: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        config = suite.get("provider_config") or {}
        if not isinstance(config, dict):
            raise SchemaError("provider_config must be a mapping")
        try:
            provider_version = version("gimitest")
        except PackageNotFoundError as e:
            raise RlxError(
                "Gimitest provider is optional. Install the compatibility-tested provider "
                "with `pip install --no-deps gimitest==1.0` after RLX/PettingZoo."
            ) from e
        config_digest = digest_uri(sha256_bytes(canonical_json(config)))
        task = dict(suite["task"])
        task["_eval_provider"] = {
            "kind": "gimitest",
            "config": config,
            "config_digest": config_digest,
        }
        provider_lineage = {
            "kind": "gimitest",
            "version": provider_version,
            "config_digest": config_digest,
        }
        from rlx.runtime.evaluation import _run_native_evaluation

        identity_suite = kwargs.pop("identity_suite", suite)

        return _run_native_evaluation(
            {**suite, "task": task},
            provider_lineage=provider_lineage,
            identity_suite=identity_suite,
            **kwargs,
        )
