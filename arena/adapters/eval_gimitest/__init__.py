"""Gimitest evaluation provider (Arena 0.3)."""

from __future__ import annotations

import importlib
import sys
import tempfile
import uuid
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from arena.core.errors import ArenaError, SchemaError
from arena.core.identity import canonical_json, digest_uri, sha256_bytes
from arena.core.io import atomic_write_bytes
from arena.core.manifests import load_manifest
from arena.core.supervisor import run_supervised


def _resolve_test_class(ref: str, *, allow_external: bool) -> type[Any]:
    if ":" not in ref:
        raise SchemaError("gimitest provider_config.test_class must be module:Class")
    module_name, attr = ref.split(":", 1)
    if not allow_external and not (
        module_name == "gimitest"
        or module_name.startswith("gimitest.")
        or module_name == "arena.adapters.eval_gimitest.scenarios"
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
    """Attach upstream Gimitest hooks to an Arena-created task environment."""
    try:
        from gimitest.env_decorator import EnvDecorator
    except ImportError as e:
        raise ArenaError(
            "Gimitest provider support is incomplete. Install "
            "`arena[torch,pettingzoo,gimitest]` first, then use "
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
    setattr(decorated, "_arena_gimitest", test)
    return decorated


class GimitestEvalProvider:
    kind = "gimitest"

    def run(self, suite: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        config = suite.get("provider_config") or {}
        if not isinstance(config, dict):
            raise SchemaError("provider_config must be a mapping")
        isolation = dict(config.get("isolation") or {})
        if isolation.get("mode", "in_process") == "subprocess":
            return self._run_subprocess(suite, isolation=isolation, **kwargs)
        if isolation.get("mode", "in_process") != "in_process":
            raise SchemaError("gimitest isolation.mode must be in_process|subprocess")
        return self._run_in_process(suite, **kwargs)

    def _run_in_process(
        self,
        suite: dict[str, Any],
        *,
        _worker_lineage: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        config = suite.get("provider_config") or {}
        try:
            provider_version = version("gimitest")
        except PackageNotFoundError as e:
            raise ArenaError(
                "Gimitest provider is optional. Install "
                "`arena[torch,pettingzoo,gimitest]`, then install the "
                "compatibility-tested provider with "
                "`pip install --no-deps gimitest==1.0`."
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
        if _worker_lineage is not None:
            provider_lineage["worker"] = _worker_lineage
        from arena.runtime.evaluation import _run_native_evaluation

        identity_suite = kwargs.pop("identity_suite", suite)

        return _run_native_evaluation(
            {**suite, "task": task},
            provider_lineage=provider_lineage,
            identity_suite=identity_suite,
            **kwargs,
        )

    def _run_subprocess(
        self,
        suite: dict[str, Any],
        *,
        isolation: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run the complete provider in a user-selected Python environment.

        This isolates incompatible Python dependencies. It is process isolation,
        not a malware sandbox; OS/container restrictions remain the caller's job.
        """
        if kwargs.get("store") is not None:
            raise SchemaError(
                "subprocess eval providers require path/digest inputs; a live LocalStore "
                "object cannot cross the JSON worker boundary"
            )
        python = Path(str(isolation.get("python") or sys.executable))
        if not python.is_absolute():
            raise SchemaError("gimitest isolation.python must be an absolute executable path")
        policy_index = {
            str(key): str(Path(value).resolve())
            for key, value in dict(kwargs.get("policy_index") or {}).items()
        }
        out_dir = kwargs.get("out_dir")
        if out_dir is None:
            raise SchemaError(
                "gimitest subprocess isolation requires an explicit evaluation out_dir"
            )
        request = {
            "schema": "arena.eval-provider-request/v1",
            "request_id": str(uuid.uuid4()),
            "suite": suite,
            "identity_suite": kwargs.get("identity_suite", suite),
            "policy_index": policy_index,
            "populations": kwargs.get("populations"),
            "out_dir": str(Path(out_dir).resolve()),
            "workers": int(kwargs.get("workers", 1)),
            "record": bool(kwargs.get("record", True)),
        }
        with tempfile.TemporaryDirectory(prefix="arena-gimitest-worker-") as raw:
            request_path = Path(raw) / "request.json"
            response_path = Path(raw) / "response.json"
            request_digest = digest_uri(sha256_bytes(canonical_json(request)))
            request["request_digest"] = request_digest
            atomic_write_bytes(request_path, canonical_json(request) + b"\n")
            command = [
                str(python),
                "-m",
                "arena.adapters.eval_gimitest.worker",
                str(request_path),
                str(response_path),
            ]
            completed = run_supervised(
                command,
                timeout_seconds=float(isolation.get("timeout_seconds", 300)),
                max_stdout_bytes=int(isolation.get("max_stdout_bytes", 1_048_576)),
                max_stderr_bytes=int(isolation.get("max_stderr_bytes", 1_048_576)),
            )
            if completed.returncode != 0:
                raise ArenaError(
                    "Gimitest subprocess failed: "
                    f"exit={completed.returncode}, stderr={completed.stderr[-2000:]}"
                )
            if not response_path.is_file():
                raise ArenaError("Gimitest subprocess did not produce a response")
            response = load_manifest(response_path, max_bytes=16 * 1024 * 1024)
        if response.get("schema") != "arena.eval-provider-response/v1":
            raise ArenaError("Gimitest subprocess returned an unsupported response envelope")
        if response.get("request_id") != request["request_id"]:
            raise ArenaError("Gimitest subprocess response request_id mismatch")
        if response.get("request_digest") != request_digest:
            raise ArenaError("Gimitest subprocess response request digest mismatch")
        if response.get("ok") is not True or not isinstance(response.get("result"), dict):
            raise ArenaError("Gimitest subprocess returned an invalid success response")
        result = response["result"]
        if result.get("provider", {}).get("kind") != "gimitest":
            raise ArenaError("Gimitest subprocess response lost provider lineage")
        result["provider"]["worker"] = {
            "protocol": "arena.eval-provider/v1",
            "arena_version": response.get("arena_version"),
            "python": response.get("python"),
            "duration_seconds": round(completed.duration_seconds, 6),
        }
        return result
