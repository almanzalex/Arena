"""Fail-loud lab mistake UX: errors include next-step recipes."""

from __future__ import annotations

import pytest

from arena.core.errors import (
    SchemaError,
    bad_uri,
    diagnostic_from_exception,
    missing_extra,
)
from arena.core.identity import parse_digest
from arena.core.manifests import _require_digest, validate_policy_manifest
from arena.core.registry import ACTION_CASES, UnknownKindError, ensure_plugins_loaded


def test_wrong_schema_includes_supported_and_repair() -> None:
    with pytest.raises(SchemaError) as exc:
        validate_policy_manifest({"schema": "arena.policy/v999"})
    err = exc.value
    assert err.code == "SCHEMA_VERSION_UNSUPPORTED"
    assert "arena.policy/v0alpha1" in str(err)
    assert err.repair
    assert "schema" in err.repair.lower() or "doctor" in err.repair.lower()
    diag = diagnostic_from_exception(err)
    assert diag["repair"]
    assert diag["code"] == "SCHEMA_VERSION_UNSUPPORTED"


def test_missing_extra_includes_install_recipe() -> None:
    err = missing_extra("torch", feature="arena train")
    assert err.code == "CAPABILITY_MISSING"
    assert "pip install" in str(err)
    assert "arena[torch]" in str(err)
    assert "doctor" in (err.repair or "").lower()
    diag = diagnostic_from_exception(err)
    assert diag["code"] == "CAPABILITY_MISSING"
    assert "pip install" in diag["repair"]


def test_torch_adapter_missing_extra_path_uses_helper() -> None:
    from arena.adapters.policy_custom_torch import _require_torch
    from arena.core.errors import SchemaError

    # If torch is installed this still validates the helper shape via missing_extra factory.
    err = missing_extra("torch", feature="custom-pytorch adapter", capability="torch")
    assert err.code == "CAPABILITY_MISSING"
    assert "arena[torch]" in str(err)
    assert callable(_require_torch)
    assert isinstance(err, SchemaError)


def test_unknown_kind_includes_extension_recipe_and_repair() -> None:
    ensure_plugins_loaded()
    with pytest.raises(UnknownKindError) as exc:
        ACTION_CASES.get("NotARealAction")
    err = exc.value
    assert "qualify" in str(err)
    assert err.repair and "register" in err.repair
    assert err.code == "UNKNOWN_KIND"
    diag = diagnostic_from_exception(err)
    assert diag["repair"]


def test_missing_digest_includes_sha256_recipe() -> None:
    with pytest.raises(SchemaError) as exc:
        _require_digest("not-a-digest", field="payloads.weights")
    err = exc.value
    assert err.code == "DIGEST_MISSING"
    assert "sha256" in str(err).lower()
    assert err.repair and "sha256" in err.repair


def test_parse_digest_bad_algorithm_is_actionable() -> None:
    with pytest.raises(SchemaError) as exc:
        parse_digest("md5:deadbeef")
    assert "sha256" in str(exc.value).lower()
    assert exc.value.repair
    assert exc.value.code == "DIGEST_INVALID"


def test_bad_uri_helper_includes_example_and_repair() -> None:
    err = bad_uri(
        "openenv URI must use https",
        scheme="openenv",
        example="openenv+https://host/env",
    )
    assert err.code == "URI_INVALID"
    assert "Example:" in str(err)
    assert err.repair


def test_cli_unsupported_adapter_is_actionable() -> None:
    import argparse

    from arena.cli.main import cmd_policy_export

    args = argparse.Namespace(adapter="not-a-real-adapter")
    with pytest.raises(SchemaError) as exc:
        cmd_policy_export(args)
    err = exc.value
    assert err.code == "UNKNOWN_KIND"
    assert "custom-pytorch" in str(err)
    assert err.repair
    diag = diagnostic_from_exception(err)
    assert diag["code"] == "UNKNOWN_KIND"
    assert diag["repair"]
