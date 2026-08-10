"""Isolated OCI / W&B / MLflow qualification harnesses (no live credentials)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import quote

import pytest

from arena.cli.main import main
from arena.conformance.qualification import qualify_store
from arena.core.errors import StoreError
from arena.core.mirror import push_artifact
from arena.core.store_mlflow import MLflowStoreAdapter
from arena.core.store_oci import OCIStoreAdapter
from arena.core.store_preview import PREVIEW_STORES, mode_and_live_claim
from arena.core.store_wandb import WandBStoreAdapter

SOURCE = Path("examples/eval/demo/rock.arena").resolve()


@pytest.mark.parametrize(
    ("scheme", "base"),
    [
        ("oci", "oci://registry.example/lab/arena"),
        ("wandb", "wandb://lab/project/arena"),
        ("mlflow", "mlflow://arena-experiment"),
    ],
)
def test_preview_store_simulation_qualify_never_counts_as_live(
    tmp_path: Path, scheme: str, base: str
) -> None:
    assert scheme in PREVIEW_STORES
    destination = f"{base}?simulate={quote(str((tmp_path / scheme).resolve()), safe='/')}"
    report = qualify_store(
        SOURCE,
        destination=destination,
        report_path=tmp_path / f"{scheme}-qualification.json",
    )
    assert report["ok"] is True
    assert report["backend"] == scheme
    assert report["mode"] == "simulation"
    assert report["counts_as_live_evidence"] is False
    assert report["checks"]["simulation_never_live"]["ok"] is True
    mode, live = mode_and_live_claim(destination, report["immutable_uri"])
    assert mode == "simulation"
    assert live is False
    assert "simulate=" in report["immutable_uri"]


def test_oci_live_without_oras_fails_loud(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("arena.core.store_oci.shutil.which", lambda _name: None)
    with pytest.raises(StoreError) as exc:
        push_artifact(SOURCE, "oci://registry.example/lab/arena", verify=False)
    err = exc.value
    assert err.code == "STORE_CREDENTIALS_REQUIRED"
    assert "simulate" in (err.repair or "").lower() or "simulate" in str(err).lower()
    assert "docs/qualifications/oci" in str(err) or "docs/qualifications/oci" in (
        err.repair or ""
    )


def test_wandb_live_without_credentials_fails_loud(tmp_path: Path, monkeypatch) -> None:
    del tmp_path
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    class _FakeWandb:
        class api:  # noqa: N801
            api_key = None

        class Api:  # noqa: N801
            def __init__(self) -> None:
                self.api_key = None

    monkeypatch.setattr(
        WandBStoreAdapter,
        "_wandb",
        staticmethod(lambda: _FakeWandb),
    )
    with pytest.raises(StoreError) as exc:
        push_artifact(SOURCE, "wandb://lab/project/arena", verify=False)
    err = exc.value
    assert err.code == "STORE_CREDENTIALS_REQUIRED"
    assert err.context.get("counts_as_live_evidence") is False


def test_mlflow_live_without_tracking_uri_fails_loud(tmp_path: Path, monkeypatch) -> None:
    del tmp_path
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(
        MLflowStoreAdapter,
        "_mlflow",
        staticmethod(lambda: object()),
    )
    with pytest.raises(StoreError) as exc:
        push_artifact(SOURCE, "mlflow://arena-experiment", verify=False)
    err = exc.value
    assert err.code == "STORE_CREDENTIALS_REQUIRED"
    assert "tracking" in str(err).lower()


def test_mlflow_file_tracking_uri_is_not_live(monkeypatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(
        MLflowStoreAdapter,
        "_mlflow",
        staticmethod(lambda: object()),
    )
    with pytest.raises(StoreError) as exc:
        push_artifact(
            SOURCE,
            "mlflow://arena-experiment?tracking_uri=file%3A%2F%2F%2Ftmp%2Fmlruns",
            verify=False,
        )
    assert exc.value.code == "STORE_CREDENTIALS_REQUIRED"


def test_cli_simulation_qualify_writes_non_live_report(tmp_path: Path) -> None:
    out = tmp_path / "oci-qualification.json"
    destination = (
        "oci://registry.example/lab/cli"
        f"?simulate={quote(str((tmp_path / 'mirror').resolve()), safe='/')}"
    )
    assert (
        main(
            [
                "store",
                "qualify",
                str(SOURCE),
                destination,
                "--out",
                str(out),
            ]
        )
        == 0
    )
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["mode"] == "simulation"
    assert report["counts_as_live_evidence"] is False


def test_no_live_evidence_checked_in_yet() -> None:
    """Stable promotion is gated on real live reports; none are invented here."""
    root = Path("docs/qualifications")
    for backend in sorted(PREVIEW_STORES):
        live = root / backend / "live-qualification.json"
        assert not live.exists(), f"unexpected live evidence for {backend}: {live}"
        readme = root / backend / "README.md"
        assert readme.is_file()
        text = readme.read_text(encoding="utf-8")
        assert "preview" in text.lower()
        assert "never" in text.lower() and "live" in text.lower()


def test_adapters_are_extracted_modules() -> None:
    assert OCIStoreAdapter.__module__ == "arena.core.store_oci"
    assert WandBStoreAdapter.__module__ == "arena.core.store_wandb"
    assert MLflowStoreAdapter.__module__ == "arena.core.store_mlflow"


def test_env_canaries_are_not_required_for_simulation(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    for key in list(os.environ):
        if key.upper().startswith("ORAS"):
            monkeypatch.delenv(key, raising=False)
    destination = (
        "wandb://lab/project/arena"
        f"?simulate={quote(str((tmp_path / 'sim').resolve()), safe='/')}"
    )
    report = qualify_store(SOURCE, destination=destination)
    assert report["counts_as_live_evidence"] is False
    assert report["ok"] is True
