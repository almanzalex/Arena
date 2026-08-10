"""HF live qualification is fail-closed without tokens; simulation ≠ live pass."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

import pytest

from arena.core.errors import StoreError
from arena.core.mirror import build_mirror_artifact
from arena.core.store_hf import (
    HF_LIVE_RECIPE,
    HuggingFaceStoreAdapter,
    credential_missing_report,
    hf_live_credentials_present,
    qualify_hf_live,
)
from arena.core.support import load_support_matrix

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "examples" / "eval" / "demo" / "rock.arena"


def test_no_token_is_not_live_pass(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    assert hf_live_credentials_present() is False

    report_path = tmp_path / "hf-qualification.json"
    report = qualify_hf_live(
        SOURCE,
        "hf://models/ORG/REPO/arena",
        report_path=report_path,
    )
    assert report["schema"] == "arena.store-qualification/v1"
    assert report["backend"] == "hf"
    assert report["mode"] == "credential-missing"
    assert report["ok"] is False
    assert report["stable_claim_allowed"] is False
    assert report["checks"]["credentials"]["ok"] is False
    assert report["checks"]["live_round_trip"]["ok"] is False
    assert "HF_TOKEN" in report["repair"]
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["mode"] == "credential-missing"
    assert saved["ok"] is False


def test_credential_missing_report_never_claims_stable() -> None:
    report = credential_missing_report(
        source=SOURCE,
        destination="hf://models/ORG/REPO/arena",
    )
    assert report["mode"] == "credential-missing"
    assert report["ok"] is False
    assert report["stable_claim_allowed"] is False
    assert report["immutable_uri"] is None


def test_qualify_hf_live_refuses_simulate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HF_TOKEN", "canary-not-for-live")
    simulate = quote(str((tmp_path / "mirror").resolve()), safe="/")
    with pytest.raises(StoreError, match="refuses \\?simulate="):
        qualify_hf_live(
            SOURCE,
            f"hf://models/ORG/REPO/arena?simulate={simulate}",
            report_path=tmp_path / "should-not-exist.json",
        )
    assert not (tmp_path / "should-not-exist.json").exists()


def test_live_hf_push_without_token_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    artifact = build_mirror_artifact(SOURCE)
    with pytest.raises(StoreError, match="HF_TOKEN|credential") as exc:
        HuggingFaceStoreAdapter().push(
            artifact,
            "hf://models/ORG/REPO/arena",
            verify=False,
        )
    assert exc.value.code == "HF_CREDENTIALS_MISSING"
    assert not list(tmp_path.iterdir())


def test_qualify_hf_live_script_exits_nonzero_without_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    report_path = tmp_path / "script-report.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "qualify_hf_live.py"),
            str(SOURCE),
            "hf://models/ORG/REPO/arena",
            "--report",
            str(report_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={k: v for k, v in os.environ.items() if k not in HF_TOKEN_ENV},
    )
    assert proc.returncode != 0
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["mode"] == "credential-missing"
    assert report["ok"] is False
    assert "credential-missing" in proc.stderr or "did not pass" in proc.stderr


HF_TOKEN_ENV = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def test_support_matrix_hf_remains_preview_without_live_evidence() -> None:
    """This change must not flip hf→stable; matrix edits are out of scope here."""
    caps = load_support_matrix()["capabilities"]
    assert caps["hf"]["status"] == "preview"
    assert caps["hf"]["evidence"] == "none-attached"
    requires = (caps["hf"].get("stable_requires") or "").lower()
    assert "simulate" in requires
    assert "live" in requires or "credential" in requires or "immutable" in requires


@pytest.mark.requires_hf
def test_hf_live_round_trip_when_token_present(tmp_path: Path) -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    destination = os.environ.get("ARENA_HF_LIVE_DEST")
    if not token or not destination:
        pytest.skip(HF_LIVE_RECIPE)
    if "simulate" in destination:
        pytest.fail("ARENA_HF_LIVE_DEST must not include ?simulate=")
    report = qualify_hf_live(
        SOURCE,
        destination,
        report_path=tmp_path / "live.json",
        restored_out=tmp_path / "restored.arena",
    )
    assert report["mode"] == "live"
    assert report["ok"] is True
    assert report["stable_claim_allowed"] is True
    assert report["checks"]["immutable_revision"]["ok"] is True
