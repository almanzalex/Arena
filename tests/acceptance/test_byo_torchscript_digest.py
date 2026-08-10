"""Acceptance: self-contained BYO TorchScript export → verify → inspect digests."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from arena.adapters.policy_custom_torch import (  # noqa: E402
    export_module_from_checkpoint,
    verify_bundle_self,
)
from arena.core.sdk import Policy  # noqa: E402
from examples.byo.cartpole_mlp import (  # noqa: E402
    CARTPOLE_ACTION,
    CARTPOLE_OBSERVATION,
    REFERENCE_CASES,
    build_actor,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _inspect_digest(bundle: Path) -> str:
    inspect = subprocess.run(
        [sys.executable, "-m", "arena", "inspect", str(bundle), "--json"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert inspect.returncode == 0, inspect.stderr
    payload = json.loads(inspect.stdout)
    assert payload["ok"] is True
    assert payload["digest"] == payload["data"]["digest"]
    return payload["digest"]


@pytest.mark.acceptance
@pytest.mark.requires_torch
def test_byo_export_verify_inspect_digest_stable(tmp_path: Path) -> None:
    """Published bundle identity is stable across load/inspect/copy.

    Independent TorchScript recompiles are *not* required to share a digest:
    ``torch.jit.save`` is not a byte-stable serializer across processes.
    """
    checkpoint = tmp_path / "cartpole.pt"
    torch.save(build_actor().state_dict(), checkpoint)

    bundle = export_module_from_checkpoint(
        module_ref="examples.byo.cartpole_mlp:build_actor",
        out_dir=tmp_path / "byo.arena",
        role="agent",
        name="byo-cartpole-mlp",
        observation=CARTPOLE_OBSERVATION,
        action=CARTPOLE_ACTION,
        source=checkpoint,
        reference_cases=REFERENCE_CASES,
        source_revision="examples/byo@cartpole-mlp",
    )
    verification = verify_bundle_self(bundle)
    assert verification["ok"] is True
    assert verification["verify_mode"] == "source-conformance"
    assert verification["cases"] == len(REFERENCE_CASES)

    policy = Policy.load(bundle)
    assert policy.digest.startswith("sha256:")
    assert policy.manifest["lineage"]["export_path"] == "byo-torchscript"
    assert policy.manifest["runtime"]["tier"] == "torchscript"

    digest_a = _inspect_digest(bundle)
    digest_b = _inspect_digest(bundle)
    assert digest_a == digest_b == policy.digest

    copied = tmp_path / "byo-copy.arena"
    shutil.copytree(bundle, copied)
    assert Policy.load(copied).digest == policy.digest
    assert _inspect_digest(copied) == policy.digest
    assert verify_bundle_self(copied)["ok"] is True


@pytest.mark.acceptance
@pytest.mark.requires_torch
def test_byo_export_script_cli_verify_inspect(tmp_path: Path) -> None:
    checkpoint = tmp_path / "shared.pt"
    torch.save(build_actor().state_dict(), checkpoint)
    out = tmp_path / "cli.arena"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "examples/byo/export_cartpole_mlp.py"),
            "--out",
            str(out),
            "--source",
            str(checkpoint),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["ok"] is True
    assert report["schema"] == "arena.byo-export-proof/v1"
    assert Policy.load(out).digest == report["policy_digest"]
    assert _inspect_digest(out) == report["policy_digest"]

    verify = subprocess.run(
        [sys.executable, "-m", "arena", "policy", "verify", str(out)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert verify.returncode == 0, verify.stderr


@pytest.mark.acceptance
@pytest.mark.requires_torch
def test_byo_export_script_no_source_demo_works(tmp_path: Path) -> None:
    out = tmp_path / "demo.arena"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "examples/byo/export_cartpole_mlp.py"),
            "--out",
            str(out),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["ok"] is True
    assert Policy.load(out).digest == report["policy_digest"]
    assert verify_bundle_self(out)["verify_mode"] == "source-conformance"
    assert _inspect_digest(out) == report["policy_digest"]


@pytest.mark.acceptance
def test_cleanrl_export_script_skips_without_checkout(tmp_path: Path) -> None:
    missing = tmp_path / "no-cleanrl-here"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "examples/1.0/export_cleanrl_cartpole.py"),
            "--cleanrl-checkout",
            str(missing),
            "--checkpoint",
            str(tmp_path / "missing.pt"),
            "--out",
            str(tmp_path / "out.arena"),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "CleanRL checkout" in combined
    assert "examples/byo/export_cartpole_mlp.py" in combined
