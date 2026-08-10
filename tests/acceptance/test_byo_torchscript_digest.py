"""Acceptance: self-contained BYO TorchScript export → verify → inspect digests."""

from __future__ import annotations

import json
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


@pytest.mark.acceptance
@pytest.mark.requires_torch
def test_byo_export_verify_inspect_digest_stable(tmp_path: Path) -> None:
    checkpoint = tmp_path / "cartpole.pt"
    torch.save(build_actor().state_dict(), checkpoint)

    digests: list[str] = []
    for i in range(2):
        bundle = export_module_from_checkpoint(
            module_ref="examples.byo.cartpole_mlp:build_actor",
            out_dir=tmp_path / f"byo-{i}.arena",
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
        digests.append(policy.digest)

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
        assert payload["digest"] == policy.digest
        assert payload["data"]["digest"] == policy.digest
        assert payload["data"]["runtime"]["tier"] == "torchscript"
        assert payload["data"]["lineage"]["export_path"] == "byo-torchscript"

    assert digests[0] == digests[1]
    assert digests[0].startswith("sha256:")


@pytest.mark.acceptance
@pytest.mark.requires_torch
def test_byo_export_script_cli_digest_stable(tmp_path: Path) -> None:
    # Share one checkpoint across CLI runs so digests isolate TorchScript
    # serialization stability (not RNG re-init of demo weights).
    checkpoint = tmp_path / "shared.pt"
    torch.save(build_actor().state_dict(), checkpoint)
    out_a = tmp_path / "a.arena"
    out_b = tmp_path / "b.arena"
    digests: list[str] = []
    for out in (out_a, out_b):
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
        digests.append(report["policy_digest"])
        assert Policy.load(out).digest == report["policy_digest"]
        verify = subprocess.run(
            [sys.executable, "-m", "arena", "policy", "verify", str(out)],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert verify.returncode == 0, verify.stderr
    assert digests[0] == digests[1]


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
