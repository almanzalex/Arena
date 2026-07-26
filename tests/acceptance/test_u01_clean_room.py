"""Scripted clean-room U-01 validation driving the real CLI end-to-end from bundles."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pettingzoo")

from arena.adapters.policy_custom_torch import build_module  # noqa: E402

_EXPORT_SPEC = """\
architecture:
  type: mlp_categorical
  observation_dim: 4
  hidden_dims: [32, 32]
  action_n: 3
observation:
  type: Discrete
  n: 4
  dtype: int64
action:
  type: Discrete
  n: 3
  dtype: int64
  masks: none
preprocessing:
  id: normalize_v0
  mean: 0.0
  std: 1.0
"""

_MATCH_YAML = """\
schema: arena.match/v0alpha1
task:
  adapter: pettingzoo-parallel
  env: arena/competitive_rps_v0
assignments:
  player_0: ./player_0.arena
  player_1: ./player_1.arena
seeds: {start: 0, count: 5}
action_mode: deterministic
record:
  trajectories: all
failure_policy:
  timeout_seconds: 30
  retain_incomplete: true
  retry: 0
"""


def _make_checkpoint(path: Path, *, seed: int) -> None:
    """Produce a trainer checkpoint (deleted before the clean-room phase)."""
    arch = {"type": "mlp_categorical", "observation_dim": 4, "hidden_dims": [32, 32], "action_n": 3}
    torch.manual_seed(seed)
    torch.save({"state_dict": build_module(arch).state_dict()}, path)


@pytest.mark.acceptance
@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_u01_scripted_clean_room(tmp_path: Path) -> None:
    """End-to-end U-01 via the documented CLI only.

    Researcher A: ``arena policy export`` two policies from checkpoints, then
    ``arena policy verify`` each. Researcher B: receive *only* the ``.arena`` bundles in a
    fresh directory (trainer repo + checkpoints deleted, trainer off ``PYTHONPATH``) and
    run ``arena init`` → ``inspect`` → ``check`` → ``match run --record`` → ``data inspect``.

    The only remaining residual is the genuine human step documented in docs/clean-room.md:
    a second person repeating this on a machine that never had the trainer checkout.
    """
    author = tmp_path / "author"
    author.mkdir()
    spec = author / "export_spec.yaml"
    spec.write_text(_EXPORT_SPEC, encoding="utf-8")
    _make_checkpoint(author / "ckpt_p0.pt", seed=1)
    _make_checkpoint(author / "ckpt_p1.pt", seed=2)

    def cli(*args: str, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "arena.cli.main", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )

    # --- Researcher A: export + verify via the real CLI ---
    r = cli("init", cwd=author)
    assert r.returncode == 0, r.stderr
    for role, ckpt in (("player_0", "ckpt_p0.pt"), ("player_1", "ckpt_p1.pt")):
        r = cli(
            "policy", "export",
            "--adapter", "custom-pytorch",
            "--source", f"./{ckpt}",
            "--role", role,
            "--spec", "./export_spec.yaml",
            "--out", f"./{role}.arena",
            cwd=author,
        )
        assert r.returncode == 0, r.stderr
        r = cli("policy", "verify", f"./{role}.arena", cwd=author)
        assert r.returncode == 0, r.stderr + r.stdout
        assert '"ok": true' in r.stdout, r.stdout

    # --- Handoff: copy ONLY the bundles into a clean room; destroy everything trainer-side ---
    clean = tmp_path / "clean_room"
    clean.mkdir()
    for role in ("player_0", "player_1"):
        shutil.copytree(author / f"{role}.arena", clean / f"{role}.arena")
    (clean / "match.yaml").write_text(_MATCH_YAML, encoding="utf-8")
    shutil.rmtree(author)  # trainer, checkpoints, spec, and .arena workspace all gone

    # Clean environment: nothing trainer-side on PYTHONPATH.
    clean_env = {**os.environ, "PYTHONPATH": ""}

    def clean_cli(*args: str) -> subprocess.CompletedProcess:
        return cli(*args, cwd=clean, env=clean_env)

    # --- Researcher B: documented CLI flow, bundles only ---
    r = clean_cli("init")
    assert r.returncode == 0, r.stderr

    for role in ("player_0", "player_1"):
        r = clean_cli("inspect", f"./{role}.arena")
        assert r.returncode == 0, r.stderr
        r = clean_cli("check", "arena/competitive_rps_v0", f"./{role}.arena", "--role", role)
        assert r.returncode == 0, r.stderr + r.stdout
        assert "COMPATIBLE" in r.stdout, r.stdout

    out = clean / "runs" / "baseline"
    r = clean_cli("match", "run", "./match.yaml", "--record", "--out", str(out))
    assert r.returncode == 0, r.stderr + r.stdout
    assert "failures=0" in r.stdout, r.stdout

    r = clean_cli("data", "inspect", str(out / "trajectories"))
    assert r.returncode == 0, r.stderr + r.stdout

    # Artifacts prove the handoff succeeded from bundles alone.
    assert (out / "run.yaml").exists()
    assert (out / "trajectories" / "bundle.yaml").exists()
    assert list((out / "trajectories").glob("episode_*.json"))
