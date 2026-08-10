"""End-to-end: CartPole collect → BC train → verify → seeded match/eval."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("gymnasium")

from arena.adapters.policy_custom_torch import load_runtime, verify_bundle_self
from arena.cli.main import main as arena_main
from arena.core.sdk import Policy

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_mini_train_module():
    path = REPO_ROOT / "examples" / "1.0" / "mini_train_cartpole.py"
    spec = importlib.util.spec_from_file_location("mini_train_cartpole", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mini_train_cartpole"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.acceptance
@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_mini_train_cartpole_collect_train_verify_eval(tmp_path: Path) -> None:
    pytest.importorskip("pettingzoo")
    mod = _load_mini_train_module()
    out = tmp_path / "mini-train"
    result = mod.run_mini_train(
        out,
        episodes=12,
        epochs=25,
        seed=11,
        epsilon=0.1,
        eval_seeds=[301, 302, 303],
        match_seeds=[401, 402],
    )
    assert result["ok"] is True
    assert result["policy_digest"]
    assert result["dataset_digest"]
    assert result["recipe_digest"]
    assert result["training_contract_digest"]
    assert result["train"]["examples"] > 0
    assert result["train"]["loss_final"] <= result["train"]["loss_initial"] + 1e-6
    assert result["verification"]["verify_mode"] == "source-conformance"

    bundle = Path(result["policy_path"])
    assert bundle.is_dir()
    policy = Policy.load(bundle)
    assert policy.digest == result["policy_digest"]
    assert verify_bundle_self(bundle)["verify_mode"] == "source-conformance"
    assert arena_main(["policy", "verify", str(bundle)]) == 0

    runtime = load_runtime(bundle)
    action = runtime.act([0.0, 0.0, 0.05, 0.1], mode="deterministic", agent_id="agent")
    assert action in {0, 1}

    gym_eval = result["gymnasium_eval"]
    assert gym_eval["mean_return"] > result["random_baseline_mean_return"]

    assert result["match"] is not None
    assert result["match"]["outcome"]["episodes_completed"] == 2
    assert result["match"]["outcome"]["failure_count"] == 0
    assert (out / "match-run" / "run.yaml").is_file()
    assert (out / "match-run" / "trajectories").is_dir()
    assert (out / "result.json").is_file()
    saved = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert saved["policy_digest"] == result["policy_digest"]


@pytest.mark.acceptance
@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_mini_train_cartpole_cli_script(tmp_path: Path) -> None:
    pytest.importorskip("pettingzoo")
    out = tmp_path / "cli-out"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "examples" / "1.0" / "mini_train_cartpole.py"),
            "--out",
            str(out),
            "--episodes",
            "10",
            "--epochs",
            "20",
            "--seed",
            "3",
        ],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr[-2000:] + "\n" + proc.stdout[-2000:]
    result_path = out / "result.json"
    assert result_path.is_file()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["ok"] is True
    assert (out / "train-run" / "policy.arena" / "DIGEST").is_file()
