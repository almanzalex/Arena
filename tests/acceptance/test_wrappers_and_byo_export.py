"""Acceptance: wrapper contract + BYO TorchScript export CLI (Pistonball-shaped)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from rlx.adapters.policy_custom_torch import load_runtime, verify_bundle_self
from rlx.adapters.task_pettingzoo.wrappers import apply_wrappers, normalize_wrappers
from rlx.core.errors import SchemaError
from rlx.core.sdk import Policy, Task, check

torch = pytest.importorskip("torch")
nn = torch.nn


def make_tiny_image_env(shape=(32, 48, 3), n_actions: int = 3):
    """Minimal Parallel API env with RGB Box obs — mirrors Pistonball wrapper needs."""
    import gymnasium
    from pettingzoo.utils.env import ParallelEnv

    class TinyImageParallelEnv(ParallelEnv):
        metadata = {"name": "tiny_image_v0", "render_modes": []}

        def __init__(self) -> None:
            super().__init__()
            self.possible_agents = ["agent_0", "agent_1"]
            self.agents = list(self.possible_agents)
            self._shape = shape
            self._obs_space = gymnasium.spaces.Box(
                low=0, high=255, shape=shape, dtype=np.uint8
            )
            self._act_space = gymnasium.spaces.Discrete(n_actions)
            self._step = 0

        def observation_space(self, agent: str):
            return self._obs_space

        def action_space(self, agent: str):
            return self._act_space

        def reset(self, seed=None, options=None):
            self.agents = list(self.possible_agents)
            self._step = 0
            if seed is not None:
                np.random.seed(int(seed))
            obs = {a: np.zeros(self._shape, dtype=np.uint8) for a in self.agents}
            return obs, {a: {} for a in self.agents}

        def step(self, actions):
            self._step += 1
            obs = {
                a: np.full(self._shape, self._step % 255, dtype=np.uint8)
                for a in self.agents
            }
            rewards = {a: 0.0 for a in self.agents}
            terms = {a: False for a in self.agents}
            truncs = {a: self._step >= 4 for a in self.agents}
            infos = {a: {} for a in self.agents}
            if any(truncs.values()):
                self.agents = []
            return obs, rewards, terms, truncs, infos

        def close(self) -> None:
            return None

    return TinyImageParallelEnv()


class TinyCnnActor(nn.Module):
    """Compact CNN: CHW stacked frames → Discrete logits (Pistonball-shaped)."""

    def __init__(self, in_channels: int = 4, n_actions: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1, bias=False)
        self.head = nn.Linear(4 * 8 * 8, n_actions, bias=False)
        with torch.no_grad():
            self.conv.weight.fill_(0.01)
            self.head.weight.fill_(0.02)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        # /255 inside the network (CleanRL-faithful)
        x = self.conv(observation / 255.0)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (8, 8))
        return self.head(x.flatten(1))


def build_tiny_cnn_actor(in_channels: int = 4, n_actions: int = 3) -> TinyCnnActor:
    return TinyCnnActor(in_channels=in_channels, n_actions=n_actions)


PISTONBALL_LIKE_WRAPPERS = [
    {"op": "color_reduction", "mode": "full"},
    {"op": "resize", "x_size": 16, "y_size": 16},
    {"op": "frame_stack", "stack_size": 4},
]


def _write_factory_module(directory: Path) -> Path:
    """Write an importable factory module for subprocess CLI export."""
    path = directory / "rlx_test_byo_actor.py"
    path.write_text(
        '''"""Test-only BYO actor factory (exporter-side)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyCnnActor(nn.Module):
    def __init__(self, in_channels: int = 4, n_actions: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1, bias=False)
        self.head = nn.Linear(4 * 8 * 8, n_actions, bias=False)
        with torch.no_grad():
            self.conv.weight.fill_(0.01)
            self.head.weight.fill_(0.02)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        x = self.conv(observation / 255.0)
        x = F.adaptive_avg_pool2d(x, (8, 8))
        return self.head(x.flatten(1))


def build_tiny_cnn_actor(in_channels: int = 4, n_actions: int = 3) -> TinyCnnActor:
    return TinyCnnActor(in_channels=in_channels, n_actions=n_actions)
''',
        encoding="utf-8",
    )
    return path


@pytest.mark.requires_pettingzoo
def test_wrapper_chain_changes_space_and_wrong_order_differs() -> None:
    pytest.importorskip("supersuit")
    env = make_tiny_image_env(shape=(32, 48, 3))
    wrapped = apply_wrappers(env, PISTONBALL_LIKE_WRAPPERS)
    try:
        obs, _ = wrapped.reset(seed=0)
        shape = wrapped.observation_space("agent_0").shape
        assert shape == (16, 16, 4), shape
        assert obs["agent_0"].shape == (16, 16, 4)
    finally:
        wrapped.close()

    # Wrong order / missing resize must not silently yield the training shape.
    bad = apply_wrappers(
        make_tiny_image_env(shape=(32, 48, 3)),
        [
            {"op": "color_reduction", "mode": "full"},
            {"op": "frame_stack", "stack_size": 4},
        ],
    )
    try:
        bad_shape = bad.observation_space("agent_0").shape
        assert bad_shape != (16, 16, 4)
        assert bad_shape[-1] == 4  # stacked, but not resized
    finally:
        bad.close()


@pytest.mark.requires_pettingzoo
def test_unknown_wrapper_never_applied_silently() -> None:
    with pytest.raises(SchemaError, match="unknown task wrapper"):
        normalize_wrappers([{"op": "dtype_v0"}])
    with pytest.raises(SchemaError, match="unknown task wrapper"):
        apply_wrappers(make_tiny_image_env(), [{"op": "dtype_v0"}])


@pytest.mark.requires_torch
def test_byo_torchscript_cli_export_verify_clean_room(tmp_path: Path) -> None:
    factory_dir = tmp_path / "factory"
    factory_dir.mkdir()
    _write_factory_module(factory_dir)

    actor = build_tiny_cnn_actor()
    ckpt = tmp_path / "actor.pt"
    torch.save(actor.state_dict(), ckpt)

    # HWC stacked observation cases (env-side wrappers already applied).
    rng = np.random.default_rng(0)
    cases = [
        {
            "observation": rng.integers(0, 256, size=(16, 16, 4), dtype=np.uint8).tolist(),
            "mode": "deterministic",
        }
        for _ in range(4)
    ]
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps(cases), encoding="utf-8")

    spec = {
        "observation": {
            "type": "Box",
            "shape": [16, 16, 4],
            "dtype": "uint8",
            "layout": "HWC",
            "low": 0,
            "high": 255,
        },
        "action": {"type": "Discrete", "n": 3, "masks": "none"},
        "architecture": {"type": "serialized_module", "io": {"recurrent": False}},
        "preprocessing": {
            "id": "pistonball_like_layout",
            "pipeline": {
                "version": "rlx.preprocess/v1",
                "steps": [{"op": "layout", "from": "HWC", "to": "CHW"}],
            },
        },
    }
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")

    out = tmp_path / "actor.rlx"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(factory_dir) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable,
        "-m",
        "rlx",
        "policy",
        "export",
        "--adapter",
        "custom-pytorch",
        "--module",
        "rlx_test_byo_actor:build_tiny_cnn_actor",
        "--source",
        str(ckpt),
        "--role",
        "agent_0",
        "--spec",
        str(spec_path),
        "--reference-cases",
        str(cases_path),
        "--source-revision",
        "testrev",
        "--wrappers-identity",
        "color_reduction(full)>resize(16,16)>frame_stack(4)",
        "--out",
        str(out),
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False, env=env)
    assert proc.returncode == 0, proc.stderr
    assert out.is_dir()

    verify = verify_bundle_self(out)
    assert verify["verify_mode"] == "source-conformance"

    manifest = yaml.safe_load((out / "policy.yaml").read_text(encoding="utf-8"))
    assert manifest["lineage"]["source_revision"] == "testrev"
    assert "checkpoint_digest" in manifest["lineage"]
    assert "frame_stack(4)" in manifest["lineage"]["wrappers_identity"]

    rt = load_runtime(out)
    obs = np.asarray(cases[0]["observation"], dtype=np.uint8)
    action = rt.act(obs)
    assert isinstance(action, int) and 0 <= action < 3

    # Clean-room child: no factory dir on PYTHONPATH.
    child = (
        "from rlx.adapters.policy_custom_torch import load_runtime; "
        "import numpy as np; "
        f"rt=load_runtime({str(out)!r}); "
        f"print(rt.act(np.asarray({cases[0]['observation']!r}, dtype=np.uint8)))"
    )
    clean_env = {k: v for k, v in os.environ.items() if "PYTHONPATH" not in k}
    result = subprocess.run(
        [sys.executable, "-c", child],
        env=clean_env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.strip().isdigit()

    policy = Policy.load(out)
    from rlx.core.compatibility import compose_check

    report = compose_check(
        policy=policy.manifest,
        role="agent_0",
        expected_obs={
            "type": "Box",
            "shape": [16, 16, 4],
            "dtype": "uint8",
            "layout": "CHW",  # wrong vs policy HWC
            "low": 0,
            "high": 255,
        },
        expected_act={"type": "Discrete", "n": 3},
    )
    assert not report.ok
    assert any(i.code == "OBSERVATION_MISMATCH" for i in report.issues)


@pytest.mark.requires_pettingzoo
@pytest.mark.requires_torch
def test_wrapped_task_check_matches_policy_space(tmp_path: Path, monkeypatch) -> None:
    """Semi-real path: SuperSuit wrappers on tiny env + BYO policy space agreement."""
    pytest.importorskip("supersuit")

    # Patch make_env to return our tiny env so we don't need Pistonball deps.
    import rlx.adapters.task_pettingzoo.adapter as adapter

    def _fake_make(spec):
        normalize_wrappers(spec.get("wrappers"))
        env = make_tiny_image_env(shape=(32, 48, 3))
        return apply_wrappers(env, spec.get("wrappers"))

    monkeypatch.setattr(adapter, "make_env", _fake_make)

    task = Task.load(
        {
            "adapter": "pettingzoo-parallel",
            "env": "tiny/image_v0",
            "observation_layout": "HWC",
            "wrappers": PISTONBALL_LIKE_WRAPPERS,
            "config": {"continuous": False},
        }
    )
    info = task.role_spaces()
    assert info["wrappers"]["identity"].endswith("frame_stack(4)")
    obs_space = info["roles"]["agent_0"]["observation"]
    assert obs_space["shape"] == [16, 16, 4]
    assert obs_space["layout"] == "HWC"

    factory_dir = tmp_path / "factory"
    factory_dir.mkdir()
    _write_factory_module(factory_dir)
    sys.path.insert(0, str(factory_dir))

    from rlx.adapters.policy_custom_torch import export_module_from_checkpoint

    actor = build_tiny_cnn_actor()
    ckpt = tmp_path / "w.pt"
    torch.save(actor.state_dict(), ckpt)
    cases = [
        {
            "observation": np.zeros((16, 16, 4), dtype=np.uint8).tolist(),
            "mode": "deterministic",
        }
    ]
    bundle = export_module_from_checkpoint(
        module_ref="rlx_test_byo_actor:build_tiny_cnn_actor",
        out_dir=tmp_path / "p.rlx",
        role="agent_0",
        observation=obs_space,
        action={"type": "Discrete", "n": 3, "masks": "none"},
        source=ckpt,
        preprocessing={
            "pipeline": {
                "version": "rlx.preprocess/v1",
                "steps": [{"op": "layout", "from": "HWC", "to": "CHW"}],
            }
        },
        reference_cases=cases,
        wrappers_identity=info["wrappers"]["identity"],
    )
    policy = Policy.load(bundle)
    check(task, policy.as_role("agent_0")).raise_for_errors()

    # Missing wrappers → wrong shape → fail loud.
    bare = Task.load(
        {
            "adapter": "pettingzoo-parallel",
            "env": "tiny/image_v0",
            "observation_layout": "HWC",
            "wrappers": [],
        }
    )
    report = check(bare, policy.as_role("agent_0"))
    assert not report.ok
    assert any(i.code == "OBSERVATION_MISMATCH" for i in report.issues)
