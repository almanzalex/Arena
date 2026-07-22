"""Acceptance tests P-01–P-05 for portable policy contract."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rlx.adapters.policy_custom_torch import (  # noqa: E402
    load_runtime,
    verify_bundle_self,
)
from rlx.conformance.fixtures import (  # noqa: E402
    build_f1_deterministic,
    build_f2_stochastic,
    build_f3_recurrent,
    build_f4_masked,
)
from rlx.core.errors import ConformanceError  # noqa: E402


@pytest.mark.acceptance
@pytest.mark.requires_torch
def test_p01_deterministic_equivalence(tmp_path: Path) -> None:
    bundle = build_f1_deterministic(tmp_path / "f1")
    # Source runtime = first load; exported = second load (same bytes)
    source = load_runtime(bundle)
    exported = load_runtime(bundle)
    cases = json.loads((bundle / "payloads" / "reference_cases.json").read_text())["cases"]
    for case in cases:
        source.reset()
        exported.reset()
        s = source.act(case["observation"], mode="deterministic")
        e = exported.act(case["observation"], mode="deterministic")
        assert s == e == case["expected_action"]
    assert verify_bundle_self(bundle, allow_self_consistency=True)["ok"]


@pytest.mark.acceptance
@pytest.mark.requires_torch
def test_p02_seeded_stochastic_equivalence(tmp_path: Path) -> None:
    bundle = build_f2_stochastic(tmp_path / "f2")
    source = load_runtime(bundle)
    exported = load_runtime(bundle)
    cases = json.loads((bundle / "payloads" / "reference_cases.json").read_text())["cases"]
    for case in cases:
        source.reset()
        exported.reset()
        s = source.act(
            case["observation"],
            mode="stochastic",
            rng=np.random.default_rng(case["seed"]),
        )
        e = exported.act(
            case["observation"],
            mode="stochastic",
            rng=np.random.default_rng(case["seed"]),
        )
        assert s == e == case["expected_action"]
    assert verify_bundle_self(bundle, allow_self_consistency=True)["ok"]


@pytest.mark.acceptance
@pytest.mark.requires_torch
def test_p03_recurrent_lifecycle(tmp_path: Path) -> None:
    bundle = build_f3_recurrent(tmp_path / "f3")
    rt = load_runtime(bundle)
    cases = json.loads((bundle / "payloads" / "reference_cases.json").read_text())["cases"]
    # Replay stream without resetting between non-reset cases
    for case in cases:
        if case.get("hidden_reset"):
            rt.reset(case.get("agent_id", "default"))
        action = rt.act(
            case["observation"],
            mode="deterministic",
            agent_id=case.get("agent_id", "default"),
        )
        assert action == case["expected_action"]
    # Explicit: after reset, action equals first-step action of a fresh runtime
    stream0 = cases[0]["observation"]
    a = load_runtime(bundle)
    b = load_runtime(bundle)
    a.reset("a0")
    b.reset("a0")
    # Run two steps on a
    a.act(stream0, mode="deterministic", agent_id="a0")
    a.act(cases[1]["observation"], mode="deterministic", agent_id="a0")
    a.reset("a0")
    assert a.act(stream0, mode="deterministic", agent_id="a0") == b.act(
        stream0, mode="deterministic", agent_id="a0"
    )
    assert verify_bundle_self(bundle, allow_self_consistency=True)["ok"]


@pytest.mark.acceptance
@pytest.mark.requires_torch
def test_p04_action_mask_handling(tmp_path: Path) -> None:
    bundle = build_f4_masked(tmp_path / "f4")
    rt = load_runtime(bundle)
    # Missing mask must error
    rt.reset()
    with pytest.raises(ConformanceError, match="action mask required"):
        rt.act(np.ones(4, dtype=np.float32), mode="deterministic")
    # Legal-only actions
    for mask in ([True, False, False, False], [False, False, True, False]):
        rt.reset()
        action = rt.act(
            np.ones(4, dtype=np.float32),
            mode="deterministic",
            action_mask=np.asarray(mask),
        )
        assert mask[action]
    assert verify_bundle_self(bundle, allow_self_consistency=True)["ok"]


@pytest.mark.acceptance
@pytest.mark.requires_torch
def test_p01_independent_source_vs_export(tmp_path: Path) -> None:
    """Genuine source-vs-export: an independent source model (direct forward+argmax)
    must match the exported RLX runtime's full act() pipeline on fixed observations."""
    from rlx.adapters.policy_custom_torch import build_module, export_from_checkpoint

    arch = {"type": "mlp_categorical", "observation_dim": 4, "hidden_dims": [16, 16], "action_n": 3}
    torch.manual_seed(1234)
    source = build_module(arch)
    source.eval()
    ckpt = tmp_path / "checkpoint.pt"
    torch.save({"state_dict": source.state_dict()}, ckpt)

    bundle = export_from_checkpoint(
        source=ckpt,
        out=tmp_path / "exported.rlx",
        role="agent",
        architecture=arch,
        observation={"type": "Discrete", "n": 4, "dtype": "int64"},
        action={"type": "Discrete", "n": 3, "dtype": "int64", "masks": "none"},
        preprocessing={"id": "normalize_v0", "mean": 0.0, "std": 1.0},
    )
    runtime = load_runtime(bundle)

    def onehot(i: int) -> torch.Tensor:
        v = np.zeros(4, dtype=np.float32)
        v[i] = 1.0
        return torch.as_tensor(v).view(1, -1)

    for obs in range(4):
        with torch.no_grad():
            logits, _ = source(onehot(obs))
            source_action = int(torch.argmax(logits, dim=-1).item())
        runtime.reset()
        exported_action = runtime.act(obs, mode="deterministic")
        assert exported_action == source_action, f"obs={obs}"

    # And the export auto-embedded reference cases that verify cleanly (fixes broken doc flow).
    assert verify_bundle_self(bundle, allow_self_consistency=True)["ok"]


_TRAINER_PKG = '''\
"""A stand-in training repository whose model class lives ONLY here."""

import torch
import torch.nn as nn


class MappoActor(nn.Module):
    def __init__(self, obs_dim=4, action_n=3, hidden=(16, 16)):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers.append(nn.Linear(prev, action_n))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def save_checkpoint(path):
    torch.manual_seed(0)
    torch.save({"state_dict": MappoActor().state_dict()}, path)
'''

# Loads the exported bundle and acts WITHOUT importing the trainer. Also proves the
# trainer package is genuinely unreachable in this process.
_CLEAN_RUN = '''\
import sys
from rlx.adapters.policy_custom_torch import load_runtime

rt = load_runtime(sys.argv[1])
rt.reset()
action = rt.act(0, mode="deterministic")
assert action in (0, 1, 2), action
try:
    import trainer_pkg  # noqa: F401
    print("TRAINER_PRESENT")
except ModuleNotFoundError:
    print("TRAINER_ABSENT")
print("ACTION", action)
'''

# Control: an inference path that (incorrectly) depends on the trainer import must FAIL
# in the clean environment — proving the trainer is truly gone and the bundle path avoids it.
_REQUIRES_TRAINER = '''\
import sys
import trainer_pkg  # a training-repo import; required here on purpose
from rlx.adapters.policy_custom_torch import load_runtime

print(load_runtime(sys.argv[1]).act(0, mode="deterministic"))
'''


@pytest.mark.acceptance
@pytest.mark.requires_torch
def test_p05_repository_independence(tmp_path: Path) -> None:
    """Inference works with the training repository fully removed.

    We build a checkpoint using a trainer package that is only importable via a
    dedicated repo directory, export a bundle, then delete the trainer repo and run
    inference in a *subprocess* with the trainer off ``PYTHONPATH``. The bundle must
    run; a control that requires the trainer import must fail — so the pass is not
    vacuous (it fails iff the training import were actually needed).
    """
    import os
    import subprocess

    from rlx.adapters.policy_custom_torch import export_from_checkpoint

    # 1. Training repo containing the source model class.
    trainer_repo = tmp_path / "trainer_repo"
    (trainer_repo / "trainer_pkg").mkdir(parents=True)
    (trainer_repo / "trainer_pkg" / "__init__.py").write_text(_TRAINER_PKG, encoding="utf-8")

    ckpt = tmp_path / "checkpoint.pt"
    trainer_env = {**os.environ, "PYTHONPATH": str(trainer_repo)}
    built = subprocess.run(
        [sys.executable, "-c", f"import trainer_pkg; trainer_pkg.save_checkpoint({str(ckpt)!r})"],
        env=trainer_env,
        capture_output=True,
        text=True,
    )
    assert built.returncode == 0, built.stderr
    assert ckpt.exists()

    # 2. Export a portable bundle (declarative architecture; no trainer import needed).
    arch = {"type": "mlp_categorical", "observation_dim": 4, "hidden_dims": [16, 16], "action_n": 3}
    exported = export_from_checkpoint(
        source=ckpt,
        out=tmp_path / "exported.rlx",
        role="agent",
        architecture=arch,
        observation={"type": "Discrete", "n": 4, "dtype": "int64"},
        action={"type": "Discrete", "n": 3, "dtype": "int64", "masks": "none"},
        preprocessing={"id": "normalize_v0", "mean": 0.0, "std": 1.0},
    )

    # 3. Move the bundle to a fresh directory and destroy the training repo + checkpoint.
    clean = tmp_path / "clean_room"
    clean.mkdir()
    bundle = clean / "policy.rlx"
    shutil.move(str(exported), str(bundle))
    shutil.rmtree(trainer_repo)
    ckpt.unlink()

    # Clean environment: trainer removed AND off PYTHONPATH; run from the clean dir.
    clean_env = {**os.environ, "PYTHONPATH": ""}

    # 4. Bundle runs with the trainer absent.
    ok = subprocess.run(
        [sys.executable, "-c", _CLEAN_RUN, str(bundle)],
        cwd=clean,
        env=clean_env,
        capture_output=True,
        text=True,
    )
    assert ok.returncode == 0, ok.stderr
    assert "TRAINER_ABSENT" in ok.stdout, ok.stdout
    assert "ACTION" in ok.stdout, ok.stdout

    # 5. Control: an inference path that requires the trainer import fails (and only then).
    control = subprocess.run(
        [sys.executable, "-c", _REQUIRES_TRAINER, str(bundle)],
        cwd=clean,
        env=clean_env,
        capture_output=True,
        text=True,
    )
    assert control.returncode != 0
    assert "No module named 'trainer_pkg'" in control.stderr, control.stderr
