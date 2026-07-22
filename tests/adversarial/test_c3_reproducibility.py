"""Claim 3 (adversarial): reproducibility.

Attacks:
  * same seed -> identical trajectories across separate PROCESS invocations
  * different seeds -> divergent trajectories (non-vacuous)
  * recurrent state resets exactly at episode boundaries (detected via the runner)
  * determinism holds under the stochastic RNG contract
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("pettingzoo")

from _adv_envs import BoxObsParallel  # noqa: E402

from rlx.conformance.fixtures import build_f3_recurrent, build_rps_policy  # noqa: E402
from rlx.core.sdk import Match, Policy, Task  # noqa: E402

_PILOT = "rlx/competitive_rps_v0"

# A standalone runner used to prove *cross-process* determinism. It exports nothing
# (bundles are prebuilt and shared); it only executes a match against fixed bundles.
_RUNNER = """
import sys, json
from rlx.core.sdk import Match, Policy, Task

p0, p1, out, mode, seeds = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
seed_list = [int(s) for s in seeds.split(",")]
match = Match(
    task=Task.load({"adapter": "pettingzoo-parallel", "env": "rlx/competitive_rps_v0",
                    "config": {"max_cycles": 8}}),
    assignments={"player_0": Policy.load(p0), "player_1": Policy.load(p1)},
    action_mode=mode,
    failure_policy={"timeout_seconds": 30, "retain_incomplete": True, "retry": 0},
)
match.run(seeds=seed_list, record=True, out=out)
print("DONE")
"""


def _episode_bytes(run_dir: Path, i: int) -> bytes:
    return (run_dir / "trajectories" / f"episode_{i:04d}.json").read_bytes()


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_cross_process_determinism_deterministic_mode(tmp_path: Path) -> None:
    p0 = build_rps_policy(tmp_path / "p0", role="player_0", seed=10)
    p1 = build_rps_policy(tmp_path / "p1", role="player_1", seed=20)
    seeds = "0,1,2,3"

    def run(tag: str) -> Path:
        out = tmp_path / tag
        proc = subprocess.run(
            [sys.executable, "-c", _RUNNER, str(p0), str(p1), str(out), "deterministic", seeds],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert "DONE" in proc.stdout
        return out

    a, b = run("procA"), run("procB")
    for i in range(4):
        assert _episode_bytes(a, i) == _episode_bytes(b, i), f"episode {i} differs across processes"


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_cross_process_determinism_stochastic_and_seed_divergence(tmp_path: Path) -> None:
    """Stochastic policies are byte-identical across processes for the same seeds, and
    different seeds produce genuinely different action streams."""
    p0 = build_rps_policy(tmp_path / "p0", role="player_0", seed=10)
    p1 = build_rps_policy(tmp_path / "p1", role="player_1", seed=20)

    def run(tag: str, seeds: str) -> Path:
        out = tmp_path / tag
        proc = subprocess.run(
            [sys.executable, "-c", _RUNNER, str(p0), str(p1), str(out), "stochastic", seeds],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        return out

    a = run("A", "0,1,2,3")
    b = run("B", "0,1,2,3")
    for i in range(4):
        assert _episode_bytes(a, i) == _episode_bytes(b, i)

    # Non-vacuous: different seeds -> different trajectories.
    actions_by_seed = {}
    for i in range(4):
        ep = json.loads(_episode_bytes(a, i))
        actions_by_seed[i] = json.dumps([s["actions"] for s in ep["steps"]])
    assert len(set(actions_by_seed.values())) >= 2, "distinct seeds gave identical trajectories"


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_recurrent_state_resets_at_episode_boundary(tmp_path: Path, patch_task_env) -> None:
    """Two episodes run with the *same* seed must be identical — which can only happen
    if the recurrent hidden state is reset at the episode boundary. If the runner
    leaked hidden state across episodes, the second same-seed episode would diverge."""
    patch_task_env(BoxObsParallel, max_cycles=6)
    bundle = build_f3_recurrent(tmp_path / "f3")
    match = Match(
        task=Task.load({"adapter": "pettingzoo-parallel", "env": "adv/boxobs"}),
        assignments={"agent": Policy.load(bundle)},
        action_mode="deterministic",
    )
    out = tmp_path / "run"
    match.run(seeds=[7, 7], record=True, out=out)
    ep0 = json.loads(_episode_bytes(out, 0))
    ep1 = json.loads(_episode_bytes(out, 1))
    a0 = [s["actions"]["agent"] for s in ep0["steps"]]
    a1 = [s["actions"]["agent"] for s in ep1["steps"]]
    assert len(a0) == 6
    assert a0 == a1, "same-seed episodes diverged -> recurrent state not reset at boundary"


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_recurrent_state_actually_matters(tmp_path: Path) -> None:
    """Guard against a vacuous reset test: prove the recurrent hidden state changes
    behavior, so 'reset' is meaningful. Without reset, a repeated obs stream started
    from a carried hidden state can differ from a fresh one."""
    import numpy as np

    from rlx.adapters.policy_custom_torch import load_runtime

    rt = load_runtime(build_f3_recurrent(tmp_path / "f3b"))
    stream = np.random.default_rng(1).normal(size=(6, 4)).astype(np.float32)

    # Recurrent logits accumulate hidden state across the stream.
    rt.reset("x")
    recurrent_logits = [rt.logits(o, agent_id="x").tolist() for o in stream]

    # Reset before EVERY step -> memoryless logits. If the hidden state truly matters,
    # the logits must differ somewhere (an untrained GRU can leave argmax constant, so
    # we compare the continuous logits rather than the discrete action).
    memoryless_logits = []
    for o in stream:
        rt.reset("x")
        memoryless_logits.append(rt.logits(o, agent_id="x").tolist())

    # A fresh reset reproduces the recurrent logits exactly (deterministic replay).
    rt.reset("x")
    reproduced = [rt.logits(o, agent_id="x").tolist() for o in stream]
    assert reproduced == recurrent_logits
    assert not np.allclose(
        np.asarray(recurrent_logits), np.asarray(memoryless_logits)
    ), "hidden state has no effect on logits -> reset test would be vacuous"
