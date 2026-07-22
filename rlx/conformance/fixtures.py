"""Build reference policy fixtures F1–F4 and pilot RPS policies for F5."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rlx.adapters.policy_custom_torch import (
    TorchPolicyRuntime,
    build_module,
    export_policy,
    load_runtime,
)


def _torch():
    import torch

    return torch


def _make_state_dict(architecture: dict[str, Any], seed: int = 0) -> dict[str, Any]:
    torch = _torch()
    torch.manual_seed(seed)
    module = build_module(architecture)
    return module.state_dict()


def build_f1_deterministic(out_dir: Path | str, *, seed: int = 0) -> Path:
    """F1: Deterministic feed-forward policy (Discrete actions)."""
    arch = {
        "type": "mlp_categorical",
        "observation_dim": 4,
        "hidden_dims": [16, 16],
        "action_n": 3,
    }
    state = _make_state_dict(arch, seed=seed)
    out = export_policy(
        out_dir=out_dir,
        name="f1-deterministic",
        roles=["agent"],
        observation={"type": "Box", "shape": [4], "dtype": "float32", "low": -10.0, "high": 10.0},
        action={"type": "Discrete", "n": 3, "dtype": "int64", "masks": "none"},
        architecture=arch,
        state_dict=state,
        preprocessing={"id": "normalize_v0", "mean": 0.0, "std": 1.0},
        modes=["deterministic", "stochastic"],
    )
    # Embed reference cases from the exported runtime itself (source == export for fixture)
    rt = load_runtime(out)
    cases = []
    rng = np.random.default_rng(123)
    for i in range(8):
        obs = rng.normal(size=4).astype(np.float32)
        rt.reset()
        action = rt.act(obs, mode="deterministic")
        cases.append(
            {
                "observation": obs.tolist(),
                "mode": "deterministic",
                "expected_action": action,
                "expected_logits": rt.last_logits.tolist(),
                "hidden_reset": True,
            }
        )
    _write_cases(out, cases)
    return Path(out)


def build_f2_stochastic(out_dir: Path | str, *, seed: int = 1) -> Path:
    """F2: Stochastic discrete policy with explicit RNG contract."""
    arch = {
        "type": "mlp_categorical",
        "observation_dim": 4,
        "hidden_dims": [16],
        "action_n": 3,
    }
    state = _make_state_dict(arch, seed=seed)
    out = export_policy(
        out_dir=out_dir,
        name="f2-stochastic",
        roles=["agent"],
        observation={"type": "Box", "shape": [4], "dtype": "float32", "low": -10.0, "high": 10.0},
        action={"type": "Discrete", "n": 3, "dtype": "int64", "masks": "none"},
        architecture=arch,
        state_dict=state,
        preprocessing={"id": "normalize_v0", "mean": [0, 0, 0, 0], "std": [1, 1, 1, 1]},
        modes=["deterministic", "stochastic"],
    )
    rt = load_runtime(out)
    cases = []
    for i, case_seed in enumerate([7, 11, 19, 23]):
        obs = np.zeros(4, dtype=np.float32)
        obs[i % 4] = 1.0
        rt.reset()
        rng = np.random.default_rng(case_seed)
        action = rt.act(obs, mode="stochastic", rng=rng)
        cases.append(
            {
                "observation": obs.tolist(),
                "mode": "stochastic",
                "seed": case_seed,
                "expected_action": action,
                "hidden_reset": True,
            }
        )
    _write_cases(out, cases)
    return Path(out)


def build_f3_recurrent(out_dir: Path | str, *, seed: int = 2) -> Path:
    """F3: Recurrent policy with episode-level resets."""
    arch = {
        "type": "gru_categorical",
        "observation_dim": 4,
        "rnn_hidden_size": 8,
        "action_n": 3,
    }
    state = _make_state_dict(arch, seed=seed)
    out = export_policy(
        out_dir=out_dir,
        name="f3-recurrent",
        roles=["agent"],
        observation={"type": "Box", "shape": [4], "dtype": "float32", "low": -10.0, "high": 10.0},
        action={"type": "Discrete", "n": 3, "dtype": "int64", "masks": "none"},
        architecture=arch,
        state_dict=state,
        preprocessing={"id": "normalize_v0", "mean": 0.0, "std": 1.0},
        recurrent=True,
        reset_on=["episode_start", "agent_termination"],
        modes=["deterministic", "stochastic"],
    )
    rt = load_runtime(out)
    cases = []
    stream = np.random.default_rng(0).normal(size=(5, 4)).astype(np.float32)
    rt.reset()
    for t, obs in enumerate(stream):
        action = rt.act(obs, mode="deterministic", agent_id="a0")
        cases.append(
            {
                "observation": obs.tolist(),
                "mode": "deterministic",
                "agent_id": "a0",
                "expected_action": action,
                "expected_logits": rt.last_logits.tolist(),
                "hidden_reset": t == 0,
            }
        )
    # After episode reset, first-step action must match a fresh stream start
    rt.reset("a0")
    again = rt.act(stream[0], mode="deterministic", agent_id="a0")
    cases.append(
        {
            "observation": stream[0].tolist(),
            "mode": "deterministic",
            "agent_id": "a0",
            "expected_action": again,
            "expected_logits": rt.last_logits.tolist(),
            "hidden_reset": True,
            "note": "episode_reset",
        }
    )
    _write_cases(out, cases)
    return Path(out)


def build_f4_masked(out_dir: Path | str, *, seed: int = 3) -> Path:
    """F4: Masked-action policy with changing legal actions."""
    arch = {
        "type": "mlp_categorical",
        "observation_dim": 4,
        "hidden_dims": [16],
        "action_n": 4,
    }
    state = _make_state_dict(arch, seed=seed)
    out = export_policy(
        out_dir=out_dir,
        name="f4-masked",
        roles=["agent"],
        observation={"type": "Box", "shape": [4], "dtype": "float32", "low": -10.0, "high": 10.0},
        action={"type": "Discrete", "n": 4, "dtype": "int64", "masks": "required"},
        architecture=arch,
        state_dict=state,
        preprocessing={"id": "normalize_v0", "mean": 0.0, "std": 1.0},
        modes=["deterministic", "stochastic"],
    )
    rt = load_runtime(out)
    cases = []
    masks = [
        [True, False, False, False],
        [False, True, True, False],
        [True, True, False, True],
        [False, False, False, True],
    ]
    for i, mask in enumerate(masks):
        obs = np.ones(4, dtype=np.float32) * (i + 1)
        rt.reset()
        action = rt.act(obs, mode="deterministic", action_mask=np.asarray(mask))
        assert mask[action], "fixture generator produced illegal action"
        cases.append(
            {
                "observation": obs.tolist(),
                "mode": "deterministic",
                "action_mask": mask,
                "expected_action": action,
                "hidden_reset": True,
            }
        )
    _write_cases(out, cases)
    return Path(out)


def build_rps_policy(out_dir: Path | str, *, role: str, seed: int = 0) -> Path:
    """Pilot policy for PettingZoo RPS (observation Discrete→one-hot dim 4, actions 3)."""
    # PettingZoo RPS observation is Discrete(4): 0,1,2 moves + start token
    arch = {
        "type": "mlp_categorical",
        "observation_dim": 4,
        "hidden_dims": [32, 32],
        "action_n": 3,
    }
    state = _make_state_dict(arch, seed=seed)
    return export_policy(
        out_dir=out_dir,
        name=f"rps-{role}",
        roles=[role],
        observation={"type": "Discrete", "n": 4, "dtype": "int64"},
        action={"type": "Discrete", "n": 3, "dtype": "int64", "masks": "none"},
        architecture=arch,
        state_dict=state,
        preprocessing={"id": "normalize_v0", "mean": 0.0, "std": 1.0},
        modes=["deterministic", "stochastic"],
        lineage={"fixture": "F5", "pilot": "rlx/competitive_rps_v0"},
    )


def build_fixed_action_rps_policy(
    out_dir: Path | str,
    *,
    role: str | list[str] = "player_0",
    action: int,
    name: str | None = None,
) -> Path:
    """Export an RPS policy that always selects ``action`` (0=rock, 1=paper, 2=scissors)."""
    torch = _torch()
    roles = [role] if isinstance(role, str) else list(role)
    arch = {
        "type": "mlp_categorical",
        "observation_dim": 4,
        "hidden_dims": [8],
        "action_n": 3,
    }
    module = build_module(arch)
    with torch.no_grad():
        for p in module.parameters():
            p.zero_()
        last = None
        for layer in module.net:  # type: ignore[attr-defined]
            if isinstance(layer, torch.nn.Linear):
                last = layer
        assert last is not None
        last.bias[int(action)] = 50.0
    label = name or f"rps-fixed-{action}-{'-'.join(roles)}"
    return export_policy(
        out_dir=out_dir,
        name=label,
        roles=roles,
        observation={"type": "Discrete", "n": 4, "dtype": "int64"},
        action={"type": "Discrete", "n": 3, "dtype": "int64", "masks": "none"},
        architecture=arch,
        state_dict=module.state_dict(),
        preprocessing={"id": "normalize_v0", "mean": 0.0, "std": 1.0},
        modes=["deterministic", "stochastic"],
        lineage={
            "fixture": "fixed-action",
            "action": int(action),
            "pilot": "rlx/competitive_rps_v0",
        },
    )


def source_runtime_from_bundle(bundle: Path | str) -> TorchPolicyRuntime:
    """For P-01..P-05 the 'source' is an in-process runtime loaded before export isolation."""
    return load_runtime(bundle)


def _write_cases(bundle: Path | str, cases: list[dict[str, Any]]) -> None:
    from rlx.adapters.policy_custom_torch import _embed_reference_cases

    # Fixture cases are captured from the exported runtime itself → self-consistency.
    _embed_reference_cases(bundle, cases, provenance="self-consistency")
