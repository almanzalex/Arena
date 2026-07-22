"""Qualification covers a registry BYO case (MultiDiscrete), not only Discrete templates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from rlx.adapters.policy_custom_torch import (  # noqa: E402
    export_module_policy,
    load_runtime,
    verify_bundle_self,
)
from rlx.core.action_cases import decode_action_from_params  # noqa: E402


class MultiDiscreteActor(nn.Module):
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        a = observation[:, :1].repeat(1, 2)
        b = observation[:, :1].repeat(1, 3) * 0.5
        return torch.cat((a, b), dim=1)


def _action() -> dict:
    return {
        "type": "MultiDiscrete",
        "nvec": [2, 3],
        "masks": "none",
        "logit_layout": {"kind": "concatenated"},
        "sampling_order": "sequential",
        "dtype": "int64",
    }


def _obs() -> dict:
    return {"type": "Box", "shape": [2], "dtype": "float32", "low": -5.0, "high": 5.0}


def _source(case: dict):
    x = torch.as_tensor(case["observation"], dtype=torch.float32).view(1, -1)
    logits = MultiDiscreteActor()(x).detach().numpy().reshape(-1)
    mode = case.get("mode", "deterministic")
    rng = None
    if "seed" in case:
        rng = np.random.default_rng(int(case["seed"]))
    act = decode_action_from_params(logits, action=_action(), mode=mode, rng=rng)
    return act.tolist(), logits.tolist()


@pytest.mark.acceptance
@pytest.mark.requires_torch
def test_qualify_exercises_multidiscrete_registry_case(tmp_path: Path) -> None:
    """Source-conformance verify must stay green for a registered MultiDiscrete case."""
    bundle = export_module_policy(
        out_dir=tmp_path / "md.rlx",
        name="md",
        roles=["agent"],
        module=MultiDiscreteActor(),
        observation=_obs(),
        action=_action(),
        reference_cases=[
            {"observation": [1.0, 0.0], "mode": "deterministic"},
            {"observation": [0.5, -0.5], "mode": "stochastic", "seed": 3},
        ],
        source_act_fn=_source,
    )
    result = verify_bundle_self(bundle)
    assert result["ok"]
    assert result["verify_mode"] == "source-conformance"
    rt = load_runtime(bundle)
    assert rt.manifest["action"]["type"] == "MultiDiscrete"
    from rlx.core.registry import ACTION_CASES, ensure_plugins_loaded

    ensure_plugins_loaded()
    assert "MultiDiscrete" in ACTION_CASES
