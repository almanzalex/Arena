from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.acceptance
@pytest.mark.requires_torch
def test_cleanrl_bridge_exports_source_conformance_without_receiver_import(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    from arena.adapters.policy_custom_torch import (
        export_module_from_checkpoint,
        load_runtime,
        verify_bundle_self,
    )
    from arena.core.sdk import Policy
    from examples.integrations.cleanrl_bridge import CleanRLQNetwork

    torch.manual_seed(7)
    checkpoint = tmp_path / "dqn.cleanrl_model"
    torch.save(CleanRLQNetwork().state_dict(), checkpoint)
    bundle = export_module_from_checkpoint(
        module_ref="examples.integrations.cleanrl_bridge:build_actor",
        out_dir=tmp_path / "cleanrl-cartpole.arena",
        role="agent",
        observation={
            "type": "Box",
            "shape": [4],
            "dtype": "float32",
            "low": [-4.8, -3.4028235e38, -0.41887903, -3.4028235e38],
            "high": [4.8, 3.4028235e38, 0.41887903, 3.4028235e38],
        },
        action={"type": "Discrete", "n": 2, "dtype": "int64", "masks": "none"},
        source=checkpoint,
        reference_cases=[
            {"observation": [0.0, 0.0, 0.0, 0.0], "mode": "deterministic"},
            {"observation": [0.05, 0.2, -0.03, -0.4], "mode": "deterministic"},
        ],
        source_revision=(
            "cleanrl@fe8d8a03c41a7ef5b523e2e354bd01c363e786bb"
        ),
    )
    verification = verify_bundle_self(bundle)
    assert verification["verify_mode"] == "source-conformance"
    policy = Policy.load(bundle)
    runtime = load_runtime(bundle)
    action = runtime.act([0.05, 0.2, -0.03, -0.4], mode="deterministic")
    assert action in {0, 1}
    assert policy.manifest["lineage"]["source_revision"].startswith("cleanrl@")
