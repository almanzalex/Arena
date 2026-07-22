"""Executable boundary decisions for action spaces and arbitrary modules."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from rlx.adapters.policy_custom_torch import (
    export_module_policy,
    load_runtime,
    verify_bundle_self,
)
from rlx.core.errors import SchemaError

torch = pytest.importorskip("torch")
nn = torch.nn


class ScriptableDynamicActor(nn.Module):
    """Data-dependent control flow which TorchScript preserves, unlike trace."""

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if bool((observation.sum() > 0).item()):
            return torch.stack(
                (observation[:, 0], -observation[:, 0], observation[:, 0] * 0), dim=1
            )
        return torch.stack(
            (observation[:, 0] * 0, -observation[:, 0], observation[:, 0]), dim=1
        )


class PythonListActor(nn.Module):
    """Intentionally depends on Python scalar extraction and is not safely portable."""

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.tensor([float(x) for x in observation.flatten().tolist()])


class BoundedBoxActor(nn.Module):
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.tanh(observation[:, :2])


def _box_observation() -> dict[str, object]:
    return {"type": "Box", "shape": [2], "dtype": "float32", "low": -5.0, "high": 5.0}


@pytest.mark.requires_torch
def test_scriptable_dynamic_control_flow_is_clean_room_portable(tmp_path: Path) -> None:
    bundle = export_module_policy(
        out_dir=tmp_path / "dynamic.rlx",
        name="dynamic",
        roles=["agent"],
        module=ScriptableDynamicActor(),
        observation=_box_observation(),
        action={"type": "Discrete", "n": 3},
        reference_cases=[
            {"observation": [2.0, 0.0], "mode": "deterministic"},
            {"observation": [-2.0, 0.0], "mode": "deterministic"},
        ],
        source_act_fn=lambda case: (
            0 if float(case["observation"][0]) > 0 else 1
        ),
    )
    runtime = load_runtime(bundle)
    assert runtime.act(np.array([2.0, 0.0], dtype=np.float32)) == 0
    assert runtime.act(np.array([-2.0, 0.0], dtype=np.float32)) == 1
    assert verify_bundle_self(bundle)["verify_mode"] == "source-conformance"

    child = (
        "from rlx.adapters.policy_custom_torch import load_runtime; import numpy as np; "
        f"r=load_runtime({str(bundle)!r}); "
        "print(r.act(np.array([2.,0.],dtype=np.float32)),"
        "r.act(np.array([-2.,0.],dtype=np.float32)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", child],
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.strip() == "0 1"


@pytest.mark.requires_torch
@pytest.mark.parametrize(
    ("action", "message"),
    [
        ({"type": "MultiDiscrete", "nvec": [2, 3], "masks": "none"}, "logit_layout"),
        (
            {"type": "Dict", "masks": "none", "spaces": {"move": {"type": "Discrete", "n": 2, "masks": "none"}}},
            "key_order",
        ),
        (
            {
                "type": "Box",
                "shape": [2],
                "low": [-1, -1],
                "high": [1, 1],
                "dtype": "float32",
                "masks": "none",
                "distribution": "diagonal_gaussian",
            },
            "param_layout|diagonal-Gaussian|incomplete",
        ),
    ],
)
def test_rejected_action_spaces_publish_no_partial_bundle(
    tmp_path: Path, action: dict[str, object], message: str
) -> None:
    out = tmp_path / "must-not-exist.rlx"
    with pytest.raises(SchemaError, match=message):
        export_module_policy(
            out_dir=out,
            name="bad",
            roles=["agent"],
            module=BoundedBoxActor(),
            observation=_box_observation(),
            action=action,
            reference_cases=[{"observation": [0.0, 0.0]}],
        )
    assert not out.exists()


@pytest.mark.requires_torch
def test_stochastic_box_runtime_request_fails_without_sampling(tmp_path: Path) -> None:
    bundle = export_module_policy(
        out_dir=tmp_path / "box.rlx",
        name="box",
        roles=["agent"],
        module=BoundedBoxActor(),
        observation=_box_observation(),
        action={
            "type": "Box",
            "shape": [2],
            "low": [-1, -1],
            "high": [1, 1],
            "dtype": "float32",
            "masks": "none",
        },
        reference_cases=[{"observation": [0.2, -0.2]}],
    )
    with pytest.raises(SchemaError, match="diagonal_gaussian|distribution=deterministic"):
        load_runtime(bundle).act(
            np.array([0.2, -0.2], dtype=np.float32),
            mode="stochastic",
            rng=np.random.default_rng(9),
        )


@pytest.mark.requires_torch
def test_unscriptable_python_actor_fails_before_publication(tmp_path: Path) -> None:
    out = tmp_path / "must-not-exist.rlx"
    with pytest.raises(SchemaError, match="torch.export and bundled Python source"):
        export_module_policy(
            out_dir=out,
            name="python-list",
            roles=["agent"],
            module=PythonListActor(),
            observation=_box_observation(),
            action={"type": "Discrete", "n": 2},
            reference_cases=[{"observation": [1.0, 2.0]}],
        )
    assert not out.exists()
