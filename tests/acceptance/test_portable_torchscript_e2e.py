"""Clean-room end-to-end coverage for portable TorchScript policies."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from arena.adapters.policy_custom_torch import (
    export_module_policy,
    load_runtime,
    verify_bundle_integrity,
    verify_bundle_self,
)
from arena.core.errors import ConformanceError, SchemaError
from arena.core.manifests import dump_yaml, load_manifest

torch = pytest.importorskip("torch")
nn = torch.nn


class HwcCnn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 2, kernel_size=1, bias=False)
        self.head = nn.Linear(12, 3, bias=False)
        with torch.no_grad():
            self.conv.weight.fill_(0.25)
            self.head.weight.copy_(
                torch.tensor(
                    [[0.3] * 12, [-0.1] * 12, [0.2] * 12],
                    dtype=torch.float32,
                )
            )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(observation).flatten(1))


class MaskedRecurrent(nn.Module):
    def forward(
        self, observation: torch.Tensor, hidden: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        next_hidden = hidden + observation.sum(dim=1, keepdim=True)
        logits = torch.cat((next_hidden, -next_hidden, next_hidden * 0), dim=1)
        return logits.masked_fill(~mask, -1e9), next_hidden


class BoundedBox(nn.Module):
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.tanh(observation[:, :2])


def _source_box(case: dict[str, object]) -> list[float]:
    x = torch.as_tensor(case["observation"], dtype=torch.float32).view(1, 2)
    return BoundedBox()(x).detach().numpy().reshape(-1).tolist()


def _source_image(case: dict[str, object]) -> tuple[int, list[float]]:
    raw = np.asarray(case["observation"], dtype=np.float32)
    # Mirror the declared source preprocessing, not the exported runtime.
    chw = np.transpose(raw, (2, 0, 1))
    stacked = np.concatenate((np.zeros_like(chw), (chw - 1.0) / 2.0), axis=0)
    logits = HwcCnn()(torch.as_tensor(stacked).unsqueeze(0)).detach().numpy().reshape(-1)
    return int(np.argmax(logits)), logits.tolist()


@pytest.mark.requires_torch
def test_hwc_cnn_source_capture_clean_room_and_loud_shape_failure(tmp_path: Path) -> None:
    image = np.full((2, 3, 1), 3, dtype=np.float32)
    bundle = export_module_policy(
        out_dir=tmp_path / "cnn.arena",
        name="cnn",
        roles=["agent"],
        module=HwcCnn(),
        observation={"type": "Box", "shape": [2, 3, 1], "layout": "HWC"},
        action={"type": "Discrete", "n": 3},
        preprocessing={
            "pipeline": {
                "version": "arena.preprocess/v1",
                "steps": [
                    {"op": "layout", "from": "HWC", "to": "CHW"},
                    {"op": "running_norm", "mean": 1.0, "std": 2.0},
                    {"op": "frame_stack", "k": 2, "axis": 0, "pad": "zeros"},
                ],
            }
        },
        reference_cases=[{"observation": image.tolist(), "mode": "deterministic"}],
        source_act_fn=_source_image,
    )
    assert verify_bundle_self(bundle)["verify_mode"] == "source-conformance"
    rt = load_runtime(bundle)
    assert rt.act(image) == _source_image({"observation": image.tolist()})[0]
    with pytest.raises(ConformanceError, match="raw observation shape"):
        rt.act(np.zeros((1, 2, 3), dtype=np.float32))

    # No trainer module is made available to the child: TorchScript must stand alone.
    child = (
        "from arena.adapters.policy_custom_torch import load_runtime; "
        "import numpy as np; "
        f"print(load_runtime({str(bundle)!r}).act(np.full((2,3,1), 3, dtype=np.float32)))"
    )
    env = {k: v for k, v in os.environ.items() if "PYTHONPATH" not in k}
    result = subprocess.run(
        [sys.executable, "-c", child],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.strip() == "0"


@pytest.mark.requires_torch
def test_recurrent_masked_actor_carries_and_resets_in_clean_room(tmp_path: Path) -> None:
    cases = [
        {
            "observation": [2.0],
            "action_mask": [True, False, False],
            "mode": "deterministic",
            "hidden_reset": True,
            "expected_action": 0,
        },
        {
            "observation": [-3.0],
            "action_mask": [False, True, False],
            "mode": "deterministic",
            "hidden_reset": False,
            "expected_action": 1,
        },
    ]
    bundle = export_module_policy(
        out_dir=tmp_path / "rnn.arena",
        name="rnn",
        roles=["agent"],
        module=MaskedRecurrent(),
        observation={"type": "Box", "shape": [1]},
        action={"type": "Discrete", "n": 3, "masks": "required"},
        io={"recurrent": True, "hidden_shape": [1, 1], "mask_in_graph": True},
        reference_cases=cases,
    )
    with pytest.raises(ConformanceError, match="insufficient evidence"):
        verify_bundle_self(bundle)
    assert verify_bundle_self(bundle, allow_self_consistency=True)["ok"]
    rt = load_runtime(bundle)
    with pytest.raises(ConformanceError, match="required"):
        rt.act(np.array([2], dtype=np.float32))
    rt.reset("a")
    assert rt.act(np.array([2], dtype=np.float32), action_mask=np.array([1, 0, 0]), agent_id="a") == 0
    assert rt.act(np.array([-3], dtype=np.float32), action_mask=np.array([0, 1, 0]), agent_id="a") == 1
    rt.reset("a")
    assert rt.act(np.array([-3], dtype=np.float32), action_mask=np.array([0, 1, 0]), agent_id="a") == 1
    child = (
        "from arena.adapters.policy_custom_torch import load_runtime; "
        "import numpy as np; "
        f"r=load_runtime({str(bundle)!r}); r.reset('a'); "
        "print(r.act(np.array([2], dtype=np.float32), action_mask=np.array([1,0,0]), agent_id='a'), "
        "r.act(np.array([-3], dtype=np.float32), action_mask=np.array([0,1,0]), agent_id='a'))"
    )
    result = subprocess.run(
        [sys.executable, "-c", child],
        env={k: v for k, v in os.environ.items() if "PYTHONPATH" not in k},
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.strip() == "0 1"


@pytest.mark.requires_torch
def test_box_actor_and_all_payload_tampering_fail_loudly(tmp_path: Path) -> None:
    bundle = export_module_policy(
        out_dir=tmp_path / "box.arena",
        name="box",
        roles=["agent"],
        module=BoundedBox(),
        observation={"type": "Box", "shape": [2]},
        action={"type": "Box", "shape": [2], "low": [-1, -1], "high": [1, 1], "dtype": "float32"},
        reference_cases=[{"observation": [0.5, -0.5]}],
        source_act_fn=_source_box,
    )
    action = load_runtime(bundle).act(np.array([0.5, -0.5], dtype=np.float32))
    assert action.shape == (2,) and np.all(action <= 1) and np.all(action >= -1)
    assert verify_bundle_self(bundle)["verify_mode"] == "source-conformance"
    assert verify_bundle_integrity(bundle)["ok"]
    (bundle / "payloads" / "preprocess.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(ConformanceError, match="integrity"):
        load_runtime(bundle)


@pytest.mark.requires_torch
def test_payload_path_traversal_rejects_before_loading(tmp_path: Path) -> None:
    bundle = export_module_policy(
        out_dir=tmp_path / "path.arena",
        name="path",
        roles=["agent"],
        module=BoundedBox(),
        observation={"type": "Box", "shape": [2]},
        action={"type": "Box", "shape": [2], "low": [-1, -1], "high": [1, 1], "dtype": "float32"},
        reference_cases=[{"observation": [0.5, -0.5], "expected_action": [0.1, -0.1]}],
    )
    manifest = load_manifest(bundle / "policy.yaml")
    manifest["payloads"]["model"]["path"] = "../../outside.pt"
    dump_yaml(manifest, bundle / "policy.yaml")
    with pytest.raises(SchemaError, match="escapes policy bundle"):
        load_runtime(bundle)


def test_incomplete_multidiscrete_rejects_before_publication() -> None:
    with pytest.raises(SchemaError, match="logit_layout"):
        export_module_policy(
            out_dir=Path("/tmp") / "will-not-export.arena",
            name="bad",
            roles=["agent"],
            module=object(),
            observation={"type": "Box", "shape": [1]},
            action={"type": "MultiDiscrete", "nvec": [2, 2], "masks": "none"},
            reference_cases=[{"observation": [0]}],
        )
