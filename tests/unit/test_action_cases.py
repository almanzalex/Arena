"""Adversarial / messy regressions for typed action-case support.

Re-creates prior silent-wrong scenarios: incomplete MultiDiscrete/Box/Dict claims,
action.n vs head-width desync, illegal expected_action, missing RNG contracts,
unstable Dict keys, match validation skips, trajectory round-trip, tamper digests,
and wrong transform/seed negative paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from rlx.adapters.policy_custom_torch import (  # noqa: E402
    export_module_policy,
    load_runtime,
    verify_bundle_integrity,
    verify_bundle_self,
)
from rlx.core.action_cases import (  # noqa: E402
    decode_action_from_params,
    validate_action_case,
    validate_expected_action,
    validate_runtime_action,
)
from rlx.core.errors import ConformanceError, RuntimeFailure, SchemaError  # noqa: E402
from rlx.core.manifests import load_manifest  # noqa: E402
from rlx.runtime.match import _validate_action  # noqa: E402
from rlx.runtime.trajectory import TrajectoryWriter, inspect_trajectory  # noqa: E402


def _obs() -> dict:
    return {"type": "Box", "shape": [2], "dtype": "float32", "low": -5.0, "high": 5.0}


def _complete_multidiscrete() -> dict:
    return {
        "type": "MultiDiscrete",
        "nvec": [2, 3],
        "masks": "none",
        "logit_layout": {"kind": "concatenated"},
        "sampling_order": "sequential",
        "dtype": "int64",
    }


def _complete_gaussian() -> dict:
    return {
        "type": "Box",
        "shape": [2],
        "low": [-1.0, -1.0],
        "high": [1.0, 1.0],
        "dtype": "float32",
        "masks": "none",
        "distribution": "diagonal_gaussian",
        "param_layout": {"kind": "mean_log_std_concat"},
        "transform": {"order": ["sample", "tanh", "affine"]},
        "rng": {"algorithm": "numpy_generator"},
        "deterministic_mode": "mean",
    }


def _complete_dict() -> dict:
    return {
        "type": "Dict",
        "masks": "none",
        "key_order": ["move", "aim"],
        "spaces": {
            "move": {"type": "Discrete", "n": 2, "masks": "none"},
            "aim": {
                "type": "Box",
                "shape": [1],
                "low": [-1.0],
                "high": [1.0],
                "dtype": "float32",
                "masks": "none",
            },
        },
        "param_layout": {
            "kind": "concatenated_fields",
            "fields": {
                "move": {"kind": "logits", "slice": [0, 2]},
                "aim": {"kind": "box_values", "slice": [2, 3]},
            },
        },
    }


class MultiDiscreteActor(nn.Module):
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        # Concatenated logits for nvec=[2,3]
        a = observation[:, :1].repeat(1, 2)
        b = observation[:, :1].repeat(1, 3) * 0.5
        return torch.cat((a, b), dim=1)


class GaussianActor(nn.Module):
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        mean = torch.tanh(observation[:, :2]) * 0.25
        log_std = observation[:, :2] * 0.0 - 1.0
        return torch.cat((mean, log_std), dim=1)


class DictActor(nn.Module):
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        logits = torch.stack((observation[:, 0], -observation[:, 0]), dim=1)
        aim = torch.tanh(observation[:, 1:2])
        return torch.cat((logits, aim), dim=1)


def _case_rng(case: dict) -> np.random.Generator | None:
    if case.get("rng") is not None:
        return case["rng"]
    if "seed" in case:
        return np.random.default_rng(int(case["seed"]))
    return None


def _source_md(case: dict) -> list[int]:
    x = torch.as_tensor(case["observation"], dtype=torch.float32).view(1, -1)
    logits = MultiDiscreteActor()(x).detach().numpy().reshape(-1)
    mode = case.get("mode", "deterministic")
    act = decode_action_from_params(
        logits, action=_complete_multidiscrete(), mode=mode, rng=_case_rng(case)
    )
    return act.tolist()


def _source_gauss(case: dict) -> list[float]:
    x = torch.as_tensor(case["observation"], dtype=torch.float32).view(1, -1)
    params = GaussianActor()(x).detach().numpy().reshape(-1)
    mode = case.get("mode", "deterministic")
    act = decode_action_from_params(
        params, action=_complete_gaussian(), mode=mode, rng=_case_rng(case)
    )
    return act.astype(np.float32).tolist()


def _source_dict(case: dict) -> dict:
    x = torch.as_tensor(case["observation"], dtype=torch.float32).view(1, -1)
    params = DictActor()(x).detach().numpy().reshape(-1)
    act = decode_action_from_params(
        params,
        action=_complete_dict(),
        mode=case.get("mode", "deterministic"),
        rng=_case_rng(case),
    )
    return {"move": int(act["move"]), "aim": np.asarray(act["aim"]).tolist()}


# ---------------------------------------------------------------------------
# 1. Incomplete claims fail loud (no silent scalar)
# ---------------------------------------------------------------------------


@pytest.mark.requires_torch
@pytest.mark.parametrize(
    ("action", "message"),
    [
        ({"type": "MultiDiscrete", "nvec": [2, 3], "masks": "none"}, "logit_layout"),
        ({"type": "Dict", "masks": "none", "spaces": {"a": {"type": "Discrete", "n": 2}}}, "key_order"),
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
            "param_layout",
        ),
        ({"type": "Dict", "masks": "none"}, "typed spaces|untyped"),
    ],
)
def test_incomplete_action_cases_publish_no_bundle(
    tmp_path: Path, action: dict, message: str
) -> None:
    out = tmp_path / "must-not-exist.rlx"
    with pytest.raises(SchemaError, match=message):
        export_module_policy(
            out_dir=out,
            name="bad",
            roles=["agent"],
            module=MultiDiscreteActor(),
            observation=_obs(),
            action=action,
            reference_cases=[{"observation": [0.0, 0.0]}],
        )
    assert not out.exists()


# ---------------------------------------------------------------------------
# 2. Complete MultiDiscrete / Dict / Gaussian work end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.requires_torch
def test_complete_multidiscrete_export_verify_act(tmp_path: Path) -> None:
    bundle = export_module_policy(
        out_dir=tmp_path / "md.rlx",
        name="md",
        roles=["agent"],
        module=MultiDiscreteActor(),
        observation=_obs(),
        action=_complete_multidiscrete(),
        reference_cases=[
            {"observation": [1.0, 0.0], "mode": "deterministic"},
            {"observation": [0.5, -0.5], "mode": "stochastic", "seed": 11},
        ],
        source_act_fn=_source_md,
    )
    assert verify_bundle_self(bundle)["ok"]
    rt = load_runtime(bundle)
    act = rt.act(np.array([1.0, 0.0], dtype=np.float32), mode="deterministic")
    assert act.shape == (2,) and act.dtype == np.int64
    assert act.tolist() == _source_md({"observation": [1.0, 0.0]})
    # Never silent scalar
    with pytest.raises(ConformanceError, match="int vector|Refusing scalar"):
        validate_runtime_action(0, action=_complete_multidiscrete())


@pytest.mark.requires_torch
def test_complete_gaussian_seeded_conformance_and_negatives(tmp_path: Path) -> None:
    bundle = export_module_policy(
        out_dir=tmp_path / "g.rlx",
        name="gauss",
        roles=["agent"],
        module=GaussianActor(),
        observation=_obs(),
        action=_complete_gaussian(),
        reference_cases=[
            {"observation": [0.2, -0.2], "mode": "deterministic"},
            {"observation": [0.2, -0.2], "mode": "stochastic", "seed": 9},
        ],
        source_act_fn=_source_gauss,
    )
    assert verify_bundle_self(bundle)["ok"]
    rt = load_runtime(bundle)
    det = rt.act(np.array([0.2, -0.2], dtype=np.float32), mode="deterministic")
    assert det.shape == (2,)
    a = rt.act(
        np.array([0.2, -0.2], dtype=np.float32),
        mode="stochastic",
        rng=np.random.default_rng(9),
    )
    b = rt.act(
        np.array([0.2, -0.2], dtype=np.float32),
        mode="stochastic",
        rng=np.random.default_rng(9),
    )
    c = rt.act(
        np.array([0.2, -0.2], dtype=np.float32),
        mode="stochastic",
        rng=np.random.default_rng(10),
    )
    assert np.allclose(a, b)
    assert not np.allclose(a, c)

    # Wrong seed stream → source conformance mismatch
    cases = [
        {
            "observation": [0.2, -0.2],
            "mode": "stochastic",
            "seed": 9,
            "expected_action": _source_gauss(
                {
                    "observation": [0.2, -0.2],
                    "mode": "stochastic",
                    "rng": np.random.default_rng(99),
                }
            ),
        }
    ]
    path = bundle / "payloads" / "reference_cases.json"
    path.write_text(
        json.dumps({"provenance": "source-conformance", "cases": cases}),
        encoding="utf-8",
    )
    # Digest will fail integrity first — rewrite digest via verify path after
    # integrity bypass: load_runtime checks digests. Patch digest for this negative.
    from rlx.core.identity import digest_uri, sha256_file
    from rlx.core.manifests import dump_yaml

    manifest = load_manifest(bundle / "policy.yaml")
    manifest["payloads"]["reference_cases"]["digest"] = digest_uri(sha256_file(path))
    dump_yaml(manifest, bundle / "policy.yaml")
    with pytest.raises(ConformanceError, match="self-verify failed|verify failed"):
        verify_bundle_self(bundle)


@pytest.mark.requires_torch
def test_wrong_transform_order_rejected_at_export(tmp_path: Path) -> None:
    bad = _complete_gaussian()
    bad["transform"] = {"order": ["sample", "affine", "tanh"]}
    with pytest.raises(SchemaError, match="transform.order"):
        export_module_policy(
            out_dir=tmp_path / "bad-order.rlx",
            name="bad",
            roles=["agent"],
            module=GaussianActor(),
            observation=_obs(),
            action=bad,
            reference_cases=[{"observation": [0.0, 0.0]}],
        )


@pytest.mark.requires_torch
def test_complete_dict_export_and_key_mismatch(tmp_path: Path) -> None:
    bundle = export_module_policy(
        out_dir=tmp_path / "d.rlx",
        name="dict",
        roles=["agent"],
        module=DictActor(),
        observation=_obs(),
        action=_complete_dict(),
        reference_cases=[{"observation": [1.0, -0.5], "mode": "deterministic"}],
        source_act_fn=_source_dict,
    )
    assert verify_bundle_self(bundle)["ok"]
    rt = load_runtime(bundle)
    act = rt.act(np.array([1.0, -0.5], dtype=np.float32))
    assert set(act) == {"move", "aim"}
    assert isinstance(act["move"], (int, np.integer))
    with pytest.raises(ConformanceError, match="key mismatch|Dict↔vector"):
        validate_runtime_action({"aim": [0.1]}, action=_complete_dict())
    with pytest.raises(SchemaError, match="key_order"):
        validate_action_case(
            {
                "type": "Dict",
                "masks": "none",
                "spaces": _complete_dict()["spaces"],
                "param_layout": _complete_dict()["param_layout"],
            },
            require_byo_layout=True,
        )


# ---------------------------------------------------------------------------
# 3. Illegal expected_action cannot stamp verified
# ---------------------------------------------------------------------------


@pytest.mark.requires_torch
def test_illegal_expected_actions_refuse_verify(tmp_path: Path) -> None:
    with pytest.raises(ConformanceError, match="illegal under MultiDiscrete"):
        validate_expected_action(
            {"expected_action": [0, 99]}, action=_complete_multidiscrete()
        )
    with pytest.raises(ConformanceError, match="violates Box bounds"):
        validate_expected_action(
            {"expected_action": [2.0, 0.0]}, action=_complete_gaussian()
        )
    with pytest.raises(ConformanceError, match="key mismatch"):
        validate_expected_action(
            {"expected_action": {"move": 0}}, action=_complete_dict()
        )

    bundle = export_module_policy(
        out_dir=tmp_path / "md.rlx",
        name="md",
        roles=["agent"],
        module=MultiDiscreteActor(),
        observation=_obs(),
        action=_complete_multidiscrete(),
        reference_cases=[{"observation": [0.0, 0.0], "mode": "deterministic"}],
        source_act_fn=_source_md,
    )
    cases_path = bundle / "payloads" / "reference_cases.json"
    cases_path.write_text(
        json.dumps(
            {
                "provenance": "source-conformance",
                "cases": [
                    {
                        "observation": [0.0, 0.0],
                        "mode": "deterministic",
                        "expected_action": [0, 99],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    from rlx.core.identity import digest_uri, sha256_file
    from rlx.core.manifests import dump_yaml

    manifest = load_manifest(bundle / "policy.yaml")
    manifest["payloads"]["reference_cases"]["digest"] = digest_uri(sha256_file(cases_path))
    dump_yaml(manifest, bundle / "policy.yaml")
    with pytest.raises(ConformanceError, match="illegal under MultiDiscrete"):
        verify_bundle_self(bundle)
    assert load_manifest(bundle / "policy.yaml").get("conformance", {}).get("status") != "verified"


# ---------------------------------------------------------------------------
# 4. Match _validate_action covers MultiDiscrete / Box / Dict
# ---------------------------------------------------------------------------


def test_match_validate_action_all_cases() -> None:
    task_md = {
        "roles": {"agent": {"action": _complete_multidiscrete()}},
    }
    _validate_action(
        np.array([1, 2], dtype=np.int64),
        agent="agent",
        task_info=task_md,
        episode_index=0,
        step_i=0,
    )
    with pytest.raises(RuntimeFailure, match="factor|out of range|int vector"):
        _validate_action(
            1,  # silent scalar must fail
            agent="agent",
            task_info=task_md,
            episode_index=0,
            step_i=0,
        )

    task_box = {
        "roles": {
            "agent": {
                "action": {
                    "type": "Box",
                    "shape": [2],
                    "low": [-1, -1],
                    "high": [1, 1],
                    "masks": "none",
                }
            }
        }
    }
    with pytest.raises(RuntimeFailure, match="outside"):
        _validate_action(
            np.array([2.0, 0.0], dtype=np.float32),
            agent="agent",
            task_info=task_box,
            episode_index=0,
            step_i=0,
        )

    task_dict = {"roles": {"agent": {"action": _complete_dict()}}}
    _validate_action(
        {"move": 1, "aim": np.array([0.1], dtype=np.float32)},
        agent="agent",
        task_info=task_dict,
        episode_index=0,
        step_i=0,
    )
    with pytest.raises(RuntimeFailure, match="key mismatch|Dict"):
        _validate_action(
            [1, 0.1],
            agent="agent",
            task_info=task_dict,
            episode_index=0,
            step_i=0,
        )


# ---------------------------------------------------------------------------
# 5. Trajectory completeness for non-Discrete actions
# ---------------------------------------------------------------------------


def test_trajectory_roundtrip_non_discrete(tmp_path: Path) -> None:
    writer = TrajectoryWriter(tmp_path / "traj")
    episode = {
        "schema": "rlx.trajectory/v0alpha1",
        "episode_index": 0,
        "seed": 1,
        "status": "completed",
        "action_mode": "deterministic",
        "task": {"env": "t", "adapter": "x", "version": "0"},
        "agents": ["agent"],
        "role_map": {"agent": "agent"},
        "policies": {"agent": "sha256:abc"},
        "steps": [
            {
                "t": 0,
                "observations": {"agent": [0.0, 0.0]},
                "actions": {
                    "agent": {
                        "move": 1,
                        "aim": [0.25],
                        "factors": [0, 2],
                        "box": [0.1, -0.2],
                    }
                },
                "rewards": {"agent": 0.0},
                "terminations": {"agent": False},
                "truncations": {"agent": True},
            }
        ],
    }
    writer.write_episode(episode)
    writer.finalize(
        task_info={"env": "t", "adapter": "x", "version": "0"},
        assignments={"agent": "sha256:abc"},
        seeds=[1],
        action_mode="deterministic",
        failures=[],
    )
    meta = inspect_trajectory(tmp_path / "traj")
    assert meta["completeness"]["ok"] is True
    loaded = json.loads((tmp_path / "traj" / "episode_0000.json").read_text())
    assert loaded["steps"][0]["actions"]["agent"]["factors"] == [0, 2]
    assert loaded["steps"][0]["actions"]["agent"]["box"] == [0.1, -0.2]


# ---------------------------------------------------------------------------
# 6. Tamper / digest still holds for new action bundles
# ---------------------------------------------------------------------------


@pytest.mark.requires_torch
def test_tamper_still_detected_for_multidiscrete_bundle(tmp_path: Path) -> None:
    bundle = export_module_policy(
        out_dir=tmp_path / "md.rlx",
        name="md",
        roles=["agent"],
        module=MultiDiscreteActor(),
        observation=_obs(),
        action=_complete_multidiscrete(),
        reference_cases=[{"observation": [0.0, 0.0]}],
        source_act_fn=_source_md,
    )
    assert verify_bundle_integrity(bundle)["ok"]
    model = bundle / "payloads" / "model.pt"
    model.write_bytes(model.read_bytes() + b"x")
    with pytest.raises(ConformanceError, match="integrity"):
        load_runtime(bundle)


# ---------------------------------------------------------------------------
# 7. Deterministic Box still refuses undeclared stochastic mode
# ---------------------------------------------------------------------------


@pytest.mark.requires_torch
def test_deterministic_box_refuses_stochastic_without_gaussian_case(tmp_path: Path) -> None:
    class Bounded(nn.Module):
        def forward(self, observation: torch.Tensor) -> torch.Tensor:
            return torch.tanh(observation[:, :2])

    bundle = export_module_policy(
        out_dir=tmp_path / "box.rlx",
        name="box",
        roles=["agent"],
        module=Bounded(),
        observation=_obs(),
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
