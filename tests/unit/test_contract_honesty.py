"""Fail-loud contract honesty regressions (messy-trainer silent-wrong class)."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from arena.adapters.policy_custom_torch import (  # noqa: E402
    PROVENANCE_SOURCE,
    _embed_reference_cases,
    build_module,
    export_from_checkpoint,
    export_policy,
    generate_reference_cases,
    load_runtime,
    verify_bundle_self,
)
from arena.core.contracts import (  # noqa: E402
    validate_architecture_spaces,
    validate_reference_case_action,
)
from arena.core.errors import ConformanceError, SchemaError  # noqa: E402
from arena.core.manifests import load_manifest  # noqa: E402


def _mlp(*, obs_dim: int = 4, action_n: int = 3) -> dict:
    return {
        "type": "mlp_categorical",
        "observation_dim": obs_dim,
        "hidden_dims": [16],
        "action_n": action_n,
    }


def _gru(*, obs_dim: int = 4, action_n: int = 3) -> dict:
    return {
        "type": "gru_categorical",
        "observation_dim": obs_dim,
        "rnn_hidden_size": 8,
        "action_n": action_n,
    }


def _box(dim: int = 4) -> dict:
    return {"type": "Box", "shape": [dim], "dtype": "float32", "low": -10.0, "high": 10.0}


def _disc(n: int = 3, *, masks: str = "none") -> dict:
    return {"type": "Discrete", "n": n, "dtype": "int64", "masks": masks}


# ---------------------------------------------------------------------------
# 1. architecture.action_n vs action.n desync
# ---------------------------------------------------------------------------


@pytest.mark.requires_torch
def test_export_rejects_action_n_desync(tmp_path: Path) -> None:
    arch = _mlp(obs_dim=4, action_n=5)
    torch.manual_seed(0)
    with pytest.raises(SchemaError, match="architecture.action_n"):
        export_policy(
            out_dir=tmp_path / "bad.arena",
            name="bad",
            roles=["agent"],
            observation=_box(4),
            action=_disc(3),
            architecture=arch,
            state_dict=build_module(arch).state_dict(),
            preprocessing={"id": "normalize_v0", "mean": 0.0, "std": 1.0},
        )


@pytest.mark.requires_torch
def test_check_rejects_architecture_action_desync() -> None:
    with pytest.raises(SchemaError, match="architecture.action_n"):
        validate_architecture_spaces(
            observation=_box(4),
            action=_disc(3),
            architecture=_mlp(obs_dim=4, action_n=7),
            adapter="custom-pytorch",
        )


# ---------------------------------------------------------------------------
# 2. MultiDiscrete / Box / Dict action lies
# ---------------------------------------------------------------------------


@pytest.mark.requires_torch
@pytest.mark.parametrize(
    "action",
    [
        {
            "type": "MultiDiscrete",
            "nvec": [2, 2],
            "masks": "none",
            "logit_layout": {"kind": "concatenated"},
            "sampling_order": "sequential",
        },
        {"type": "Box", "shape": [2], "dtype": "float32", "masks": "none", "low": -1, "high": 1},
        {"type": "Dict", "masks": "none"},
    ],
)
def test_export_rejects_non_discrete_actions(tmp_path: Path, action: dict) -> None:
    arch = _mlp()
    torch.manual_seed(1)
    with pytest.raises(SchemaError, match="deliberately rejected|only supports Discrete"):
        export_policy(
            out_dir=tmp_path / "bad.arena",
            name="bad",
            roles=["agent"],
            observation=_box(),
            action=action,
            architecture=arch,
            state_dict=build_module(arch).state_dict(),
            preprocessing={"id": "normalize_v0", "mean": 0.0, "std": 1.0},
        )


# ---------------------------------------------------------------------------
# 3. Silent obs truncate/pad → fail loud
# ---------------------------------------------------------------------------


@pytest.mark.requires_torch
def test_encode_obs_fails_loud_on_length_mismatch(tmp_path: Path) -> None:
    arch = _mlp(obs_dim=4)
    torch.manual_seed(2)
    bundle = export_policy(
        out_dir=tmp_path / "p.arena",
        name="p",
        roles=["agent"],
        observation=_box(4),
        action=_disc(3),
        architecture=arch,
        state_dict=build_module(arch).state_dict(),
        preprocessing={"id": "normalize_v0", "mean": 0.0, "std": 1.0},
    )
    rt = load_runtime(bundle)
    with pytest.raises(ConformanceError, match="Refusing to pad or truncate"):
        rt.act([0.0, 0.0], mode="deterministic")
    with pytest.raises(ConformanceError, match="Refusing to pad or truncate"):
        rt.act([0.0] * 8, mode="deterministic")


@pytest.mark.requires_torch
def test_discrete_obs_rejects_length1_vector(tmp_path: Path) -> None:
    arch = _mlp(obs_dim=4)
    torch.manual_seed(3)
    bundle = export_policy(
        out_dir=tmp_path / "d.arena",
        name="d",
        roles=["agent"],
        observation={"type": "Discrete", "n": 4, "dtype": "int64"},
        action=_disc(3),
        architecture=arch,
        state_dict=build_module(arch).state_dict(),
        preprocessing={"id": "normalize_v0", "mean": 0.0, "std": 1.0},
    )
    rt = load_runtime(bundle)
    assert rt.act(1, mode="deterministic") in {0, 1, 2}
    with pytest.raises(ConformanceError, match="scalar index"):
        rt.act(np.asarray([1]), mode="deterministic")


# ---------------------------------------------------------------------------
# 4. arch observation_dim ≠ observation.shape
# ---------------------------------------------------------------------------


@pytest.mark.requires_torch
def test_export_rejects_obs_dim_shape_desync(tmp_path: Path) -> None:
    arch = _mlp(obs_dim=8, action_n=3)
    torch.manual_seed(4)
    with pytest.raises(SchemaError, match="observation_dim"):
        export_policy(
            out_dir=tmp_path / "bad.arena",
            name="bad",
            roles=["agent"],
            observation=_box(4),
            action=_disc(3),
            architecture=arch,
            state_dict=build_module(arch).state_dict(),
            preprocessing={"id": "normalize_v0", "mean": 0.0, "std": 1.0},
        )


@pytest.mark.requires_torch
def test_export_rejects_3d_box_without_layout(tmp_path: Path) -> None:
    arch = _mlp(obs_dim=12, action_n=3)
    torch.manual_seed(5)
    with pytest.raises(SchemaError, match="layout"):
        export_policy(
            out_dir=tmp_path / "img.arena",
            name="img",
            roles=["agent"],
            observation={
                "type": "Box",
                "shape": [2, 2, 3],
                "dtype": "float32",
                "low": 0.0,
                "high": 1.0,
            },
            action=_disc(3),
            architecture=arch,
            state_dict=build_module(arch).state_dict(),
            preprocessing={"id": "normalize_v0", "mean": 0.0, "std": 1.0},
        )


# ---------------------------------------------------------------------------
# 5. Verify must not accept illegal expected_action
# ---------------------------------------------------------------------------


@pytest.mark.requires_torch
def test_verify_rejects_oob_expected_action(tmp_path: Path) -> None:
    arch = _mlp()
    torch.manual_seed(6)
    bundle = export_policy(
        out_dir=tmp_path / "p.arena",
        name="p",
        roles=["agent"],
        observation=_box(),
        action=_disc(3),
        architecture=arch,
        state_dict=build_module(arch).state_dict(),
        preprocessing={"id": "normalize_v0", "mean": 0.0, "std": 1.0},
    )
    # Bypass embed validation (simulates tampered / hand-edited cases on disk).
    cases_path = bundle / "payloads" / "reference_cases.json"
    cases_path.write_text(
        json.dumps(
            {
                "provenance": PROVENANCE_SOURCE,
                "cases": [
                    {
                        "observation": [0.0, 0.0, 0.0, 0.0],
                        "mode": "deterministic",
                        "expected_action": 99,
                        "hidden_reset": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConformanceError, match="illegal under Discrete"):
        verify_bundle_self(bundle)
    assert load_manifest(bundle / "policy.yaml").get("conformance", {}).get("status") != "verified"


@pytest.mark.requires_torch
def test_embed_rejects_oob_expected_action(tmp_path: Path) -> None:
    arch = _mlp()
    torch.manual_seed(6)
    bundle = export_policy(
        out_dir=tmp_path / "p.arena",
        name="p",
        roles=["agent"],
        observation=_box(),
        action=_disc(3),
        architecture=arch,
        state_dict=build_module(arch).state_dict(),
        preprocessing={"id": "normalize_v0", "mean": 0.0, "std": 1.0},
    )
    with pytest.raises(ConformanceError, match="illegal under Discrete"):
        _embed_reference_cases(
            bundle,
            [
                {
                    "observation": [0.0, 0.0, 0.0, 0.0],
                    "mode": "deterministic",
                    "expected_action": 99,
                    "hidden_reset": True,
                }
            ],
            provenance=PROVENANCE_SOURCE,
        )


@pytest.mark.requires_torch
def test_verify_rejects_masked_illegal_expected_action(tmp_path: Path) -> None:
    with pytest.raises(ConformanceError, match="illegal under the case"):
        validate_reference_case_action(
            {
                "expected_action": 1,
                "action_mask": [True, False, True],
            },
            action=_disc(3, masks="required"),
            index=0,
        )


# ---------------------------------------------------------------------------
# 6. EMA ignored → warn or prefer_ema
# ---------------------------------------------------------------------------


@pytest.mark.requires_torch
def test_ema_state_dict_warns_when_ignored(tmp_path: Path) -> None:
    arch = _mlp()
    torch.manual_seed(7)
    plain = build_module(arch).state_dict()
    torch.manual_seed(8)
    ema = build_module(arch).state_dict()
    ckpt = tmp_path / "both.pt"
    torch.save({"state_dict": plain, "ema_state_dict": ema}, ckpt)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        export_from_checkpoint(
            source=ckpt,
            out=tmp_path / "plain.arena",
            role="agent",
            architecture=arch,
            observation=_box(),
            action=_disc(),
            preprocessing={"id": "normalize_v0", "mean": 0.0, "std": 1.0},
            make_reference_cases=False,
            prefer_ema=False,
        )
    assert any("ema_state_dict" in str(w.message) for w in caught)


@pytest.mark.requires_torch
def test_prefer_ema_loads_ema_weights(tmp_path: Path) -> None:
    arch = _mlp()
    torch.manual_seed(9)
    plain = build_module(arch).state_dict()
    torch.manual_seed(10)
    ema = build_module(arch).state_dict()
    ckpt = tmp_path / "both.pt"
    torch.save({"state_dict": plain, "ema_state_dict": ema}, ckpt)
    bundle = export_from_checkpoint(
        source=ckpt,
        out=tmp_path / "ema.arena",
        role="agent",
        architecture=arch,
        observation=_box(),
        action=_disc(),
        preprocessing={"id": "normalize_v0", "mean": 0.0, "std": 1.0},
        make_reference_cases=False,
        prefer_ema=True,
    )
    loaded = torch.load(bundle / "payloads" / "weights.pt", weights_only=True)
    for k in ema:
        assert torch.allclose(loaded[k], ema[k])
        assert not torch.allclose(loaded[k], plain[k])


@pytest.mark.requires_torch
def test_ema_is_selected_by_default_and_base_opt_out_is_recorded(tmp_path: Path) -> None:
    arch = _mlp()
    torch.manual_seed(12)
    plain = build_module(arch).state_dict()
    torch.manual_seed(13)
    ema = build_module(arch).state_dict()
    ckpt = tmp_path / "both.pt"
    torch.save({"state_dict": plain, "ema_state_dict": ema}, ckpt)
    ema_bundle = export_from_checkpoint(
        source=ckpt, out=tmp_path / "ema.arena", role="agent", architecture=arch,
        observation=_box(), action=_disc(), make_reference_cases=False,
    )
    base_bundle = export_from_checkpoint(
        source=ckpt, out=tmp_path / "base.arena", role="agent", architecture=arch,
        observation=_box(), action=_disc(), make_reference_cases=False, prefer_ema=False,
    )
    assert torch.allclose(
        torch.load(ema_bundle / "payloads" / "weights.pt", weights_only=True)["net.0.weight"],
        ema["net.0.weight"],
    )
    assert torch.allclose(
        torch.load(base_bundle / "payloads" / "weights.pt", weights_only=True)["net.0.weight"],
        plain["net.0.weight"],
    )


# ---------------------------------------------------------------------------
# 7. MPE → mpe2 on PettingZoo ≥1.25
# ---------------------------------------------------------------------------


@pytest.mark.requires_pettingzoo
def test_mpe_simple_tag_resolves_via_mpe2() -> None:
    pytest.importorskip("mpe2")
    from arena.adapters.task_pettingzoo.adapter import make_env

    env = make_env(
        {
            "adapter": "pettingzoo-parallel",
            "env": "mpe/simple_tag_v3",
            "config": {"num_good": 1, "num_adversaries": 1},
        }
    )
    try:
        obs, _ = env.reset(seed=0)
        assert "adversary_0" in env.agents
        assert env.observation_space("adversary_0").shape == (12,)
    finally:
        env.close()


# ---------------------------------------------------------------------------
# 8. reset_on=[] must not coerce to defaults
# ---------------------------------------------------------------------------


@pytest.mark.requires_torch
def test_reset_on_empty_list_preserved(tmp_path: Path) -> None:
    arch = _gru()
    torch.manual_seed(11)
    bundle = export_policy(
        out_dir=tmp_path / "gru.arena",
        name="gru",
        roles=["agent"],
        observation=_box(),
        action=_disc(),
        architecture=arch,
        state_dict=build_module(arch).state_dict(),
        preprocessing={"id": "normalize_v0", "mean": 0.0, "std": 1.0},
        recurrent=True,
        reset_on=[],
    )
    manifest = load_manifest(bundle / "policy.yaml")
    assert manifest["state"]["reset_on"] == []
    rt = load_runtime(bundle)
    rt.reset("a0")
    # agent_termination not declared → reset_agent is a no-op (stale carry kept)
    rt.act([0.1, 0.2, 0.3, 0.4], agent_id="a0")
    after_act = rt._hidden["a0"].clone()
    rt.reset_agent("a0")
    assert torch.equal(rt._hidden["a0"], after_act)

    # Explicit agent_termination still works when declared
    bundle2 = export_policy(
        out_dir=tmp_path / "gru2.arena",
        name="gru2",
        roles=["agent"],
        observation=_box(),
        action=_disc(),
        architecture=arch,
        state_dict=build_module(arch).state_dict(),
        preprocessing={"id": "normalize_v0", "mean": 0.0, "std": 1.0},
        recurrent=True,
        reset_on=["agent_termination"],
    )
    rt2 = load_runtime(bundle2)
    rt2.reset("a0")
    rt2.act([0.1, 0.2, 0.3, 0.4], agent_id="a0")
    after = rt2._hidden["a0"].clone()
    rt2.reset_agent("a0")
    assert torch.equal(rt2._hidden["a0"], rt2.module.initial_state(1))
    assert not torch.equal(rt2._hidden["a0"], after)


# ---------------------------------------------------------------------------
# 9. Optional / required mask verify coverage
# ---------------------------------------------------------------------------


@pytest.mark.requires_torch
@pytest.mark.parametrize("masks", ["optional", "required"])
def test_generate_reference_cases_exercises_masks(tmp_path: Path, masks: str) -> None:
    arch = _mlp()
    torch.manual_seed(12)
    bundle = export_policy(
        out_dir=tmp_path / f"{masks}.arena",
        name=masks,
        roles=["agent"],
        observation=_box(),
        action=_disc(3, masks=masks),
        architecture=arch,
        state_dict=build_module(arch).state_dict(),
        preprocessing={"id": "normalize_v0", "mean": 0.0, "std": 1.0},
    )
    rt = load_runtime(bundle)
    cases = generate_reference_cases(
        rt, observation=_box(), action=_disc(3, masks=masks)
    )
    assert any(c.get("action_mask") is not None for c in cases)
    _embed_reference_cases(bundle, cases, provenance=PROVENANCE_SOURCE)
    assert verify_bundle_self(bundle)["ok"]


@pytest.mark.requires_torch
def test_verify_refuses_optional_masks_without_masked_case(tmp_path: Path) -> None:
    arch = _mlp()
    torch.manual_seed(13)
    bundle = export_policy(
        out_dir=tmp_path / "opt.arena",
        name="opt",
        roles=["agent"],
        observation=_box(),
        action=_disc(3, masks="optional"),
        architecture=arch,
        state_dict=build_module(arch).state_dict(),
        preprocessing={"id": "normalize_v0", "mean": 0.0, "std": 1.0},
    )
    rt = load_runtime(bundle)
    obs = [0.0, 0.0, 0.0, 0.0]
    act = rt.act(obs, mode="deterministic")
    _embed_reference_cases(
        bundle,
        [
            {
                "observation": obs,
                "mode": "deterministic",
                "expected_action": int(act),
                "expected_logits": rt.last_logits.tolist(),
                "hidden_reset": True,
            }
        ],
        provenance=PROVENANCE_SOURCE,
    )
    with pytest.raises(ConformanceError, match="no reference case"):
        verify_bundle_self(bundle)
