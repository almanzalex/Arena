"""Regression tests for messy-trainer Arena 0.1 conformance defects."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from arena.adapters.policy_custom_torch import (  # noqa: E402
    PROVENANCE_SELF,
    PROVENANCE_SOURCE,
    build_module,
    export_from_checkpoint,
    export_policy,
    generate_reference_cases,
    load_runtime,
    verify_bundle_self,
)
from arena.cli.main import main  # noqa: E402
from arena.core.errors import ConformanceError, SchemaError  # noqa: E402
from arena.core.sdk import Policy, Task, check  # noqa: E402


class _BenignNonTensorMarker:
    """Module-level stand-in so torch can pickle a non-weights_only checkpoint."""


def _gru_arch() -> dict:
    return {
        "type": "gru_categorical",
        "observation_dim": 4,
        "rnn_hidden_size": 8,
        "action_n": 3,
    }


def _mlp_arch() -> dict:
    return {
        "type": "mlp_categorical",
        "observation_dim": 4,
        "hidden_dims": [16],
        "action_n": 3,
    }


def _box_obs() -> dict:
    return {"type": "Box", "shape": [4], "dtype": "float32", "low": -10.0, "high": 10.0}


def _disc_act() -> dict:
    return {"type": "Discrete", "n": 3, "dtype": "int64", "masks": "none"}


@pytest.mark.requires_torch
def test_recurrent_verify_same_forward_pass_and_catches_wrong(tmp_path: Path) -> None:
    """Defect 1: verify must use logits from the same act() forward (no GRU double-step)."""
    arch = _gru_arch()
    torch.manual_seed(0)
    state = build_module(arch).state_dict()
    bundle = export_policy(
        out_dir=tmp_path / "gru.arena",
        name="gru",
        roles=["agent"],
        observation=_box_obs(),
        action=_disc_act(),
        architecture=arch,
        state_dict=state,
        preprocessing={"id": "normalize_v0", "mean": 0.0, "std": 1.0},
        recurrent=True,
    )
    rt = load_runtime(bundle)
    cases = generate_reference_cases(rt, observation=_box_obs(), action=_disc_act())
    assert any(c.get("hidden_reset") is False for c in cases)
    assert any(c.get("expected_logits") for c in cases)

    from arena.adapters.policy_custom_torch import _embed_reference_cases

    _embed_reference_cases(bundle, cases, provenance=PROVENANCE_SOURCE)
    assert verify_bundle_self(bundle)["ok"]
    assert verify_bundle_self(bundle)["verify_mode"] == PROVENANCE_SOURCE

    # Genuinely wrong expected action on a carry step must still fail.
    bad = json.loads((bundle / "payloads" / "reference_cases.json").read_text())
    carry = next(c for c in bad["cases"] if c.get("note") == "recurrent_carry")
    carry["expected_action"] = (int(carry["expected_action"]) + 1) % 3
    (bundle / "payloads" / "reference_cases.json").write_text(json.dumps(bad), encoding="utf-8")
    # Reference evidence is itself authenticated at load time.
    with pytest.raises(ConformanceError, match="integrity check failed"):
        verify_bundle_self(bundle)


@pytest.mark.requires_torch
def test_reference_cases_exercise_recurrence_and_catch_double_step(tmp_path: Path) -> None:
    """Defect 2: auto cases carry hidden; a double-step logits bug would be caught."""
    arch = _gru_arch()
    torch.manual_seed(1)
    bundle = export_policy(
        out_dir=tmp_path / "gru2.arena",
        name="gru2",
        roles=["agent"],
        observation=_box_obs(),
        action=_disc_act(),
        architecture=arch,
        state_dict=build_module(arch).state_dict(),
        preprocessing={"id": "normalize_v0", "mean": 0.0, "std": 1.0},
        recurrent=True,
    )
    rt = load_runtime(bundle)
    cases = generate_reference_cases(rt, observation=_box_obs(), action=_disc_act())
    notes = {c.get("note") for c in cases}
    assert "recurrent_carry" in notes
    assert "recurrent_reset_boundary" in notes
    assert any(c.get("hidden_reset") is False for c in cases)

    # Simulate the old bug: expected_logits captured by act() then logits() without
    # restoring hidden (double-step). Same-forward verify must reject those cases.
    rt2 = load_runtime(bundle)
    poisoned = []
    for case in cases:
        agent = case.get("agent_id", "default")
        if case.get("hidden_reset", True):
            rt2.reset(agent)
        if case.get("mode") != "deterministic" or "expected_logits" not in case:
            poisoned.append(case)
            continue
        mask = (
            np.asarray(case["action_mask"]) if case.get("action_mask") is not None else None
        )
        rt2.act(case["observation"], mode="deterministic", action_mask=mask, agent_id=agent)
        # BUG: re-step without reset for carry cases
        bogus = rt2.logits(
            case["observation"], action_mask=mask, agent_id=agent, update_hidden=False
        )
        # Restore hidden consistency for the next case by not leaving a second step:
        # we already advanced twice; rebuild stream state by replaying is hard — instead
        # only poison carry cases and stop after first poisoned carry.
        bad = dict(case)
        bad["expected_logits"] = bogus.tolist()
        poisoned.append(bad)
        if case.get("hidden_reset") is False:
            break
    else:
        pytest.fail("no carry case to poison")

    # Finish with only the prefix we poisoned + ensure at least one carry mismatch.
    from arena.adapters.policy_custom_torch import _embed_reference_cases

    _embed_reference_cases(bundle, poisoned, provenance=PROVENANCE_SOURCE)
    with pytest.raises(ConformanceError, match="logits_mismatch|self-verify failed"):
        verify_bundle_self(bundle)


@pytest.mark.requires_torch
def test_export_embeds_source_conformance_and_self_warns(tmp_path: Path) -> None:
    """Defect 3: export embeds source-conformance; unlabeled cases warn as self-only."""
    arch = _mlp_arch()
    torch.manual_seed(2)
    ckpt = tmp_path / "w.pt"
    torch.save(build_module(arch).state_dict(), ckpt)
    bundle = export_from_checkpoint(
        source=ckpt,
        out=tmp_path / "exp.arena",
        role="agent",
        architecture=arch,
        observation=_box_obs(),
        action=_disc_act(),
        preprocessing={"id": "normalize_v0", "mean": 0.0, "std": 1.0},
    )
    payload = json.loads((bundle / "payloads" / "reference_cases.json").read_text())
    assert payload["provenance"] == PROVENANCE_SOURCE
    result = verify_bundle_self(bundle)
    assert result["ok"]
    assert result["verify_mode"] == PROVENANCE_SOURCE
    assert "warning" not in result

    # Relabel as self-consistency → verify warns.
    payload["provenance"] = PROVENANCE_SELF
    (bundle / "payloads" / "reference_cases.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    # Fix digest after rewrite so integrity of cases entry matches if checked later.
    from arena.adapters.policy_custom_torch import _embed_reference_cases

    _embed_reference_cases(bundle, payload["cases"], provenance=PROVENANCE_SELF)
    with pytest.raises(ConformanceError, match="insufficient evidence"):
        verify_bundle_self(bundle)
    self_result = verify_bundle_self(bundle, allow_self_consistency=True)
    assert self_result["ok"]
    assert self_result["verify_mode"] == PROVENANCE_SELF
    assert "self-consistency" in self_result["warning"]


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_check_honors_task_config_simple_tag(tmp_path: Path) -> None:
    """Defect 4: check must use task config (not env defaults) for spaces."""
    # MPE may live in pettingzoo.mpe (older PZ) or mpe2 (PZ ≥1.25/1.26).
    try:
        import importlib

        importlib.import_module("pettingzoo.mpe.simple_tag_v3")
    except ImportError:
        pytest.importorskip("mpe2.simple_tag_v3")

    # Configured simple_tag: agent_0 obs dim 10; default is 14.
    cfg = {"num_good": 1, "num_adversaries": 1}
    task_cfg = Task.load(
        {
            "adapter": "pettingzoo-parallel",
            "env": "mpe/simple_tag_v3",
            "config": cfg,
        }
    )
    spaces = task_cfg.role_spaces()
    agent_obs = spaces["roles"]["agent_0"]["observation"]
    assert agent_obs["shape"] == [10]

    arch = {
        "type": "mlp_categorical",
        "observation_dim": 10,
        "hidden_dims": [8],
        "action_n": 5,
    }
    torch.manual_seed(3)
    bundle = export_policy(
        out_dir=tmp_path / "tag.arena",
        name="tag-agent",
        roles=["agent_0"],
        observation={
            "type": "Box",
            "shape": [10],
            "dtype": "float32",
            "low": -np.inf,
            "high": np.inf,
        },
        action={"type": "Discrete", "n": 5, "dtype": "int64", "masks": "none"},
        architecture=arch,
        state_dict=build_module(arch).state_dict(),
        preprocessing={"id": "normalize_v0", "mean": 0.0, "std": 1.0},
    )
    policy = Policy.load(bundle)

    # Without config (defaults): observation mismatch.
    task_default = Task.load(
        {"adapter": "pettingzoo-parallel", "env": "mpe/simple_tag_v3"}
    )
    bad = check(task_default, policy.as_role("agent_0"))
    assert not bad.ok
    assert any(i.code == "OBSERVATION_MISMATCH" for i in bad.issues)

    # With configured task: compatible.
    good = check(task_cfg, policy.as_role("agent_0"))
    assert good.ok, good.format_human()

    # CLI --config path (and task YAML) also work.
    task_yaml = tmp_path / "task.yaml"
    task_yaml.write_text(
        "adapter: pettingzoo-parallel\n"
        "env: mpe/simple_tag_v3\n"
        "config:\n"
        "  num_good: 1\n"
        "  num_adversaries: 1\n",
        encoding="utf-8",
    )
    assert main(["check", str(task_yaml), str(bundle), "--role", "agent_0"]) == 0
    assert (
        main(
            [
                "check",
                "mpe/simple_tag_v3",
                str(bundle),
                "--role",
                "agent_0",
                "--config",
                json.dumps(cfg),
            ]
        )
        == 0
    )


@pytest.mark.requires_torch
def test_nested_training_checkpoint_fails_atomically(tmp_path: Path) -> None:
    """Defect 5: nested training blobs fail cleanly with no partial bundle."""
    ckpt = tmp_path / "train.pt"
    # All-string-key training dict that is NOT a state_dict (no tensor values).
    torch.save(
        {
            "epoch": 12,
            "optimizer": {"lr": 1e-3, "betas": [0.9, 0.999]},
            "config": {"algo": "mappo"},
            "loss": 0.42,
            "hparams": {"gamma": 0.99},
        },
        ckpt,
    )
    out = tmp_path / "should_not_exist.arena"
    with pytest.raises(SchemaError, match="training checkpoint"):
        export_from_checkpoint(
            source=ckpt,
            out=out,
            role="agent",
            architecture=_mlp_arch(),
            observation=_box_obs(),
            action=_disc_act(),
            preprocessing={"id": "normalize_v0", "mean": 0.0, "std": 1.0},
        )
    assert not out.exists()
    leftovers = [
        p
        for p in tmp_path.iterdir()
        if p.name.startswith(".arena-export-") or p.name.endswith(".arena")
    ]
    assert leftovers == [], f"partial export left behind: {leftovers}"


@pytest.mark.requires_torch
def test_full_module_pickle_rejected_without_unsafe_opt_in(tmp_path: Path) -> None:
    """Defect 6: fail closed on non-weights_only checkpoints; no partial bundle."""
    arch = _mlp_arch()
    ckpt = tmp_path / "full_pickle.pt"
    torch.save({"note": "not-a-state-dict", "marker": _BenignNonTensorMarker()}, ckpt)
    out = tmp_path / "nope.arena"
    with pytest.raises(SchemaError, match="unsafe pickle|weights_only|refusing"):
        export_from_checkpoint(
            source=ckpt,
            out=out,
            role="agent",
            architecture=arch,
            observation=_box_obs(),
            action=_disc_act(),
            preprocessing={"id": "normalize_v0", "mean": 0.0, "std": 1.0},
            allow_unsafe_checkpoint=False,
        )
    assert not out.exists()
