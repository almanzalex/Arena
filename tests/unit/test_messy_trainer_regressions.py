"""Regression tests for messy-trainer evaluation defects in Arena 0.1."""

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
    pack_reference_cases,
    verify_bundle_self,
)
from arena.core.errors import ConformanceError, SchemaError  # noqa: E402


def _gru_arch(*, obs_dim: int = 4, action_n: int = 3, hidden: int = 8) -> dict:
    return {
        "type": "gru_categorical",
        "observation_dim": obs_dim,
        "rnn_hidden_size": hidden,
        "action_n": action_n,
    }


def _mlp_arch(*, obs_dim: int = 4, action_n: int = 3) -> dict:
    return {
        "type": "mlp_categorical",
        "observation_dim": obs_dim,
        "hidden_dims": [16],
        "action_n": action_n,
    }


def _box_obs(dim: int = 4) -> dict:
    return {"type": "Box", "shape": [dim], "dtype": "float32", "low": -10.0, "high": 10.0}


def _disc_act(n: int = 3) -> dict:
    return {"type": "Discrete", "n": n, "dtype": "int64", "masks": "none"}


# ---------------------------------------------------------------------------
# 1. Recurrent verify must not double-step the GRU
# ---------------------------------------------------------------------------


@pytest.mark.requires_torch
def test_recurrent_bundle_verifies_clean_with_carried_logits(tmp_path: Path) -> None:
    """A correct recurrent bundle with carried hidden + logits verifies cleanly."""
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
    assert any(c.get("note") == "recurrent_reset_boundary" for c in cases)
    assert all("expected_logits" in c for c in cases if c.get("mode") == "deterministic")

    from arena.adapters.policy_custom_torch import _embed_reference_cases

    _embed_reference_cases(bundle, cases, provenance=PROVENANCE_SOURCE)
    result = verify_bundle_self(bundle, allow_self_consistency=True)
    assert result["ok"]
    assert result["verify_mode"] == PROVENANCE_SOURCE


@pytest.mark.requires_torch
def test_recurrent_verify_detects_genuinely_wrong_logits(tmp_path: Path) -> None:
    """Tampered expected_logits on a carry step must still fail verify."""
    arch = _gru_arch()
    torch.manual_seed(1)
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
    # Corrupt a carried step's logits (not the reset step).
    carry = next(c for c in cases if c.get("hidden_reset") is False and "expected_logits" in c)
    carry["expected_logits"] = [float(x) + 0.6 for x in carry["expected_logits"]]

    from arena.adapters.policy_custom_torch import _embed_reference_cases

    _embed_reference_cases(bundle, cases, provenance=PROVENANCE_SOURCE)
    with pytest.raises(ConformanceError, match="logits_mismatch"):
        verify_bundle_self(bundle)


# ---------------------------------------------------------------------------
# 2. Auto reference cases must exercise recurrence (catch carry bugs)
# ---------------------------------------------------------------------------


@pytest.mark.requires_torch
def test_generate_reference_cases_carries_hidden_and_catches_reset_bug(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Carried reference cases must fail a runtime that incorrectly resets every step."""
    from arena.adapters.policy_custom_torch import TorchPolicyRuntime, _embed_reference_cases

    arch = _gru_arch()
    torch.manual_seed(2)
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
    good = load_runtime(bundle)
    cases = generate_reference_cases(good, observation=_box_obs(), action=_disc_act())
    assert sum(1 for c in cases if c.get("hidden_reset") is False) >= 2
    assert any(c.get("note") == "recurrent_reset_boundary" for c in cases)

    # Prove carry changes logits vs per-step reset (more sensitive than argmax).
    stream = [c for c in cases if c.get("note") in {"recurrent_start", "recurrent_carry"}]
    probe = load_runtime(bundle)
    probe.reset("default")
    carried_logits = []
    for c in stream:
        probe.act(c["observation"], mode="deterministic", agent_id="default")
        carried_logits.append(probe.last_logits.copy())
    reset_logits = []
    for c in stream:
        probe.reset("default")
        probe.act(c["observation"], mode="deterministic", agent_id="default")
        reset_logits.append(probe.last_logits.copy())
    diffs = [
        float(np.max(np.abs(a - b)))
        for a, b in zip(carried_logits[1:], reset_logits[1:], strict=True)
    ]
    assert max(diffs) > 1e-4, "GRU carry must change logits vs per-step reset"

    _embed_reference_cases(bundle, cases, provenance=PROVENANCE_SOURCE)
    assert verify_bundle_self(bundle)["ok"]

    # Inject an always-reset bug: a generator that never carried would not catch this.
    real_act = TorchPolicyRuntime.act

    def always_reset_act(self, observation, **kwargs):  # noqa: ANN001
        self.reset(kwargs.get("agent_id", "default"))
        return real_act(self, observation, **kwargs)

    monkeypatch.setattr(TorchPolicyRuntime, "act", always_reset_act)
    with pytest.raises(ConformanceError):
        verify_bundle_self(bundle)


# ---------------------------------------------------------------------------
# 3. Verify semantics: label self-consistency vs source-conformance + warn
# ---------------------------------------------------------------------------


@pytest.mark.requires_torch
def test_export_embeds_source_conformance_cases_and_verify_labels(tmp_path: Path) -> None:
    arch = _mlp_arch()
    torch.manual_seed(3)
    ckpt = tmp_path / "w.pt"
    torch.save({"state_dict": build_module(arch).state_dict()}, ckpt)
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


@pytest.mark.requires_torch
def test_self_consistency_verify_emits_warning(tmp_path: Path) -> None:
    arch = _mlp_arch()
    torch.manual_seed(4)
    bundle = export_policy(
        out_dir=tmp_path / "p.arena",
        name="p",
        roles=["agent"],
        observation=_box_obs(),
        action=_disc_act(),
        architecture=arch,
        state_dict=build_module(arch).state_dict(),
        preprocessing={"id": "normalize_v0", "mean": 0.0, "std": 1.0},
        reference_cases=pack_reference_cases(
            [
                {
                    "observation": [0.0, 0.0, 0.0, 0.0],
                    "mode": "deterministic",
                    "expected_action": 0,
                    "hidden_reset": True,
                }
            ],
            provenance=PROVENANCE_SELF,
        ),
    )
    # Fix expected_action to whatever the runtime actually does.
    rt = load_runtime(bundle)
    rt.reset()
    action = rt.act([0.0, 0.0, 0.0, 0.0], mode="deterministic")
    from arena.adapters.policy_custom_torch import _embed_reference_cases

    _embed_reference_cases(
        bundle,
        [
            {
                "observation": [0.0, 0.0, 0.0, 0.0],
                "mode": "deterministic",
                "expected_action": int(action),
                "expected_logits": rt.last_logits.tolist(),
                "hidden_reset": True,
            }
        ],
        provenance=PROVENANCE_SELF,
    )
    with pytest.raises(ConformanceError, match="insufficient evidence"):
        verify_bundle_self(bundle)
    result = verify_bundle_self(bundle, allow_self_consistency=True)
    assert result["ok"]
    assert result["verify_mode"] == PROVENANCE_SELF
    assert "self-consistency" in result["warning"].lower()


# ---------------------------------------------------------------------------
# 4. arena check must honor task config (simple_tag non-default spaces)
# ---------------------------------------------------------------------------


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_check_uses_task_config_not_env_defaults(tmp_path: Path) -> None:
    pytest.importorskip("pettingzoo")
    try:
        import importlib

        importlib.import_module("pettingzoo.mpe.simple_tag_v3")
    except ImportError:
        pytest.importorskip("mpe2.simple_tag_v3")
    from arena.cli.main import main
    from arena.core.sdk import Policy, Task, check

    # Configured simple_tag: adversary obs dim 12 (default is 16).
    task_yaml = tmp_path / "task.yaml"
    task_yaml.write_text(
        "adapter: pettingzoo-parallel\n"
        "env: mpe/simple_tag_v3\n"
        "config:\n"
        "  num_good: 1\n"
        "  num_adversaries: 1\n",
        encoding="utf-8",
    )
    task = Task.load(task_yaml)
    spaces = task.role_spaces()
    adv = spaces["roles"]["adversary_0"]["observation"]
    assert adv["shape"] == [12] or list(adv.get("shape", [])) == [12]

    arch = _mlp_arch(obs_dim=12, action_n=5)
    torch.manual_seed(5)
    bundle = export_policy(
        out_dir=tmp_path / "tag.arena",
        name="tag-adv",
        roles=["adversary_0"],
        observation={
            "type": "Box",
            "shape": [12],
            "dtype": "float32",
            "low": -np.inf,
            "high": np.inf,
        },
        action={"type": "Discrete", "n": 5, "dtype": "int64", "masks": "none"},
        architecture=arch,
        state_dict=build_module(arch).state_dict(),
        preprocessing={"id": "normalize_v0", "mean": 0.0, "std": 1.0},
    )
    report = check(task, Policy.load(bundle).as_role("adversary_0"))
    assert report.ok, report.format_human()

    # Without config, default spaces (obs dim 16) must false-flag the same policy.
    default_task = Task.load(
        {"adapter": "pettingzoo-parallel", "env": "mpe/simple_tag_v3"}
    )
    bad = check(default_task, Policy.load(bundle).as_role("adversary_0"))
    assert not bad.ok
    assert any(i.code == "OBSERVATION_MISMATCH" for i in bad.issues)

    # CLI: --config merges the same way.
    assert (
        main(
            [
                "check",
                "mpe/simple_tag_v3",
                str(bundle),
                "--role",
                "adversary_0",
                "--config",
                '{"num_good": 1, "num_adversaries": 1}',
            ]
        )
        == 0
    )


# ---------------------------------------------------------------------------
# 5. Atomic export + actionable training-checkpoint error
# ---------------------------------------------------------------------------


@pytest.mark.requires_torch
def test_nested_training_checkpoint_fails_clean_no_partial_bundle(tmp_path: Path) -> None:
    """A training blob with string keys (no tensor state_dict) must fail with an
    actionable message and leave no partial bundle on disk."""
    ckpt = tmp_path / "train.pt"
    # All top-level keys are strings — previously misclassified as a raw state_dict.
    torch.save(
        {
            "epoch": 10,
            "optimizer": {"lr": 0.001, "betas": [0.9, 0.999]},
            "loss": "cross_entropy",
            "notes": "no weights here",
            "scheduler": {"type": "cosine"},
        },
        ckpt,
    )
    out = tmp_path / "should_not_exist.arena"
    with pytest.raises(SchemaError, match="training checkpoint") as exc:
        export_from_checkpoint(
            source=ckpt,
            out=out,
            role="agent",
            architecture=_mlp_arch(),
            observation=_box_obs(),
            action=_disc_act(),
        )
    msg = str(exc.value).lower()
    assert "state_dict" in msg or "extract" in msg
    assert not out.exists(), "partial bundle was left on disk after failed export"
    # No leftover staging dirs either.
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".arena-export-")]
    assert leftovers == [], leftovers


# ---------------------------------------------------------------------------
# 6. Unsafe-pickle export surface: fail closed before executing pickles
# ---------------------------------------------------------------------------


@pytest.mark.requires_torch
def test_full_module_pickle_rejected_without_partial_bundle(tmp_path: Path) -> None:
    """A full-module pickle must be rejected under weights_only safety with no
    partial bundle. We use a benign nn.Linear (no exploit payload)."""
    module = torch.nn.Linear(4, 3)
    ckpt = tmp_path / "full_module.pt"
    torch.save(module, ckpt)  # full module pickle, not a state_dict
    out = tmp_path / "bundle.arena"
    with pytest.raises(SchemaError, match="unsafe pickle|weights_only|refusing"):
        export_from_checkpoint(
            source=ckpt,
            out=out,
            role="agent",
            architecture=_mlp_arch(),
            observation=_box_obs(),
            action=_disc_act(),
            allow_unsafe_checkpoint=False,
        )
    assert not out.exists()
