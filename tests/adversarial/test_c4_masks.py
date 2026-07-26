"""Claim 4 (adversarial): action masks / legality.

Attacks:
  * masked policy never emits an illegal action, even under adversarial masks
    (single legal action, alternating masks, all-but-one illegal) in BOTH modes
  * all-illegal mask errors clearly
  * missing-but-required mask errors clearly
  * mask length mismatch errors clearly
  * an unmasked (masks='none') policy given a mask still never picks an illegal action
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("pettingzoo")

from _adv_envs import make_discrete_policy  # noqa: E402

from arena.adapters.policy_custom_torch import load_runtime  # noqa: E402
from arena.conformance.fixtures import build_f4_masked  # noqa: E402
from arena.core.errors import ConformanceError  # noqa: E402


def _masked_runtime(tmp_path: Path, action_n: int = 4):
    return load_runtime(
        make_discrete_policy(tmp_path / "m", role="player_0", action_n=action_n, masks="required")
    )


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
@pytest.mark.parametrize("mode", ["deterministic", "stochastic"])
def test_single_legal_action_always_chosen(tmp_path: Path, mode: str) -> None:
    rt = _masked_runtime(tmp_path)
    for legal in range(4):
        mask = np.zeros(4, dtype=bool)
        mask[legal] = True
        for i in range(50):
            rt.reset()
            # Discrete observation contract: scalar index (not a float vector).
            a = rt.act(
                int(i % 4),
                mode=mode,
                action_mask=mask,
                rng=np.random.default_rng(i),
            )
            assert a == legal, f"mode={mode} legal={legal} got {a}"


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
@pytest.mark.parametrize("mode", ["deterministic", "stochastic"])
def test_alternating_and_adversarial_masks_never_illegal(tmp_path: Path, mode: str) -> None:
    rt = _masked_runtime(tmp_path)
    rng = np.random.default_rng(0)
    for i in range(400):
        mask = rng.random(4) > 0.5
        if not mask.any():
            mask[i % 4] = True
        rt.reset()
        a = rt.act(
            int(i % 4),
            mode=mode,
            action_mask=mask,
            rng=np.random.default_rng(i),
        )
        assert bool(mask[a]), f"illegal action {a} under mask {mask.tolist()} (mode={mode})"


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_all_illegal_mask_errors(tmp_path: Path) -> None:
    rt = _masked_runtime(tmp_path)
    rt.reset()
    with pytest.raises(ConformanceError, match="no legal actions"):
        rt.act(0, mode="deterministic", action_mask=np.zeros(4, dtype=bool))


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_missing_required_mask_errors(tmp_path: Path) -> None:
    rt = _masked_runtime(tmp_path)
    rt.reset()
    with pytest.raises(ConformanceError, match="action mask required"):
        rt.act(0, mode="deterministic")


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_mask_length_mismatch_errors(tmp_path: Path) -> None:
    rt = _masked_runtime(tmp_path, action_n=4)
    rt.reset()
    with pytest.raises(ConformanceError, match="length mismatch"):
        rt.act(
            0,
            mode="deterministic",
            action_mask=np.array([True, False, True]),  # wrong length (3 vs 4)
        )


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_unmasked_policy_respects_supplied_mask(tmp_path: Path) -> None:
    """Even a masks='none' policy, if handed a mask, must not select an illegal action
    (defensive legality: the mask is authoritative when present)."""
    rt = load_runtime(make_discrete_policy(tmp_path / "n", role="player_0", action_n=4, masks="none"))
    for i in range(100):
        mask = np.zeros(4, dtype=bool)
        mask[i % 4] = True
        rt.reset()
        a = rt.act(
            int(i % 4),
            mode="deterministic",
            action_mask=mask,
        )
        assert a == i % 4


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_f4_fixture_masks_all_legal(tmp_path: Path) -> None:
    """The shipped F4 masked fixture: every embedded reference case chose a legal action
    and self-verify passes (masks reproduce across a re-load)."""
    import json

    from arena.adapters.policy_custom_torch import verify_bundle_self

    bundle = build_f4_masked(tmp_path / "f4")
    cases = json.loads((bundle / "payloads" / "reference_cases.json").read_text())["cases"]
    for c in cases:
        if c.get("action_mask") is not None:
            assert c["action_mask"][c["expected_action"]]
    assert verify_bundle_self(bundle, allow_self_consistency=True)["ok"]
