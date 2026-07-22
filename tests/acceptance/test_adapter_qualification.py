"""Qualification report is executable release evidence, not prose."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("pettingzoo")

from rlx.adapters.policy_custom_torch import (
    _embed_reference_cases,
    generate_reference_cases,
    load_runtime,
)
from rlx.conformance.fixtures import build_rps_policy
from rlx.conformance.qualification import qualify_adapter_fixture


@pytest.mark.acceptance
@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_qualifies_custom_torch_pettingzoo_fixture(tmp_path: Path) -> None:
    bundles = {}
    for index, role in enumerate(("player_0", "player_1")):
        bundle = build_rps_policy(tmp_path / f"{role}.rlx", role=role, seed=index + 1)
        runtime = load_runtime(bundle)
        _embed_reference_cases(
            bundle,
            generate_reference_cases(
                runtime,
                observation=runtime.manifest["observation"],
                action=runtime.manifest["action"],
            ),
            provenance="source-conformance",
        )
        bundles[role] = bundle
    fixture = tmp_path / "match.yaml"
    fixture.write_text(
        """schema: rlx.match/v0alpha1
task: {adapter: pettingzoo-parallel, env: rlx/competitive_rps_v0}
assignments:
  player_0: ./player_0.rlx
  player_1: ./player_1.rlx
seeds: {start: 0, count: 2}
action_mode: deterministic
record: {trajectories: all}
""",
        encoding="utf-8",
    )
    out = tmp_path / "qualification.json"
    report = qualify_adapter_fixture(fixture, report_path=out)
    assert report["ok"]
    assert report["checks"]["tamper_detection"]["ok"]
    persisted = json.loads(out.read_text(encoding="utf-8"))
    assert persisted["schema"] == "rlx.adapter-qualification/v1"
