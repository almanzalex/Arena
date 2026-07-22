"""Q-02: adapter qualify accepts evaluation fixtures (population + eval)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("pettingzoo")

from _eval_fixtures import build_cyclic_rps_eval_fixture

from rlx.conformance.qualification import qualify_adapter_fixture


@pytest.mark.acceptance
@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_qualifies_evaluation_population_fixture(tmp_path: Path) -> None:
    fx = build_cyclic_rps_eval_fixture(tmp_path / "fx")
    out = tmp_path / "qualification.json"
    report = qualify_adapter_fixture(fx["evaluation"], report_path=out)
    assert report["ok"]
    assert report["kind"] == "evaluation"
    assert report["checks"]["eval_reproducibility"]["ok"]
    assert report["checks"]["eval_report_evidence"]["ok"]
    assert report["checks"]["eval_report_evidence"]["nontransitivity_warning"]
    assert report["checks"]["eval_report_evidence"]["ranking"] is None
    persisted = json.loads(out.read_text(encoding="utf-8"))
    assert persisted["schema"] == "rlx.adapter-qualification/v1"
