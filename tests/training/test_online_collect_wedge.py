"""CPU spike: bounded online collect → dataset bind → offline BC train."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("pettingzoo")

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from arena.adapters.policy_custom_torch import load_runtime, verify_bundle_self
from arena.core.manifests import load_manifest
from arena.core.sdk import Policy
from arena.dataset import PROVENANCE_BINDING_SCHEMA, verify_dataset_provenance

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_spike():
    path = REPO_ROOT / "examples" / "training" / "online_collect_loop.py"
    spec = importlib.util.spec_from_file_location("online_collect_loop", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["online_collect_loop"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_online_collect_bind_offline_train_wedge(tmp_path: Path) -> None:
    mod = _load_spike()
    out = tmp_path / "online-wedge"
    result = mod.run_online_collect_loop(
        out,
        rounds=2,
        episodes_per_round=3,
        epochs=6,
        seed=23,
    )

    assert result["ok"] is True
    assert result["rfc"].endswith("011-online-collection-dataset-binding.md")
    assert "Ray" in result["not_in_scope"]
    assert len(result["rounds"]) == 2

    digests = [row["trained_policy_digest"] for row in result["rounds"]]
    assert digests[-1] == result["final_policy_digest"]
    # Each round trains from that round's collector policy binding.
    assert result["rounds"][0]["collector_policy_digest"] != digests[0]
    assert result["rounds"][1]["collector_policy_digest"] == digests[0]

    for row in result["rounds"]:
        assert row["examples"] > 0
        assert row["loss_final"] <= row["loss_initial"] + 1e-6
        assert row["verification"] == "source-conformance"
        assert row["task"]["env"] == "arena/competitive_rps_v0"
        assert row["task"]["adapter"] == "pettingzoo-parallel"

    # Portable dataset after materialize was re-bound (RFC 011 gap).
    portable = load_manifest(out / "round-00" / "portable-dataset" / "dataset.yaml")
    assert portable["provenance"]["schema"] == PROVENANCE_BINDING_SCHEMA
    verify_dataset_provenance(
        portable,
        expect_policy=result["rounds"][0]["collector_policy_digest"],
        expect_task=result["rounds"][0]["task"],
        dataset_path=out / "round-00" / "portable-dataset" / "dataset.yaml",
    )

    bundle = Path(result["final_policy_path"])
    assert bundle.is_dir()
    policy = Policy.load(bundle)
    assert policy.digest == result["final_policy_digest"]
    assert verify_bundle_self(bundle)["verify_mode"] == "source-conformance"
    runtime = load_runtime(bundle)
    for obs in range(4):
        assert runtime.act(obs) in {0, 1, 2}

    saved = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert saved["final_policy_digest"] == result["final_policy_digest"]
    assert (out / "round-01" / "collect-run" / "trajectories").is_dir()
