from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.acceptance
def test_i03_native_core_does_not_import_external_integrations(tmp_path: Path) -> None:
    script = tmp_path / "offline_core.py"
    script.write_text(
        """
import importlib.abc
import sys

BLOCKED = {"openenv", "pyspiel", "gimitest", "huggingface_hub"}

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in BLOCKED:
            raise ImportError(f"blocked optional integration: {fullname}")
        return None

sys.meta_path.insert(0, Blocker())

import arena
from arena.adapters.task_pettingzoo.adapter import describe_task
from arena.core.manifests import validate_evaluation_manifest
from arena.core.registry import capability_matrix

info = describe_task({
    "adapter": "pettingzoo-parallel",
    "env": "arena/competitive_rps_v0",
    "interaction": "parallel",
})
assert info["adapter"] == "pettingzoo-parallel"
suite = validate_evaluation_manifest({
    "schema": "arena.evaluation/v0alpha1",
    "name": "offline-native",
    "provider": "native",
    "task": {"adapter": "pettingzoo-parallel", "env": "arena/competitive_rps_v0"},
    "assignments": {"player_0": "sha256:" + "0" * 64},
    "seeds": [0],
    "action_mode": "deterministic",
    "metrics": ["mean_return"],
})
assert suite["provider"] == "native"
matrix = capability_matrix()
assert "native" in matrix["eval_provider"]
print(arena.__version__)
""",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "1.0.0rc1"
