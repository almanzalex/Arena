from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.acceptance
def test_out_of_tree_v1_plugin_installs_loads_lazily_and_removes_cleanly(
    tmp_path: Path,
) -> None:
    plugin = Path("examples/plugins/arena-example-store").resolve()
    wheelhouse = tmp_path / "wheelhouse"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(wheelhouse),
            str(plugin),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = list(wheelhouse.glob("arena_example_store-*.whl"))
    assert len(wheels) == 1
    plugin_site = tmp_path / "plugin-site"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(plugin_site),
            str(wheels[0]),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    plugin_env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            value
            for value in (str(plugin_site), os.environ.get("PYTHONPATH", ""))
            if value
        ),
    }
    probe = """
from arena.core.registry import EXTERNAL_STORES, ensure_plugins_loaded
ensure_plugins_loaded()
assert "example" not in EXTERNAL_STORES.known()
assert EXTERNAL_STORES.get("example").scheme == "example"
assert "example" in EXTERNAL_STORES.known()
"""
    subprocess.run(
        [sys.executable, "-c", probe],
        cwd=Path.cwd(),
        env=plugin_env,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
from importlib.metadata import entry_points
assert not any(
    ep.name == "external_store:example"
    for ep in entry_points(group="arena.plugins.v1")
)
from arena.core.support import doctor_report
assert doctor_report("core")["ok"]
""",
        ],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )
