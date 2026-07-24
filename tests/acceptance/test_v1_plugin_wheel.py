from __future__ import annotations

import subprocess
import sys
import venv
from pathlib import Path

import pytest


@pytest.mark.acceptance
def test_out_of_tree_v1_plugin_installs_loads_lazily_and_uninstalls(
    tmp_path: Path,
) -> None:
    environment = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(environment)
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    plugin = Path("examples/plugins/rlx-example-store").resolve()
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            str(plugin),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    probe = """
from rlx.core.registry import EXTERNAL_STORES, ensure_plugins_loaded
ensure_plugins_loaded()
assert "example" not in EXTERNAL_STORES.known()
assert EXTERNAL_STORES.get("example").scheme == "example"
assert "example" in EXTERNAL_STORES.known()
"""
    subprocess.run(
        [str(python), "-c", probe],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "uninstall",
            "-y",
            "rlx-example-store",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            str(python),
            "-c",
            "from rlx.core.support import doctor_report; assert doctor_report('core')['ok']",
        ],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )
