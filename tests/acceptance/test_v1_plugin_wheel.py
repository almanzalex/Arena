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
    source = Path("examples/eval/demo/rock.arena").resolve()
    destination = (tmp_path / "example-mirror").resolve().as_uri().replace(
        "file:", "example:", 1
    )
    probe = f"""
from arena.core.registry import EXTERNAL_STORES, ensure_plugins_loaded, UnknownKindError
from arena.conformance.qualification import qualify_store

ensure_plugins_loaded()
assert "example" not in EXTERNAL_STORES.known()
assert EXTERNAL_STORES.get("example").scheme == "example"
assert "example" in EXTERNAL_STORES.known()

try:
    EXTERNAL_STORES.get("s3")
except UnknownKindError as exc:
    assert "arena store qualify" in str(exc)
else:
    raise AssertionError("expected UnknownKindError for unknown store kind")

report = qualify_store({str(source)!r}, destination={destination!r})
assert report["ok"] is True
assert report["backend"] == "example"
assert report["schema"] == "arena.store-qualification/v1"
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
