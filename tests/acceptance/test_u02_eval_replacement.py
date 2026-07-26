"""U-02: hand-written cross-play script retired in favor of population+eval."""

from __future__ import annotations

import runpy
from pathlib import Path


def test_u02_crossplay_script_retired() -> None:
    script = Path(__file__).resolve().parents[2] / "examples" / "eval" / "crossplay_script.py"
    assert script.exists()
    # Running the retired script must fail loud and point at the replacement.
    ns = runpy.run_path(str(script), run_name="not_main")
    assert callable(ns.get("main"))
    code = ns["main"]()
    assert code == 2


def test_u02_docs_point_to_replacement() -> None:
    root = Path(__file__).resolve().parents[2]
    readme = (root / "examples" / "eval" / "README.md").read_text(encoding="utf-8")
    assert "arena eval run" in readme
    assert "retired" in readme.lower() or "Replacement" in readme
    assert (root / "docs" / "evaluation.md").exists()
    assert (root / "docs" / "populations.md").exists()
    assert (root / "docs" / "eval-clean-room.md").exists()
