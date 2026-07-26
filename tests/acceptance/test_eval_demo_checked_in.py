"""Checked-in examples/eval/demo must run without regenerate (usable claim)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("pettingzoo")

from arena.cli.main import main
from arena.core.sdk import Policy
from arena.core.store import LocalStore

DEMO = Path(__file__).resolve().parents[2] / "examples" / "eval" / "demo"


@pytest.mark.acceptance
@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_checked_in_eval_demo_cli(tmp_path: Path, monkeypatch, capsys) -> None:
    assert (DEMO / "evaluation.yaml").exists(), "run python examples/eval/generate_demo.py"
    # Copy demo into tmp so we don't dirty the tree with .arena/eval-run
    import shutil

    work = tmp_path / "demo"
    shutil.copytree(DEMO, work)
    monkeypatch.chdir(work)
    LocalStore(work).init()

    assert main(["population", "create", "population.yaml", "--ref", "populations/opp"]) == 0
    digests = {n: Policy.load(f"{n}.arena").digest for n in ("rock", "paper", "scissors")}
    argv = [
        "eval",
        "run",
        "evaluation.yaml",
        *[x for n, d in digests.items() for x in ("--policy", f"{d}=./{n}.arena")],
        "--population",
        "population.yaml",
        "--out",
        "eval-run",
        "--json",
    ]
    code = main(argv)
    assert code == 0, capsys.readouterr()
    assert main(["eval", "report", "eval-run", "--json"]) == 0
    out = capsys.readouterr().out
    assert "nontransitivity_warning" in out
    assert main(
        [
            "data",
            "select",
            "eval-run",
            "--out",
            "slice",
            "--outcome",
            "loss",
            "--role",
            "player_0",
        ]
    ) == 0
    assert main(
        ["eval", "bundle", "eval-run", "--out", "bundle", "--report", "eval-run/report.json"]
    ) == 0
    assert (work / "bundle" / "bundle.json").exists()
