"""CLI end-to-end for population / eval / select / bundle (0.2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("pettingzoo")

from _eval_fixtures import build_cyclic_rps_eval_fixture

from arena.cli.main import main
from arena.core.sdk import Policy
from arena.core.store import LocalStore


def _run(argv: list[str], capsys) -> tuple[int, str, str]:
    try:
        code = main(argv)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 2)
    out = capsys.readouterr()
    return code, out.out, out.err


@pytest.mark.acceptance
@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_cli_population_eval_select_bundle(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    LocalStore(tmp_path).init()
    fx = build_cyclic_rps_eval_fixture(tmp_path / "lab")

    code, out, err = _run(
        ["population", "create", str(fx["population"]), "--ref", "populations/cyclic", "--json"],
        capsys,
    )
    assert code == 0, err
    pop = json.loads(out)
    assert pop["digest"].startswith("sha256:")
    assert len(pop["members"]) == 3

    code, out, err = _run(["population", "inspect", "populations/cyclic", "--json"], capsys)
    assert code == 0, err
    assert json.loads(out)["digest"] == pop["digest"]

    policy_args: list[str] = []
    for path in fx["bundles"].values():
        d = Policy.load(path).digest
        policy_args.extend(["--policy", f"{d}={path}"])

    code, _out, err = _run(
        ["eval", "validate", str(fx["evaluation"]), "--population", str(fx["population"])],
        capsys,
    )
    assert code == 0, err

    eval_out = tmp_path / "eval-run"
    code, out, err = _run(
        [
            "eval",
            "run",
            str(fx["evaluation"]),
            *policy_args,
            "--population",
            str(fx["population"]),
            "--out",
            str(eval_out),
            "--json",
        ],
        capsys,
    )
    assert code == 0, err
    summary = json.loads(out)
    assert summary["cells"] == 9
    run_dir = Path(summary["run_dir"])
    assert (run_dir / "eval_run.json").exists()

    code, out, err = _run(["eval", "report", str(run_dir), "--json"], capsys)
    assert code == 0, err
    report = json.loads(out)
    assert report["nontransitivity_warning"]
    assert report["metrics"]["payoff_matrix"]["ranking"] is None
    assert "ci" in report["metrics"]["mean_return"]

    rock_d = fx["digests"]["rock"]
    slice_dir = tmp_path / "slice"
    code, out, err = _run(
        [
            "data",
            "select",
            str(run_dir),
            "--out",
            str(slice_dir),
            "--policy",
            rock_d,
            "--role",
            "player_0",
            "--outcome",
            "loss",
            "--json",
        ],
        capsys,
    )
    assert code == 0, err
    dataset = json.loads(out)
    assert len(dataset["episodes"]) >= 1

    bundle_dir = tmp_path / "bundle"
    code, _out, err = _run(
        [
            "eval",
            "bundle",
            str(run_dir),
            "--out",
            str(bundle_dir),
            "--report",
            str(run_dir / "report.json"),
        ],
        capsys,
    )
    assert code == 0, err
    assert (bundle_dir / "bundle.json").exists()
