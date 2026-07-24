"""Claim 9 (adversarial): CLI hostility.

Attacks: bad/missing args, nonexistent artifacts, malformed manifests, wrong schema
version -> clear non-zero exit, no raw traceback vomit, and no partial writes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rlx.cli.main import main


def _run(argv: list[str], capsys) -> tuple[int, str, str]:
    """Run the CLI, normalizing argparse SystemExit into an exit code."""
    try:
        code = main(argv)
    except SystemExit as e:  # argparse / explicit SystemExit
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 2)
    out = capsys.readouterr()
    return code, out.out, out.err


def test_no_subcommand_errors(capsys) -> None:
    code, _out, err = _run([], capsys)
    assert code != 0
    assert "Traceback" not in err


@pytest.mark.parametrize(
    "argv",
    [
        ["check"],  # missing task/policy/--role
        ["check", "task"],  # missing policy/--role
        ["policy"],  # missing subcommand
        ["match"],  # missing subcommand
        ["match", "run"],  # missing manifest
        ["data", "inspect"],  # missing trajectory
    ],
)
def test_missing_args_error_cleanly(argv, capsys) -> None:
    code, _out, err = _run(argv, capsys)
    assert code != 0
    assert "Traceback" not in err


def test_inspect_nonexistent_artifact(tmp_path: Path, capsys) -> None:
    code, _out, err = _run(["inspect", str(tmp_path / "nope.rlx")], capsys)
    assert code == 3
    assert err.startswith("error [")
    assert "Traceback" not in err


def test_inspect_malformed_manifest(tmp_path: Path, capsys) -> None:
    bad = tmp_path / "manifest.yaml"
    bad.write_text("- just\n- a\n- list\n", encoding="utf-8")  # root is not a mapping
    code, _out, err = _run(["inspect", str(bad)], capsys)
    assert code == 3
    assert err.startswith("error [")
    assert "Traceback" not in err


def test_inspect_unparseable_yaml(tmp_path: Path, capsys) -> None:
    bad = tmp_path / "manifest.yaml"
    bad.write_text("key: : : [unbalanced\n", encoding="utf-8")
    code, _out, err = _run(["inspect", str(bad)], capsys)
    assert code == 3
    assert "Traceback" not in err


def test_match_run_wrong_schema_no_partial_write(tmp_path: Path, monkeypatch, capsys) -> None:
    """A match manifest with the wrong schema version fails validation before execution;
    exit is non-zero and no run/output directory is created."""
    monkeypatch.chdir(tmp_path)
    bad_match = tmp_path / "match.yaml"
    bad_match.write_text(
        "schema: rlx.match/v999\n"
        "task: {adapter: pettingzoo-parallel, env: rlx/competitive_rps_v0}\n"
        "assignments: {player_0: ./p0}\n"
        "seeds: {start: 0, count: 5}\n"
        "action_mode: deterministic\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "should_not_exist"
    code, _out, err = _run(["match", "run", str(bad_match), "--out", str(out_dir)], capsys)
    assert code != 0
    assert "Traceback" not in err
    assert not out_dir.exists()
    assert not (tmp_path / "runs").exists()


def test_match_run_missing_manifest_file(tmp_path: Path, capsys) -> None:
    code, _out, err = _run(["match", "run", str(tmp_path / "absent.yaml")], capsys)
    assert code == 3
    assert err.startswith("error [")
    assert "Traceback" not in err


def test_policy_export_without_spec_errors(tmp_path: Path, capsys) -> None:
    """Missing arch/obs/action (and no --spec) is a clean error, not a crash."""
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(b"not-a-real-checkpoint")
    code, _out, err = _run(
        ["policy", "export", "--source", str(ckpt), "--out", str(tmp_path / "o"), "--role", "x"],
        capsys,
    )
    assert code != 0
    assert "Traceback" not in err
    assert not (tmp_path / "o" / "policy.yaml").exists()
