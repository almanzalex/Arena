"""CLI smoke tests (core commands without heavy extras)."""

from pathlib import Path

from arena.cli.main import main


def test_cli_init(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["init"]) == 0
    assert (tmp_path / ".arena" / "workspace.toml").exists()
    # idempotent
    assert main(["init"]) == 0
