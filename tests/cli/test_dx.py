"""DX: help topics and shell completion."""

from __future__ import annotations

import json

import pytest

from arena.cli.dx import COMMAND_TREE, HELP_TOPICS, render_completion, render_help
from arena.cli.main import main


def test_help_overview(capsys) -> None:
    assert main(["help"]) == 0
    out = capsys.readouterr().out
    assert "Arena CLI overview" in out
    assert "arena doctor" in out


def test_help_topics_list(capsys) -> None:
    assert main(["help", "topics"]) == 0
    out = capsys.readouterr().out
    for topic in HELP_TOPICS:
        assert topic in out


def test_help_naming_mentions_rename_candidates(capsys) -> None:
    assert main(["help", "naming"]) == 0
    out = capsys.readouterr().out
    assert "diambra-arena" in out
    assert "arena-rl" in out
    assert "rlx-arena" in out


def test_help_unknown_topic_fails_loud(capsys) -> None:
    code = main(["help", "not-a-topic"])
    assert code == 2
    err = capsys.readouterr().err
    assert "unknown help topic" in err
    assert "Traceback" not in err


def test_help_json_envelope(capsys) -> None:
    assert main(["--json", "help", "install"]) == 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["schema"] == "arena.cli-result/v1"
    assert envelope["ok"] is True
    assert "pip install" in envelope["data"]["text"]


@pytest.mark.parametrize("shell", ["bash", "zsh", "fish"])
def test_completion_scripts(shell: str, capsys) -> None:
    assert main(["completion", shell]) == 0
    out = capsys.readouterr().out
    assert "arena" in out
    assert "doctor" in out
    assert "eval" in out
    # Nested commands appear for shells that complete them.
    if shell != "fish":
        assert "handoff" in out
    else:
        assert "demo" in out


def test_completion_invalid_shell_via_argparse(capsys) -> None:
    code = main(["completion", "tcsh"])
    assert code == 2
    err = capsys.readouterr().err
    assert "Traceback" not in err


def test_render_helpers_direct() -> None:
    assert "Pinned release candidate" in render_help("install")
    bash = render_completion("bash")
    assert "complete -F _arena_completion arena" in bash
    assert set(COMMAND_TREE) >= {"doctor", "demo", "eval", "help", "completion"}
