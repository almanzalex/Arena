"""DX helpers: shell completion scripts and `arena help` topics."""

from __future__ import annotations

from typing import Any

# Top-level command → nested subcommands (empty = leaf).
# Keep in sync with arena.cli.main parser grammar.
COMMAND_TREE: dict[str, list[str]] = {
    "adapter": ["qualify"],
    "attest": ["keygen", "sign", "verify"],
    "capture": [],
    "check": [],
    "completion": [],
    "data": ["inspect", "select", "materialize"],
    "demo": ["handoff", "multiagent"],
    "doctor": [],
    "eval": ["validate", "run", "report", "bundle", "compare", "matrix"],
    "help": [],
    "init": [],
    "inspect": [],
    "match": ["run"],
    "policy": ["export", "verify"],
    "population": ["create", "inspect"],
    "pull": [],
    "push": [],
    "release": ["build", "assemble", "sign", "verify"],
    "schema": ["list"],
    "store": ["qualify"],
    "task": ["import", "verify-equivalence"],
    "train": [],
}

HELP_TOPICS: dict[str, str] = {
    "overview": """\
Arena CLI overview
==================

Local-first interoperability for portable RL policies, evaluations, and release
evidence. Common entry points:

  arena doctor [--capability NAME] [--json]
  arena demo handoff --out ./arena-demo
  arena inspect PATH
  arena schema list --json

Global flags (may appear before the subcommand):

  --json      Emit a versioned JSON result / diagnostic
  --debug     Include a redacted traceback on failure
  --version   Print the installed Arena version

Topics: arena help {overview,install,handoff,completion,naming}
Docs:   https://github.com/almanzalex/Arena/tree/main/docs
""",
    "install": """\
Install
=======

Pinned release candidate (recommended):

  python -m pip install 'arena[quickstart]==1.0.0rc1'
  arena --version
  arena doctor --json

From a checkout:

  python -m pip install -e '.[dev]'

Optional extras: openenv, openspiel, gimitest, hf, wandb, mlflow, signing,
completion (argcomplete for richer tab completion).

Always pin a version. After install, confirm identity with `arena --version`
and `arena doctor` so a wrong PyPI package cannot silently look like Arena.
See also: arena help naming
""",
    "handoff": """\
Handoff demo
============

Source-free local value demonstration (no network after install):

  arena demo handoff --out ./arena-demo
  arena inspect ./arena-demo/restored-policy.arena

This exports a reference policy, verifies it, mirrors through file://, pulls to
a new path, and proves the digest is unchanged. More flows:
docs/1.0-user-flows.md
""",
    "completion": """\
Shell completion
================

Static completion (no extra deps) — add to your shell rc:

  eval "$(arena completion bash)"    # bash
  eval "$(arena completion zsh)"     # zsh
  arena completion fish | source     # fish

Richer argparse completion (optional):

  python -m pip install 'arena[completion]'
  eval "$(register-python-argcomplete arena)"

`arena completion` always prints a script you can eval; it does not modify files.
""",
    "naming": """\
PyPI naming
===========

This project's distribution name on PyPI is currently `arena`. That short name
is easy to confuse with unrelated packages that also use “arena” in the title:

  diambra-arena   fighting-game / DIAMBRA environments
  rl-arena        competitive RL environment tooling

They are different products. Prefer a pinned install and verify:

  python -m pip install 'arena[quickstart]==1.0.0rc1'
  arena --version
  arena doctor --json

If install collisions cause real harm, candidate distribution renames (deferred;
CLI entry point would stay `arena` unless a coordinated break is planned):

  arena-rl
  arena-interop
  rlx-arena
  portable-arena

Do not rename casually — digests, docs, and published wheels share the current
identity. Tracked in TODOS.md.
""",
}


def help_topic_names() -> list[str]:
    return sorted(HELP_TOPICS)


def render_help(topic: str | None) -> str:
    key = (topic or "overview").strip().lower()
    if key in {"topics", "list", "--list"}:
        lines = ["Available help topics:", ""]
        for name in help_topic_names():
            first = HELP_TOPICS[name].strip().splitlines()[0]
            lines.append(f"  {name:<12} {first}")
        lines.append("")
        lines.append("Usage: arena help [topic]")
        return "\n".join(lines) + "\n"
    body = HELP_TOPICS.get(key)
    if body is None:
        known = ", ".join(help_topic_names())
        raise ValueError(f"unknown help topic {topic!r}; choose one of: {known}")
    return body if body.endswith("\n") else body + "\n"


def enable_argcomplete(parser: Any) -> bool:
    """Register argcomplete when the optional extra is installed."""
    try:
        import argcomplete
    except ImportError:
        return False
    argcomplete.autocomplete(parser)
    return True


def render_completion(shell: str) -> str:
    shell = shell.strip().lower()
    if shell == "bash":
        return _bash_completion()
    if shell == "zsh":
        return _zsh_completion()
    if shell == "fish":
        return _fish_completion()
    raise ValueError(f"unsupported shell {shell!r}; choose bash, zsh, or fish")


def _top_level_commands() -> list[str]:
    return sorted(COMMAND_TREE)


def _bash_completion() -> str:
    tops = " ".join(_top_level_commands())
    cases: list[str] = []
    for cmd, subs in sorted(COMMAND_TREE.items()):
        if not subs:
            continue
        joined = " ".join(subs)
        cases.append(
            f"    {cmd})\n      COMPREPLY=( $(compgen -W '{joined}' -- \"$cur\") )\n      return 0\n      ;;"
        )
    nested = "\n".join(cases)
    help_topics = " ".join(help_topic_names())
    return f"""\
# Arena bash completion — eval "$(arena completion bash)"
_arena_completion() {{
  local cur prev
  COMPREPLY=()
  cur="${{COMP_WORDS[COMP_CWORD]}}"
  prev="${{COMP_WORDS[COMP_CWORD-1]}}"
  if [[ ${{COMP_CWORD}} -eq 1 ]]; then
    COMPREPLY=( $(compgen -W '{tops}' -- "$cur") )
    return 0
  fi
  if [[ ${{COMP_WORDS[1]}} == help && ${{COMP_CWORD}} -eq 2 ]]; then
    COMPREPLY=( $(compgen -W '{help_topics}' -- "$cur") )
    return 0
  fi
  if [[ ${{COMP_WORDS[1]}} == completion && ${{COMP_CWORD}} -eq 2 ]]; then
    COMPREPLY=( $(compgen -W 'bash zsh fish' -- "$cur") )
    return 0
  fi
  case "${{COMP_WORDS[1]}}" in
{nested}
  esac
  return 0
}}
complete -F _arena_completion arena
"""


def _zsh_completion() -> str:
    tops = " ".join(_top_level_commands())
    lines = [
        "# Arena zsh completion — eval \"$(arena completion zsh)\"",
        "autoload -U +X compinit && compinit -u >/dev/null 2>&1 || true",
        "autoload -U +X bashcompinit && bashcompinit >/dev/null 2>&1 || true",
        "_arena_completion() {",
        '  local cur prev',
        "  COMPREPLY=()",
        '  cur="${COMP_WORDS[COMP_CWORD]}"',
        '  if [[ ${COMP_CWORD} -eq 1 ]]; then',
        f"    COMPREPLY=( $(compgen -W '{tops}' -- \"$cur\") )",
        "    return 0",
        "  fi",
        '  case "${COMP_WORDS[1]}" in',
    ]
    for cmd, subs in sorted(COMMAND_TREE.items()):
        if not subs:
            continue
        joined = " ".join(subs)
        lines.append(f"    {cmd})")
        lines.append(f"      COMPREPLY=( $(compgen -W '{joined}' -- \"$cur\") )")
        lines.append("      ;;")
    help_topics = " ".join(help_topic_names())
    lines.extend(
        [
            "    help)",
            f"      COMPREPLY=( $(compgen -W '{help_topics}' -- \"$cur\") )",
            "      ;;",
            "    completion)",
            "      COMPREPLY=( $(compgen -W 'bash zsh fish' -- \"$cur\") )",
            "      ;;",
            "  esac",
            "}",
            "complete -F _arena_completion arena",
            "",
        ]
    )
    return "\n".join(lines)


def _fish_completion() -> str:
    lines = ["# Arena fish completion — arena completion fish | source"]
    for cmd, subs in sorted(COMMAND_TREE.items()):
        lines.append(f"complete -c arena -f -n '__fish_use_subcommand' -a {cmd}")
        for sub in subs:
            lines.append(
                f"complete -c arena -f -n '__fish_seen_subcommand_from {cmd}' -a {sub}"
            )
    for topic in help_topic_names():
        lines.append(
            f"complete -c arena -f -n '__fish_seen_subcommand_from help' -a {topic}"
        )
    for shell in ("bash", "zsh", "fish"):
        lines.append(
            f"complete -c arena -f -n '__fish_seen_subcommand_from completion' -a {shell}"
        )
    lines.append("")
    return "\n".join(lines)
