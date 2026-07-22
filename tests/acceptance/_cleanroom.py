"""Shared helpers for the hermetic, doc-driven U-01 clean-room gate.

The single source of truth for the clean-room command sequence is
``docs/clean-room.md``. These helpers parse the commands out of that document
so the test executes *exactly* what a human reader would type, and fail loudly
if the documentation drifts from the canonical U-01 flow.
"""

from __future__ import annotations

import re
from pathlib import Path

# Info string that marks the machine-parseable command block in clean-room.md.
CLEAN_ROOM_TAG = "rlx-clean-room"

# The canonical U-01 clean-room flow. The documented block must match this
# exactly (order included): any drift is treated as a release blocker.
EXPECTED_COMMANDS: tuple[str, ...] = (
    "rlx init",
    "rlx inspect ./player_0.rlx",
    "rlx inspect ./player_1.rlx",
    "rlx check rlx/competitive_rps_v0 ./player_0.rlx --role player_0",
    "rlx check rlx/competitive_rps_v0 ./player_1.rlx --role player_1",
    "rlx match run ./match.yaml --record --out ./runs/baseline-match",
    "rlx data inspect ./runs/baseline-match/trajectories",
)

# Tokens that must never appear in the clean-room block: they would imply the
# recipient needs the trainer, the source checkpoints, or a re-install.
FORBIDDEN_TOKENS: tuple[str, ...] = (
    "trainer",
    "checkpoint",
    ".pt",
    "policy export",
    "pip install",
    "pip3 install",
    "python ",
    "python3 ",
    "git clone",
    "-e .",
    "import ",
    "conda ",
)

_FENCE_RE = re.compile(
    r"^```(?P<info>[^\n]*)\n(?P<body>.*?)^```",
    re.DOTALL | re.MULTILINE,
)


def parse_cleanroom_commands(markdown_path: Path | str) -> list[str]:
    """Return the ordered command list from the ``rlx-clean-room`` fenced block.

    Raises ``AssertionError`` (so the gate fails) if the block is missing or empty.
    """
    text = Path(markdown_path).read_text(encoding="utf-8")
    bodies: list[str] = []
    for m in _FENCE_RE.finditer(text):
        info = m.group("info").strip().split()
        # Info string is e.g. ``bash rlx-clean-room`` — highlighted as bash,
        # tagged for automation.
        if CLEAN_ROOM_TAG in info:
            bodies.append(m.group("body"))

    assert bodies, (
        f"no fenced block tagged '{CLEAN_ROOM_TAG}' found in {markdown_path}: "
        "the clean-room command block is missing or lost its automation tag"
    )
    assert len(bodies) == 1, (
        f"expected exactly one '{CLEAN_ROOM_TAG}' block in {markdown_path}, "
        f"found {len(bodies)}"
    )

    commands: list[str] = []
    for raw in bodies[0].splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        commands.append(line)

    assert commands, f"'{CLEAN_ROOM_TAG}' block in {markdown_path} contains no commands"
    return commands


def validate_cleanroom_commands(commands: list[str]) -> None:
    """Fail if the documented commands drift from the canonical U-01 flow.

    This turns documentation drift (missing/wrong/out-of-order commands, or any
    reference to the trainer) into a test failure.
    """
    assert commands == list(EXPECTED_COMMANDS), (
        "docs/clean-room.md command block drifted from the canonical U-01 flow.\n"
        f"  expected: {list(EXPECTED_COMMANDS)}\n"
        f"  parsed:   {commands}\n"
        "Update the docs or the CLI so they agree (doc drift is a release blocker)."
    )
    for cmd in commands:
        assert cmd.startswith("rlx "), (
            f"clean-room command does not invoke the rlx CLI: {cmd!r} "
            "(a recipient must only need `rlx`, never the trainer or a re-install)"
        )
        lowered = cmd.lower()
        for token in FORBIDDEN_TOKENS:
            assert token not in lowered, (
                f"clean-room command {cmd!r} references forbidden token {token!r}: "
                "the documented flow must not depend on the trainer/checkpoints/re-install"
            )
