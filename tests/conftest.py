"""Test-process defaults for headless PettingZoo/MPE environments."""

from __future__ import annotations

import os

# Must be set before mpe2 imports pygame. This makes the test suite headless on
# macOS and CI without requiring an undocumented shell environment override.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
