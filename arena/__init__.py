"""Arena — local-first portable RL policy handoff and evaluation."""

from arena._version import VERSION as __version__
from arena.core.sdk import Evaluation, Match, Policy, Population, Task, check

__all__ = ["Task", "Policy", "Match", "Population", "Evaluation", "check", "__version__"]
