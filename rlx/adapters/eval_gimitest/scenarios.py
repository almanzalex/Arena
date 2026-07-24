"""RLX-owned non-no-op Gimitest qualification scenarios."""

from __future__ import annotations

from typing import Any

from gimitest.gtest import GTest


class RewardTransformScenario(GTest):
    """Deterministically transform rewards to prove Gimitest hooks executed.

    This is a qualification fixture, not a claim that transformed rewards are
    equivalent to the native task. Its parameters are semantic provider config
    and therefore participate in the evaluation-intent digest.
    """

    def post_step_test(
        self,
        state: Any,
        action: Any,
        next_state: Any,
        reward: Any,
        terminated: Any,
        truncated: Any,
        info: Any,
        agent_selection: Any,
    ) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
        scale = float(self.parameters.get("reward_scale", -1.0))
        offset = float(self.parameters.get("reward_offset", 0.0))
        if isinstance(reward, dict):
            transformed = {
                key: float(value) * scale + offset
                for key, value in reward.items()
            }
        else:
            transformed = float(reward) * scale + offset
        return (
            state,
            action,
            next_state,
            transformed,
            terminated,
            truncated,
            info,
        )
