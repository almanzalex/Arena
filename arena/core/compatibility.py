"""Compatibility checks between tasks, policies, and match assignments."""

from __future__ import annotations

from typing import Any

from arena.core.contracts import architecture_space_issues
from arena.core.errors import CompatibilityIssue, CompatibilityReport
from arena.core.spaces import spaces_compatible


def check_policy_role(policy: dict[str, Any], role: str) -> list[CompatibilityIssue]:
    allowed = list(policy.get("roles", {}).get("allowed", []))
    if role not in allowed:
        return [
            CompatibilityIssue(
                code="ROLE_MISMATCH",
                message=f'policy "{policy.get("name")}" cannot control role "{role}"',
                evidence={
                    "policy.roles.allowed": allowed,
                    "match.assignment.role": role,
                },
                repairs=[
                    f'assign the policy to one of: {allowed}',
                    "export a new policy with verified role eligibility",
                ],
            )
        ]
    return []


def check_spaces(
    *,
    role: str,
    expected_obs: dict[str, Any] | None,
    expected_act: dict[str, Any] | None,
    policy: dict[str, Any],
) -> list[CompatibilityIssue]:
    issues: list[CompatibilityIssue] = []
    if expected_obs is not None:
        mism = spaces_compatible(expected_obs, policy["observation"])
        if mism:
            issues.append(
                CompatibilityIssue(
                    code="OBSERVATION_MISMATCH",
                    message=f'observation contract mismatch for role "{role}"',
                    evidence={
                        "expected": expected_obs,
                        "policy.observation": policy["observation"],
                        "details": mism,
                    },
                    repairs=["re-export the policy against the task observation schema"],
                )
            )
    if expected_act is not None:
        mism = spaces_compatible(expected_act, policy["action"])
        if mism:
            issues.append(
                CompatibilityIssue(
                    code="ACTION_MISMATCH",
                    message=f'action contract mismatch for role "{role}"',
                    evidence={
                        "expected": expected_act,
                        "policy.action": policy["action"],
                        "details": mism,
                    },
                    repairs=["re-export the policy against the task action schema"],
                )
            )
    return issues


def check_inference_mode(policy: dict[str, Any], action_mode: str) -> list[CompatibilityIssue]:
    modes = set(policy.get("inference", {}).get("modes", []))
    if action_mode not in modes:
        return [
            CompatibilityIssue(
                code="INFERENCE_MODE",
                message=f'policy does not support action_mode "{action_mode}"',
                evidence={"policy.inference.modes": sorted(modes), "action_mode": action_mode},
                repairs=[f"export with inference.modes including {action_mode}"],
            )
        ]
    return []


def check_masks(policy: dict[str, Any], task_provides_masks: bool | None) -> list[CompatibilityIssue]:
    req = policy.get("action", {}).get("masks", "none")
    if req == "required" and task_provides_masks is False:
        return [
            CompatibilityIssue(
                code="MASK_REQUIRED",
                message="policy requires action masks but task does not provide them",
                evidence={"policy.action.masks": req, "task.provides_masks": task_provides_masks},
                repairs=["use a task that emits action masks", "export without required masks"],
            )
        ]
    return []


def check_recurrent(policy: dict[str, Any], task_supports_recurrent: bool = True) -> list[CompatibilityIssue]:
    if policy.get("state", {}).get("recurrent") and not task_supports_recurrent:
        return [
            CompatibilityIssue(
                code="RECURRENT_UNSUPPORTED",
                message="policy is recurrent but task adapter cannot reset agent state",
                evidence={"policy.state.recurrent": True},
                repairs=["use a non-recurrent policy export"],
            )
        ]
    return []


def check_preprocessing(
    policy: dict[str, Any],
    expected_id: str | None = None,
) -> list[CompatibilityIssue]:
    prep = policy.get("preprocessing", {})
    if not prep.get("included", False):
        return [
            CompatibilityIssue(
                code="PREPROCESSING_MISSING",
                message="policy preprocessing must be included in the bundle for MVP",
                evidence={"preprocessing": prep},
                repairs=["re-export with preprocessing.included=true"],
            )
        ]
    if expected_id is not None and prep.get("id") != expected_id:
        return [
            CompatibilityIssue(
                code="PREPROCESSING_ID",
                message="preprocessing identifier mismatch",
                evidence={"expected": expected_id, "policy.preprocessing.id": prep.get("id")},
                repairs=["align preprocessing ids or re-export"],
            )
        ]
    return []


def compose_check(
    *,
    policy: dict[str, Any],
    role: str,
    expected_obs: dict[str, Any] | None = None,
    expected_act: dict[str, Any] | None = None,
    action_mode: str | None = None,
    task_provides_masks: bool | None = None,
    expected_preprocessing_id: str | None = None,
) -> CompatibilityReport:
    issues: list[CompatibilityIssue] = []
    issues.extend(check_policy_role(policy, role))
    issues.extend(check_spaces(role=role, expected_obs=expected_obs, expected_act=expected_act, policy=policy))
    issues.extend(architecture_space_issues(policy))
    issues.extend(check_preprocessing(policy, expected_preprocessing_id))
    issues.extend(check_masks(policy, task_provides_masks))
    issues.extend(check_recurrent(policy))
    if action_mode is not None:
        issues.extend(check_inference_mode(policy, action_mode))
    return CompatibilityReport(ok=len(issues) == 0, issues=issues)
