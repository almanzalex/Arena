"""Public Python SDK: Task, Policy, Match, Population, Evaluation, check."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rlx.core.compatibility import compose_check
from rlx.core.errors import CompatibilityReport, SchemaError
from rlx.core.manifests import (
    evaluation_content_digest,
    load_manifest,
    policy_content_digest,
    population_content_digest,
    resolve_artifact_path,
    validate_evaluation_manifest,
    validate_match_manifest,
    validate_policy_manifest,
    validate_population_manifest,
    validate_task_manifest,
)


class Policy:
    def __init__(self, manifest: dict[str, Any], *, root: Path | None = None) -> None:
        self.manifest = validate_policy_manifest(manifest)
        self.root = root
        self.digest = policy_content_digest(self.manifest)

    @classmethod
    def load(cls, path: str | Path) -> Policy:
        path = Path(path)
        manifest_path = resolve_artifact_path(path)
        root = manifest_path.parent if manifest_path.parent.is_dir() else path.parent
        if path.is_dir():
            root = path
        manifest = load_manifest(manifest_path)
        return cls(manifest, root=root)

    @property
    def name(self) -> str:
        return str(self.manifest["name"])

    def as_role(self, role: str) -> PolicyBinding:
        return PolicyBinding(self, role)

    def payload_path(self, key: str = "weights") -> Path:
        if self.root is None:
            raise SchemaError("policy has no root directory for payloads")
        payloads = self.manifest["payloads"]
        entry = payloads[key]
        if isinstance(entry, dict) and "path" in entry:
            return (self.root / entry["path"]).resolve()
        # digest-only: look under payloads/
        candidate = self.root / "payloads" / f"{key}.pt"
        if candidate.exists():
            return candidate
        raise SchemaError(f"cannot resolve payload {key!r} under {self.root}")


class PolicyBinding:
    def __init__(self, policy: Policy, role: str) -> None:
        self.policy = policy
        self.role = role


class Task:
    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = spec

    @classmethod
    def load(cls, ref: str | Path | dict[str, Any]) -> Task:
        if isinstance(ref, dict):
            return cls(ref)
        path = Path(ref)
        if path.exists():
            manifest = load_manifest(path)
            if manifest.get("schema") == "rlx.task/v0alpha1":
                validate_task_manifest(manifest)
            return cls(manifest)
        # Inline adapter ref string: pettingzoo://rlx/competitive_rps_v0
        if isinstance(ref, str) and ref.startswith("pettingzoo://"):
            env = ref.removeprefix("pettingzoo://")
            return cls(
                {
                    "adapter": "pettingzoo-parallel",
                    "env": env,
                    "version": "pettingzoo",
                }
            )
        if isinstance(ref, str) and ref.startswith("openspiel://"):
            from rlx.adapters.task_openspiel import interaction_for_game

            return cls(
                {
                    "adapter": "openspiel",
                    "env": ref,
                    "interaction": interaction_for_game(ref),
                    "packaging": {"kind": "openspiel"},
                }
            )
        raise SchemaError(f"cannot load task: {ref!r}")

    def role_spaces(self) -> dict[str, dict[str, Any]]:
        from rlx.adapters.task_pettingzoo.adapter import describe_task

        return describe_task(self.spec)

    def provides_masks(self) -> bool:
        info = self.role_spaces()
        return bool(info.get("provides_masks", False))


def check(
    task: Task | dict[str, Any] | str | Path,
    *bindings: PolicyBinding | tuple[Policy, str],
    action_mode: str | None = None,
) -> CompatibilityReport:
    if not isinstance(task, Task):
        task = Task.load(task)
    spaces = task.role_spaces()
    roles_meta = spaces.get("roles", {})
    issues_report = CompatibilityReport(ok=True, issues=[])
    for binding in bindings:
        if isinstance(binding, tuple):
            policy, role = binding
            binding = PolicyBinding(policy, role)
        role = binding.role
        meta = roles_meta.get(role) or roles_meta.get(_agent_to_role(role, roles_meta), {})
        expected_obs = meta.get("observation") if meta else None
        expected_act = meta.get("action") if meta else None
        # Also try agent id directly
        if expected_obs is None and role in roles_meta:
            expected_obs = roles_meta[role].get("observation")
            expected_act = roles_meta[role].get("action")
        part = compose_check(
            policy=binding.policy.manifest,
            role=_normalize_role(role, binding.policy.manifest),
            expected_obs=expected_obs,
            expected_act=expected_act,
            action_mode=action_mode,
            task_provides_masks=task.provides_masks(),
        )
        issues_report.issues.extend(part.issues)
    issues_report.ok = len(issues_report.issues) == 0
    return issues_report


def _normalize_role(role: str, policy: dict[str, Any]) -> str:
    allowed = policy.get("roles", {}).get("allowed", [])
    if role in allowed:
        return role
    # agent ids like player_0 may map via alias
    return role


def _agent_to_role(agent: str, roles_meta: dict[str, Any]) -> str:
    for role, meta in roles_meta.items():
        if agent in meta.get("agents", []) or agent == role:
            return role
    return agent


class Match:
    def __init__(
        self,
        *,
        task: Task,
        assignments: dict[str, Policy],
        action_mode: str = "deterministic",
        failure_policy: dict[str, Any] | None = None,
    ) -> None:
        self.task = task
        self.assignments = assignments
        self.action_mode = action_mode
        self.failure_policy = failure_policy or {
            "timeout_seconds": 60,
            "retain_incomplete": True,
            "retry": 0,
        }

    @classmethod
    def from_manifest(cls, path: str | Path) -> Match:
        data = validate_match_manifest(load_manifest(resolve_artifact_path(path)))
        task = Task.load(data["task"])
        assignments: dict[str, Policy] = {}
        base = Path(path).parent if Path(path).exists() else Path.cwd()
        for role, pref in data["assignments"].items():
            ppath = Path(pref)
            if not ppath.is_absolute():
                ppath = (base / ppath).resolve()
            assignments[role] = Policy.load(ppath)
        return cls(
            task=task,
            assignments=assignments,
            action_mode=data.get("action_mode", "deterministic"),
            failure_policy=data.get("failure_policy"),
        )

    def run(
        self,
        seeds: range | list[int] | None = None,
        *,
        record: bool = True,
        out: str | Path | None = None,
    ) -> dict[str, Any]:
        from rlx.plugins.interactions import get_interaction

        seed_list: list[int]
        if seeds is None:
            seed_list = list(range(1))
        elif isinstance(seeds, range):
            seed_list = list(seeds)
        else:
            seed_list = list(seeds)

        interaction = str(self.task.spec.get("interaction", "parallel"))
        return get_interaction(interaction).run_match(
            task_spec=self.task.spec,
            assignments=self.assignments,
            seeds=seed_list,
            action_mode=self.action_mode,
            record=record,
            out_dir=Path(out) if out else None,
            failure_policy=self.failure_policy,
        )


class Population:
    """Content-addressed set of policy digests (RLX 0.2)."""

    def __init__(self, manifest: dict[str, Any]) -> None:
        self.manifest = validate_population_manifest(manifest)
        self.digest = self.manifest.get("digest") or population_content_digest(self.manifest)

    @classmethod
    def load(cls, ref: str | Path, *, store: Any | None = None) -> Population:
        from rlx.core.population import load_population

        return cls(load_population(ref, store=store))

    @classmethod
    def create(
        cls,
        *,
        name: str,
        members: list[dict[str, Any]],
        store: Any,
        ref: str | None = None,
    ) -> Population:
        from rlx.core.population import create_population

        return cls(create_population(name=name, members=members, store=store, ref=ref))

    @property
    def name(self) -> str:
        return str(self.manifest["name"])

    @property
    def members(self) -> list[dict[str, Any]]:
        return list(self.manifest["members"])


class Evaluation:
    """Versioned evaluation suite (RLX 0.2)."""

    def __init__(self, manifest: dict[str, Any]) -> None:
        self.manifest = validate_evaluation_manifest(manifest)
        self.digest = evaluation_content_digest(self.manifest)

    @classmethod
    def load(cls, path: str | Path) -> Evaluation:
        from rlx.runtime.evaluation import load_evaluation

        return cls(load_evaluation(path))

    def validate(self, *, populations: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
        from rlx.runtime.evaluation import validate_evaluation

        return validate_evaluation(self.manifest, populations=populations)

    def run(
        self,
        *,
        policy_index: dict[str, Path],
        populations: dict[str, dict[str, Any]] | None = None,
        store: Any | None = None,
        out_dir: Path | None = None,
        workers: int = 1,
    ) -> dict[str, Any]:
        from rlx.runtime.evaluation import run_evaluation

        return run_evaluation(
            self.manifest,
            policy_index=policy_index,
            populations=populations,
            store=store,
            out_dir=out_dir,
            workers=workers,
        )
