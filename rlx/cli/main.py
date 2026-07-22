"""RLX command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rlx",
        description=(
            "RLX 0.2 — portable policies, populations, and versioned evaluation "
            "(Parallel + AEC); Discrete templates plus BYO TorchScript actors"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Create a local .rlx workspace")
    p_init.add_argument("--path", default=".", help="Workspace root")
    p_init.add_argument("--force", action="store_true")

    p_inspect = sub.add_parser("inspect", help="Inspect an artifact without executing it")
    p_inspect.add_argument("artifact")
    p_inspect.add_argument("--json", action="store_true")

    p_check = sub.add_parser(
        "check",
        help="Validate task/policy composition against the configured task spaces",
    )
    p_check.add_argument(
        "task",
        help=(
            "Task env id, task YAML (with optional config), match.yaml (uses its task:), "
            "or pettingzoo://… URI"
        ),
    )
    p_check.add_argument("policy")
    p_check.add_argument("--role", required=True)
    p_check.add_argument("--action-mode", default=None)
    p_check.add_argument(
        "--config",
        default=None,
        help=(
            "Task hyperparameter config JSON/YAML (or path). Merged into task.config so "
            "spaces match a non-default env (e.g. simple_tag num_good/num_adversaries), "
            "consistent with match.yaml task.config"
        ),
    )
    p_check.add_argument("--json", action="store_true")

    p_policy = sub.add_parser(
        "policy",
        help=(
            "Export/verify Discrete templates or BYO TorchScript bundles "
            "(Discrete/MultiDiscrete/Box/Dict action cases when fully declared)"
        ),
    )
    p_pol = p_policy.add_subparsers(dest="policy_command", required=True)

    _export_help = (
        "Export a portable policy bundle. Template path: Discrete architecture + "
        "state_dict only. BYO TorchScript path: --module pkg.mod:factory "
        "(exporter-side import only; receiver loads TorchScript) supporting complete "
        "Discrete / MultiDiscrete / deterministic Box / diagonal_gaussian Box / typed "
        "Dict action cases. Incomplete action claims fail before publish. Embeds "
        "source-conformance reference cases by default. Checkpoints load with "
        "weights_only=True (fail closed); use --allow-unsafe-checkpoint only for "
        "trusted legacy pickles."
    )
    p_export = p_pol.add_parser("export", help=_export_help, description=_export_help)
    p_export.add_argument("--adapter", default="custom-pytorch")
    p_export.add_argument(
        "--source",
        default=None,
        help="Checkpoint path (.pt). Required for template export; optional weight load for --module",
    )
    p_export.add_argument("--out", required=True)
    p_export.add_argument("--role", required=True)
    p_export.add_argument("--name", default=None)
    p_export.add_argument(
        "--arch",
        default=None,
        help="Architecture JSON or path to JSON file (template export)",
    )
    p_export.add_argument("--obs", default=None, help="Observation space JSON")
    p_export.add_argument("--action", default=None, help="Action space JSON")
    p_export.add_argument("--recurrent", action="store_true")
    p_export.add_argument("--masks", default="none", choices=["none", "optional", "required"])
    p_export.add_argument("--preprocess-mean", default=None)
    p_export.add_argument("--preprocess-std", default=None)
    p_export.add_argument("--spec", default=None, help="Export spec YAML bundling arch/obs/action")
    p_export.add_argument(
        "--module",
        default=None,
        help=(
            "BYO TorchScript export: 'package.module:factory_or_class'. "
            "Imported only on the exporter; the bundle stores TorchScript + preprocess IR."
        ),
    )
    p_export.add_argument(
        "--module-args",
        default=None,
        help="JSON/YAML object of kwargs passed to the --module factory/class",
    )
    p_export.add_argument(
        "--reference-cases",
        default=None,
        help="JSON list (or {cases: [...]}) of observation cases for BYO source capture",
    )
    p_export.add_argument(
        "--allow-trace",
        action="store_true",
        help="Opt in to torch.jit.trace when scripting fails (control-flow-free actors only)",
    )
    p_export.add_argument(
        "--source-revision",
        default=None,
        help="Optional trainer/source git revision recorded in policy lineage",
    )
    p_export.add_argument(
        "--wrappers-identity",
        default=None,
        help="Optional task wrapper chain identity string recorded in lineage",
    )
    p_export.add_argument(
        "--allow-unsafe-checkpoint",
        action="store_true",
        help=(
            "Opt in to torch.load(..., weights_only=False) when the checkpoint is not "
            "tensor-only. Dangerous: can execute pickle code. Trusted sources only."
        ),
    )
    p_export.add_argument(
        "--prefer-base-weights",
        action="store_true",
        help=(
            "Opt out of the safe default: when both state_dict and ema_state_dict "
            "exist, use plain/base weights instead of EMA."
        ),
    )
    p_export.add_argument(
        "--trust-source",
        action="store_true",
        help=(
            "Opt in to trusted_source payload export/load (digest-pinned Python). "
            "NOT sandboxed. Prefer TorchScript. Refused by default."
        ),
    )
    p_export.add_argument(
        "--trusted-source-py",
        default=None,
        help="Single .py inference module for --trust-source trusted_source export",
    )
    p_export.add_argument(
        "--trusted-source-factory",
        default="build_actor",
        help="Factory/attr name inside --trusted-source-py (default: build_actor)",
    )

    _verify_help = (
        "Verify a policy bundle against reference cases. "
        "Guarantee depends on case provenance: "
        "'source-conformance' (default for `policy export`) checks the bundle "
        "against cases captured from checkpoint weights at export time; "
        "'self-consistency' only replays cases generated from the export itself "
        "and does NOT prove trainer match — a warning is emitted in that mode. "
        "Pass --source-test with trainer-captured cases for external source checks."
    )
    p_verify = p_pol.add_parser("verify", help=_verify_help, description=_verify_help)
    p_verify.add_argument("policy")
    p_verify.add_argument(
        "--source-test",
        default=None,
        help="External reference cases JSON (treated as source-conformance unless labeled otherwise)",
    )
    p_verify.add_argument(
        "--allow-self-consistency",
        action="store_true",
        help="Allow a non-passing, self-consistency-only replay when source evidence is unavailable.",
    )
    p_verify.add_argument(
        "--trust-source",
        action="store_true",
        help="Opt in to load trusted_source payload tiers (NOT sandboxed; prefer TorchScript)",
    )

    p_match = sub.add_parser("match", help="Match commands")
    p_m = p_match.add_subparsers(dest="match_command", required=True)
    p_run = p_m.add_parser("run", help="Run a seeded match")
    p_run.add_argument("match_manifest")
    p_run.add_argument("--record", action="store_true", default=True)
    p_run.add_argument("--no-record", action="store_true")
    p_run.add_argument("--out", default=None)
    p_run.add_argument(
        "--trust-task-code",
        action="store_true",
        help=(
            "Opt in to entrypoint_bundle task packaging (digest-pinned Python). "
            "NOT sandboxed. Prefer pettingzoo_wrappers."
        ),
    )

    p_data = sub.add_parser("data", help="Data commands")
    p_d = p_data.add_subparsers(dest="data_command", required=True)
    p_di = p_d.add_parser("inspect", help="Inspect a trajectory bundle")
    p_di.add_argument("trajectory")
    p_di.add_argument("--json", action="store_true")
    p_ds = p_d.add_parser("select", help="Slice trajectories into a lineage-preserving dataset")
    p_ds.add_argument("run_dir", help="Eval run directory (or match run directory)")
    p_ds.add_argument("--out", required=True, help="Output dataset directory")
    p_ds.add_argument("--name", default="slice")
    p_ds.add_argument("--policy", default=None)
    p_ds.add_argument("--opponent", default=None)
    p_ds.add_argument("--role", default=None)
    p_ds.add_argument("--seed", type=int, default=None)
    p_ds.add_argument("--outcome", default=None, help="win|loss|draw")
    p_ds.add_argument("--task", default=None, help="Filter by task env id")
    p_ds.add_argument("--json", action="store_true")

    p_population = sub.add_parser("population", help="Population commands (0.2)")
    p_pop = p_population.add_subparsers(dest="population_command", required=True)
    p_pc = p_pop.add_parser("create", help="Create a content-addressed population from YAML")
    p_pc.add_argument("manifest", help="population.yaml")
    p_pc.add_argument("--ref", default=None, help="Optional ref name under .rlx/refs/")
    p_pc.add_argument("--out", default=None, help="Write resolved population.yaml")
    p_pc.add_argument("--json", action="store_true")
    p_pi = p_pop.add_parser("inspect", help="Inspect a population ref or YAML")
    p_pi.add_argument("ref")
    p_pi.add_argument("--json", action="store_true")

    p_eval = sub.add_parser("eval", help="Evaluation suite commands (0.2)")
    p_e = p_eval.add_subparsers(dest="eval_command", required=True)
    p_ev = p_e.add_parser("validate", help="Validate an evaluation suite YAML")
    p_ev.add_argument("suite")
    p_ev.add_argument(
        "--population",
        action="append",
        default=[],
        help="population.yaml or digest=path mapping (repeatable)",
    )
    p_ev.add_argument("--json", action="store_true")
    p_erun = p_e.add_parser("run", help="Run a versioned evaluation suite")
    p_erun.add_argument("suite")
    p_erun.add_argument(
        "--policy",
        action="append",
        default=[],
        help="digest=path or name=path for policy bundles (repeatable)",
    )
    p_erun.add_argument(
        "--population",
        action="append",
        default=[],
        help="digest=path or path to population.yaml (repeatable)",
    )
    p_erun.add_argument("--out", default=None)
    p_erun.add_argument("--workers", type=int, default=1)
    p_erun.add_argument("--json", action="store_true")
    p_erep = p_e.add_parser("report", help="Build metrics report from an eval run directory")
    p_erep.add_argument("run_dir")
    p_erep.add_argument("--out", default=None, help="Write report.json/yaml here")
    p_erep.add_argument("--json", action="store_true")
    p_eb = p_e.add_parser("bundle", help="Build a releaseable evaluation bundle")
    p_eb.add_argument("run_dir")
    p_eb.add_argument("--out", required=True)
    p_eb.add_argument("--report", default=None, help="Optional report.json to include")

    p_release = sub.add_parser("release", help="Release bundle commands (0.2 eval-scoped)")
    p_rel = p_release.add_subparsers(dest="release_command", required=True)
    p_rb = p_rel.add_parser("build", help="Build an evaluation release bundle")
    p_rb.add_argument("--eval", required=True, dest="eval_run", help="Eval run directory")
    p_rb.add_argument("--out", required=True)
    p_rb.add_argument("--report", default=None)

    p_adapter = sub.add_parser(
        "adapter",
        help="Qualify adapter fixtures; qualification evidence is required for support claims",
    )
    p_a = p_adapter.add_subparsers(dest="adapter_command", required=True)
    p_aq = p_a.add_parser(
        "qualify",
        help="Run source, reproducibility, tamper, provenance, and handoff qualification gates",
    )
    p_aq.add_argument("fixture", help="Match YAML fixture with policy bundle assignments")
    p_aq.add_argument("--out", required=True, help="Machine-readable qualification report JSON")

    p_capture = sub.add_parser(
        "capture",
        help=(
            "Draft a contract from a source env/space (best-effort; human must confirm "
            "before publish). Does not publish bundles."
        ),
    )
    p_capture.add_argument(
        "--task",
        default=None,
        help="Task YAML or pettingzoo:// env id to observe spaces from",
    )
    p_capture.add_argument("--agent", default=None, help="Agent id to capture (default: first)")
    p_capture.add_argument("--out", default=None, help="Write draft JSON to this path")
    p_capture.add_argument("--json", action="store_true", help="Print draft as JSON")

    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            return cmd_init(args)
        if args.command == "inspect":
            return cmd_inspect(args)
        if args.command == "check":
            return cmd_check(args)
        if args.command == "policy":
            if args.policy_command == "export":
                return cmd_policy_export(args)
            if args.policy_command == "verify":
                return cmd_policy_verify(args)
        if args.command == "match" and args.match_command == "run":
            return cmd_match_run(args)
        if args.command == "data":
            if args.data_command == "inspect":
                return cmd_data_inspect(args)
            if args.data_command == "select":
                return cmd_data_select(args)
        if args.command == "population":
            if args.population_command == "create":
                return cmd_population_create(args)
            if args.population_command == "inspect":
                return cmd_population_inspect(args)
        if args.command == "eval":
            if args.eval_command == "validate":
                return cmd_eval_validate(args)
            if args.eval_command == "run":
                return cmd_eval_run(args)
            if args.eval_command == "report":
                return cmd_eval_report(args)
            if args.eval_command == "bundle":
                return cmd_eval_bundle(args)
        if args.command == "release" and args.release_command == "build":
            return cmd_release_build(args)
        if args.command == "adapter" and args.adapter_command == "qualify":
            return cmd_adapter_qualify(args)
        if args.command == "capture":
            return cmd_capture(args)
    except Exception as e:  # noqa: BLE001
        print(f"error: {e}", file=sys.stderr)
        return 1
    parser.error(f"unknown command {args.command}")
    return 2


def cmd_init(args: argparse.Namespace) -> int:
    from rlx.core.store import LocalStore

    store = LocalStore(args.path)
    path = store.init(force=args.force)
    print(f"Initialized RLX workspace at {path}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    from rlx.core.manifests import load_manifest, resolve_artifact_path
    from rlx.core.sdk import Policy

    artifact = Path(args.artifact)
    if artifact.is_dir() and (artifact / "bundle.yaml").exists():
        from rlx.runtime.trajectory import inspect_trajectory

        info = inspect_trajectory(artifact)
        _print(info, as_json=args.json)
        return 0

    path = resolve_artifact_path(artifact)
    data = load_manifest(path)
    schema = data.get("schema", "")
    if schema.startswith("rlx.policy"):
        pol = Policy.load(artifact)
        info = {
            "kind": "policy",
            "name": pol.name,
            "digest": pol.digest,
            "roles": pol.manifest.get("roles"),
            "runtime": pol.manifest.get("runtime"),
            "observation": pol.manifest.get("observation"),
            "action": pol.manifest.get("action"),
            "state": pol.manifest.get("state"),
            "inference": pol.manifest.get("inference"),
            "preprocessing": pol.manifest.get("preprocessing"),
            "payloads": pol.manifest.get("payloads"),
            "lineage": pol.manifest.get("lineage"),
            "conformance": pol.manifest.get("conformance"),
        }
    elif schema.startswith("rlx.match"):
        info = {"kind": "match", **data}
    else:
        info = {"kind": "manifest", **data}
    _print(info, as_json=args.json)
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    from rlx.core.sdk import Policy, check

    config = _parse_json_arg(args.config) if getattr(args, "config", None) else None
    if config is not None and not isinstance(config, dict):
        raise SystemExit("--config must be a JSON/YAML object")
    task = _load_task_arg(args.task, config=config)
    policy = Policy.load(args.policy)
    report = check(task, policy.as_role(args.role), action_mode=args.action_mode)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.format_human())
    return 0 if report.ok else 1


def cmd_policy_export(args: argparse.Namespace) -> int:
    from rlx.adapters.policy_custom_torch import (
        export_from_checkpoint,
        export_module_from_checkpoint,
    )
    from rlx.core.manifests import load_manifest

    if args.adapter != "custom-pytorch":
        raise SystemExit(f"unsupported adapter: {args.adapter}")

    spec: dict[str, Any] = {}
    if args.spec:
        spec = load_manifest(args.spec)

    obs = _parse_json_arg(args.obs) or spec.get("observation")
    action = _parse_json_arg(args.action) or spec.get("action")
    if obs is None or action is None:
        raise SystemExit("export requires --spec or --obs/--action")

    action = dict(action)
    action["masks"] = args.masks if args.masks else action.get("masks", "none")

    preprocess = dict(spec.get("preprocessing") or {})
    if args.preprocess_mean is not None:
        preprocess["mean"] = json.loads(args.preprocess_mean)
    if args.preprocess_std is not None:
        preprocess["std"] = json.loads(args.preprocess_std)

    if getattr(args, "trusted_source_py", None):
        if not getattr(args, "trust_source", False):
            raise SystemExit(
                "trusted_source export requires --trust-source (NOT sandboxed; "
                "prefer TorchScript --module)"
            )
        if not args.source:
            raise SystemExit("trusted_source export requires --source weights checkpoint")
        cases_raw = _parse_json_arg(args.reference_cases)
        if cases_raw is None:
            raise SystemExit("trusted_source export requires --reference-cases")
        if isinstance(cases_raw, dict) and "cases" in cases_raw:
            cases = list(cases_raw["cases"])
        elif isinstance(cases_raw, list):
            cases = cases_raw
        else:
            raise SystemExit("--reference-cases must be a JSON list or {cases: [...]}")
        from rlx.adapters.policy_custom_torch import (
            _extract_state_dict,
            load_checkpoint_file,
        )
        from rlx.core.identity import digest_uri, sha256_file
        from rlx.plugins.payloads import export_trusted_source_bundle

        obj = load_checkpoint_file(
            args.source,
            allow_unsafe_checkpoint=bool(getattr(args, "allow_unsafe_checkpoint", False)),
        )
        state = _extract_state_dict(
            obj,
            Path(args.source),
            prefer_ema=not bool(getattr(args, "prefer_base_weights", False)),
        )
        io = dict(spec.get("architecture", {}).get("io") or spec.get("io") or {})
        out = export_trusted_source_bundle(
            out_dir=args.out,
            name=args.name or f"{args.role}-policy",
            roles=[args.role],
            source_py=args.trusted_source_py,
            state_dict=state,
            observation=obs,
            action=action,
            factory=args.trusted_source_factory,
            io=io,
            preprocessing=preprocess or None,
            reference_cases=cases,
            trust_source=True,
            lineage={
                "export_path": "trusted_source",
                "source_checkpoint": Path(args.source).name,
                "checkpoint_digest": digest_uri(sha256_file(Path(args.source))),
            },
        )
        print(f"Exported trusted_source policy bundle to {out}")
        print(
            "WARNING: trusted_source is NOT sandboxed. Prefer TorchScript for new exports.",
            file=sys.stderr,
        )
        return 0

    if args.module:
        cases_raw = _parse_json_arg(args.reference_cases)
        if cases_raw is None:
            raise SystemExit("BYO --module export requires --reference-cases")
        if isinstance(cases_raw, dict) and "cases" in cases_raw:
            cases = list(cases_raw["cases"])
        elif isinstance(cases_raw, list):
            cases = cases_raw
        else:
            raise SystemExit("--reference-cases must be a JSON list or {cases: [...]}")
        module_args = _parse_json_arg(args.module_args) or spec.get("module_args") or {}
        if module_args is not None and not isinstance(module_args, dict):
            raise SystemExit("--module-args must be a JSON/YAML object")
        io = dict(spec.get("architecture", {}).get("io") or spec.get("io") or {})
        if args.recurrent:
            io["recurrent"] = True
        out = export_module_from_checkpoint(
            module_ref=args.module,
            out_dir=args.out,
            role=args.role,
            observation=obs,
            action=action,
            source=args.source,
            name=args.name,
            io=io,
            preprocessing=preprocess or None,
            reference_cases=cases,
            module_args=module_args or None,
            allow_trace=bool(args.allow_trace),
            allow_unsafe_checkpoint=bool(getattr(args, "allow_unsafe_checkpoint", False)),
            prefer_ema=not bool(getattr(args, "prefer_base_weights", False)),
            source_revision=args.source_revision or spec.get("source_revision"),
            wrappers_identity=args.wrappers_identity or spec.get("wrappers_identity"),
        )
        print(f"Exported BYO TorchScript policy bundle to {out}")
        return 0

    if not args.source:
        raise SystemExit("template export requires --source checkpoint")
    arch = _parse_json_arg(args.arch) or spec.get("architecture")
    if arch is None:
        raise SystemExit("template export requires --spec or --arch/--obs/--action")

    out = export_from_checkpoint(
        source=args.source,
        out=args.out,
        role=args.role,
        name=args.name,
        architecture=arch,
        observation=obs,
        action=action,
        preprocessing=preprocess or None,
        recurrent=args.recurrent or bool(spec.get("state", {}).get("recurrent")),
        allow_unsafe_checkpoint=bool(getattr(args, "allow_unsafe_checkpoint", False)),
        prefer_ema=not bool(getattr(args, "prefer_base_weights", False)),
    )
    print(f"Exported policy bundle to {out}")
    return 0


def cmd_policy_verify(args: argparse.Namespace) -> int:
    from rlx.adapters.policy_custom_torch import verify_bundle_self

    result = verify_bundle_self(
        args.policy,
        args.source_test,
        allow_self_consistency=args.allow_self_consistency,
        trust_source=bool(getattr(args, "trust_source", False)),
    )
    if result.get("warning"):
        print(f"WARNING: {result['warning']}", file=sys.stderr)
    print(json.dumps(result, indent=2))
    return 0


def cmd_match_run(args: argparse.Namespace) -> int:
    from rlx.core.manifests import expand_seeds, load_manifest, validate_match_manifest
    from rlx.core.sdk import Match, Policy, Task

    path = Path(args.match_manifest)
    data = validate_match_manifest(load_manifest(path))
    task_spec = dict(data["task"]) if isinstance(data["task"], dict) else {"env": data["task"]}
    if getattr(args, "trust_task_code", False):
        task_spec["trust_task_code"] = True
        packaging = dict(task_spec.get("packaging") or {})
        if packaging:
            packaging["trust_task_code"] = True
            task_spec["packaging"] = packaging
    task = Task.load(task_spec)
    base = path.parent
    assignments = {}
    for role, pref in data["assignments"].items():
        ppath = Path(pref)
        if not ppath.is_absolute():
            ppath = (base / ppath).resolve()
        assignments[role] = Policy.load(ppath)
    match = Match(
        task=task,
        assignments=assignments,
        action_mode=data.get("action_mode", "deterministic"),
        failure_policy=data.get("failure_policy"),
    )
    record = not args.no_record
    out = args.out
    result = match.run(seeds=expand_seeds(data["seeds"]), record=record, out=out)
    print(f"Match complete: {result['run_id']}")
    print(f"  completed={result['outcome']['episodes_completed']} failures={result['outcome']['failure_count']}")
    if out:
        print(f"  output={out}")
    return 0 if result["outcome"]["failure_count"] == 0 else 1


def cmd_data_inspect(args: argparse.Namespace) -> int:
    from rlx.runtime.trajectory import inspect_trajectory

    info = inspect_trajectory(args.trajectory)
    _print(info, as_json=args.json)
    return 0 if info.get("completeness", {}).get("ok", True) else 1


def cmd_data_select(args: argparse.Namespace) -> int:
    from rlx.core.dataset import select_episodes

    query: dict[str, Any] = {}
    if args.policy:
        query["policy"] = args.policy
    if args.opponent:
        query["opponent"] = args.opponent
    if args.role:
        query["role"] = args.role
    if args.seed is not None:
        query["seed"] = args.seed
    if args.outcome:
        query["outcome"] = args.outcome
    if args.task:
        query["task"] = args.task
    dataset = select_episodes(
        source_runs=[args.run_dir],
        query=query,
        name=args.name,
        out_dir=args.out,
    )
    _print(dataset, as_json=args.json)
    return 0


def cmd_population_create(args: argparse.Namespace) -> int:
    from rlx.core.population import create_population_from_yaml, write_population_yaml
    from rlx.core.store import LocalStore

    store = LocalStore.find()
    pop = create_population_from_yaml(args.manifest, store=store, ref=args.ref)
    if args.out:
        write_population_yaml(pop, args.out)
    _print(pop, as_json=args.json)
    return 0


def cmd_population_inspect(args: argparse.Namespace) -> int:
    from rlx.core.manifests import population_content_digest
    from rlx.core.population import load_population
    from rlx.core.store import LocalStore

    store = LocalStore.find()
    pop = load_population(args.ref, store=store)
    pop = {**pop, "digest": pop.get("digest") or population_content_digest(pop)}
    _print(pop, as_json=args.json)
    return 0


def cmd_eval_validate(args: argparse.Namespace) -> int:
    from rlx.core.manifests import evaluation_content_digest
    from rlx.core.population import load_population
    from rlx.core.store import LocalStore
    from rlx.runtime.evaluation import load_evaluation, validate_evaluation

    suite_path = Path(args.suite)
    suite = load_evaluation(suite_path)
    base = suite_path.parent.resolve()
    populations: dict[str, dict] = {}
    store = None
    try:
        store = LocalStore.find()
    except Exception:
        store = None
    for item in args.population:
        if "=" in item:
            key, path = item.split("=", 1)
            populations[key] = load_population(path, store=store)
        else:
            pop = load_population(item, store=store)
            populations[pop.get("digest", item)] = pop
            populations[item] = pop
    for role, spec in suite["assignments"].items():
        if isinstance(spec, dict) and spec.get("kind") in {"population", "crossplay"}:
            pref = str(spec["population"])
            if pref in populations:
                continue
            cand = Path(pref) if Path(pref).is_absolute() else (base / pref)
            if cand.exists():
                pop = load_population(cand, store=store)
                populations[pref] = pop
                populations[str(cand)] = pop
                populations[pop["digest"]] = pop
    validate_evaluation(suite, populations=populations)
    print("ok")
    if args.json:
        _print({"evaluation_digest": evaluation_content_digest(suite)}, as_json=True)
    return 0


def cmd_eval_run(args: argparse.Namespace) -> int:
    from pathlib import Path

    from rlx.core.population import load_population
    from rlx.core.store import LocalStore
    from rlx.runtime.evaluation import load_evaluation, run_evaluation

    suite_path = Path(args.suite)
    suite = load_evaluation(suite_path)
    base = suite_path.parent.resolve()
    policy_index: dict[str, Path] = {}
    for item in args.policy:
        if "=" not in item:
            raise SystemExit("--policy must be digest=path or name=path")
        key, path = item.split("=", 1)
        policy_index[key] = Path(path)
        from rlx.core.sdk import Policy

        pol = Policy.load(path)
        policy_index[pol.digest] = Path(path)

    populations: dict[str, dict] = {}
    store = None
    try:
        store = LocalStore.find()
    except Exception:
        store = None
    for item in args.population:
        if "=" in item:
            key, path = item.split("=", 1)
            pop = load_population(path, store=store)
            populations[key] = pop
            populations[pop.get("digest", key)] = pop
        else:
            pop = load_population(item, store=store)
            populations[item] = pop
            if "digest" in pop:
                populations[pop["digest"]] = pop

    # Resolve suite-relative population paths and rewrite assignments to digests.
    assigns = {}
    for role, spec in suite["assignments"].items():
        if isinstance(spec, dict) and spec.get("kind") in {"population", "crossplay"}:
            pref = str(spec["population"])
            pop = populations.get(pref)
            if pop is None:
                cand = Path(pref) if Path(pref).is_absolute() else (base / pref)
                if cand.exists():
                    pop = load_population(cand, store=store)
                    populations[pref] = pop
                    populations[str(cand)] = pop
                    populations[pop["digest"]] = pop
            if pop is None:
                raise SystemExit(f"population {pref!r} not provided for eval run")
            assigns[role] = {**spec, "population": pop["digest"]}
            populations[pop["digest"]] = pop
        else:
            assigns[role] = spec
    suite = {**suite, "assignments": assigns}

    result = run_evaluation(
        suite,
        policy_index=policy_index,
        populations=populations,
        store=store,
        out_dir=Path(args.out) if args.out else None,
        workers=args.workers,
    )
    summary = {
        "run_id": result["run_id"],
        "run_dir": result["run_dir"],
        "evaluation_digest": result["evaluation_digest"],
        "cells": len(result["cells"]),
        "sampling_ledger": result["sampling_ledger"],
    }
    _print(summary, as_json=args.json)
    return 0


def cmd_eval_report(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from rlx.core.manifests import dump_json, dump_yaml
    from rlx.runtime.evaluation import build_eval_report

    run_dir = Path(args.run_dir)
    eval_run = json.loads((run_dir / "eval_run.json").read_text(encoding="utf-8"))
    # Reload rich cell results by scanning cell dirs.
    cell_results = []
    for cell in eval_run.get("cells") or []:
        cell_dir = run_dir / cell["cell_id"]
        episodes = []
        evidence = []
        traj = cell_dir / "trajectories"
        if traj.is_dir():
            for ep_path in sorted(traj.glob("episode_*.json")):
                ep = json.loads(ep_path.read_text(encoding="utf-8"))
                episodes.append(
                    {
                        "path": str(ep_path),
                        "seed": ep.get("seed"),
                        "returns": ep.get("returns")
                        or {
                            a: sum(float(s.get("rewards", {}).get(a, 0.0)) for s in ep.get("steps", []))
                            for a in (ep.get("steps", [{}])[0].get("rewards", {}) if ep.get("steps") else {})
                        },
                        "outcomes": ep.get("outcomes") or {},
                    }
                )
                evidence.append(str(ep_path))
        cell_results.append({**cell, "episodes": episodes, "evidence_refs": evidence})
    # Attach suite metrics list if present beside run.
    suite_path = run_dir / "suite.yaml"
    suite = {}
    if suite_path.exists():
        from rlx.core.manifests import load_manifest

        suite = load_manifest(suite_path)
    eval_run["cell_results"] = cell_results
    eval_run["suite"] = suite or {
        "metrics": ["payoff_matrix", "mean_return", "win_rate"],
    }
    report = build_eval_report(eval_run)
    out = Path(args.out) if args.out else run_dir
    out.mkdir(parents=True, exist_ok=True)
    dump_json(report, out / "report.json")
    dump_yaml(report, out / "report.yaml")
    _print(report, as_json=args.json)
    return 0


def cmd_eval_bundle(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from rlx.core.eval_bundle import build_eval_bundle

    report = None
    if args.report:
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    bundle = build_eval_bundle(eval_run_dir=args.run_dir, report=report, out_dir=args.out)
    print(json.dumps({"digest": bundle["digest"], "out": args.out}, indent=2))
    return 0


def cmd_release_build(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from rlx.core.eval_bundle import build_eval_bundle

    report = None
    if args.report:
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    bundle = build_eval_bundle(eval_run_dir=args.eval_run, report=report, out_dir=args.out)
    print(json.dumps({"digest": bundle["digest"], "out": args.out}, indent=2))
    return 0


def cmd_adapter_qualify(args: argparse.Namespace) -> int:
    from rlx.conformance.qualification import qualify_adapter_fixture

    report = qualify_adapter_fixture(args.fixture, report_path=args.out)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


def cmd_capture(args: argparse.Namespace) -> int:
    from rlx.adapters.task_pettingzoo.adapter import make_env
    from rlx.core.capture import capture_draft_from_env

    if not args.task:
        raise SystemExit("capture requires --task")
    task = _load_task_arg(args.task)
    env = make_env(task.spec, trust_task_code=bool(task.spec.get("trust_task_code")))
    try:
        draft = capture_draft_from_env(env, agent=args.agent)
    finally:
        env.close()
    if args.out:
        Path(args.out).write_text(json.dumps(draft, indent=2), encoding="utf-8")
        print(f"Wrote draft contract to {args.out}")
    _print(draft, as_json=True)
    return 0


def _load_task_arg(task_arg: str, *, config: dict[str, Any] | None = None) -> Any:
    """Load a task from env id, task YAML, or match.yaml (using its nested task).

    Optional ``config`` is merged into ``task.spec["config"]`` so ``rlx check``
    validates against the configured spaces (not just env defaults).
    """
    from rlx.core.manifests import MATCH_SCHEMA, load_manifest
    from rlx.core.sdk import Task

    path = Path(task_arg)
    if path.exists():
        data = load_manifest(path)
        if data.get("schema") == MATCH_SCHEMA or (
            isinstance(data.get("task"), (dict, str)) and "assignments" in data
        ):
            task_ref = data["task"]
            task = Task.load(task_ref) if not isinstance(task_ref, Task) else task_ref
        else:
            task = Task(data) if "adapter" in data or "env" in data else Task.load(data)
    elif task_arg.startswith("pettingzoo://"):
        task = Task.load(task_arg)
    else:
        # Treat as env id
        task = Task.load({"adapter": "pettingzoo-parallel", "env": task_arg})

    if config:
        merged = dict(task.spec.get("config") or {})
        merged.update(config)
        task.spec["config"] = merged
    return task


def _parse_json_arg(value: str | None) -> Any:
    if value is None:
        return None
    path = Path(value)
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if path.suffix in {".yaml", ".yml"}:
            import yaml

            return yaml.safe_load(text)
        return json.loads(text)
    return json.loads(value)


def _print(data: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(data, indent=2, default=str))
        return
    import yaml

    print(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


if __name__ == "__main__":
    raise SystemExit(main())
