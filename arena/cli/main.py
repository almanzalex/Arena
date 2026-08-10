"""Arena command-line interface."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from arena.core.errors import (
    ArenaError,
    CliUsageError,
    diagnostic_from_exception,
    exit_code_for_exception,
    redact,
)


class _ArenaArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CliUsageError(
            message,
            repair=f"Run `{self.prog} --help` and use the documented argument grammar.",
            context={"program": self.prog},
        )


def _global_options(argv: list[str]) -> tuple[list[str], bool, bool, bool]:
    cleaned: list[str] = []
    as_json = False
    debug = False
    show_version = False
    for token in argv:
        if token == "--json":
            as_json = True
        elif token == "--debug":
            debug = True
        elif token == "--version":
            show_version = True
        else:
            cleaned.append(token)
    return cleaned, as_json, debug, show_version


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    clean_argv, global_json, debug, show_version = _global_options(raw_argv)
    nested_commands = {
        "attest",
        "data",
        "demo",
        "eval",
        "match",
        "policy",
        "population",
        "release",
        "schema",
        "store",
        "task",
    }
    command_parts = clean_argv[:1]
    if (
        command_parts
        and command_parts[0] in nested_commands
        and len(clean_argv) > 1
        and not clean_argv[1].startswith("-")
    ):
        command_parts.append(clean_argv[1])
    command_label = " ".join(command_parts) or "arena"
    if show_version:
        from arena import __version__

        if global_json:
            print(
                json.dumps(
                    {
                        "schema": "arena.cli-result/v1",
                        "ok": True,
                        "command": "version",
                        "data": {"version": __version__},
                        "warnings": [],
                    },
                    ensure_ascii=False,
                )
            )
        else:
            print(f"arena {__version__}")
        return 0

    parser = _ArenaArgumentParser(
        prog="arena",
        description=(
            "Arena 1.0 — verifiable policy/evaluation handoff across native and "
            "qualified external runtimes, providers, and stores"
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit a versioned JSON result")
    parser.add_argument("--debug", action="store_true", help="Emit a redacted traceback")
    parser.add_argument("--version", action="store_true", help="Show the installed Arena version")
    sub = parser.add_subparsers(
        dest="command",
        required=True,
        parser_class=_ArenaArgumentParser,
    )

    p_doctor = sub.add_parser(
        "doctor",
        help="Report release support, local dependencies, and repairs without authenticating",
    )
    p_doctor.add_argument("--capability", default=None)

    p_demo = sub.add_parser("demo", help="Packaged source-free value demonstrations")
    p_dem = p_demo.add_subparsers(dest="demo_command", required=True)
    p_dh = p_dem.add_parser("handoff", help="Build, mirror, pull, and verify a policy locally")
    p_dh.add_argument("--out", default="./arena-demo")
    p_dm = p_dem.add_parser(
        "multiagent",
        help="Export portable RPS policies and run PettingZoo classic/rps_v2 matches",
    )
    p_dm.add_argument("--out", default="./arena-ma-demo")
    p_dm.add_argument("--json", action="store_true")

    p_schema = sub.add_parser("schema", help="Inspect the installed compatibility registry")
    p_sch = p_schema.add_subparsers(dest="schema_command", required=True)
    p_sch.add_parser("list", help="List stable and legacy-frozen manifest schemas")

    p_init = sub.add_parser("init", help="Create a local .arena workspace")
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

    p_task = sub.add_parser("task", help="Import external tasks and verify trace semantics")
    p_t = p_task.add_subparsers(dest="task_command", required=True)
    p_ti = p_t.add_parser("import", help="Import and identity-pin a registered external task")
    p_ti.add_argument("source", help="openenv://host/env or a qualified openspiel:// game")
    p_ti.add_argument("--name", required=True)
    p_ti.add_argument("--out", default=None, help="Task YAML output (defaults from --name)")
    p_ti.add_argument("--contract", default=None, help="Arena role-space contract YAML")
    p_ti.add_argument("--source-revision", default=None)
    p_ti.add_argument("--timeout", type=float, default=10.0)
    p_ti.add_argument("--json", action="store_true")
    p_tv = p_t.add_parser(
        "verify-equivalence",
        help="Compare seeded observations/actions/rewards/done semantics from a trace suite",
    )
    p_tv.add_argument("left")
    p_tv.add_argument("right", nargs="?", default=None)
    p_tv.add_argument("--trace-suite", required=True)
    p_tv.add_argument("--out", default=None)
    p_tv.add_argument("--json", action="store_true")

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
    p_dm = p_d.add_parser(
        "materialize", help="Copy a selected dataset into a portable verified directory"
    )
    p_dm.add_argument("dataset", help="dataset.yaml produced by data select")
    p_dm.add_argument("--out", required=True)
    p_dm.add_argument(
        "--split",
        action="append",
        default=[],
        metavar="NAME=WEIGHT",
        help="Deterministic digest-bucket split; repeat for train/validation/test",
    )
    p_dm.add_argument("--split-seed", type=int, default=0)
    p_dm.add_argument("--json", action="store_true")

    p_train = sub.add_parser(
        "train", help="Run a reproducible training recipe over an Arena dataset"
    )
    p_train.add_argument("recipe", help="arena.train/v1 YAML recipe")
    p_train.add_argument("--out", required=True, help="Training run + policy output directory")
    p_train.add_argument("--json", action="store_true")

    p_population = sub.add_parser("population", help="Population commands (0.2)")
    p_pop = p_population.add_subparsers(dest="population_command", required=True)
    p_pc = p_pop.add_parser("create", help="Create a content-addressed population from YAML")
    p_pc.add_argument("manifest", help="population.yaml")
    p_pc.add_argument("--ref", default=None, help="Optional ref name under .arena/refs/")
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
    p_erun.add_argument("--provider", default=None, help="Override native|gimitest provider")
    p_erun.add_argument("--json", action="store_true")
    p_erep = p_e.add_parser("report", help="Build metrics report from an eval run directory")
    p_erep.add_argument("run_dir")
    p_erep.add_argument("--out", default=None, help="Write report.json/yaml here")
    p_erep.add_argument("--json", action="store_true")
    p_eb = p_e.add_parser("bundle", help="Build a releaseable evaluation bundle")
    p_eb.add_argument("run_dir")
    p_eb.add_argument("--out", required=True)
    p_eb.add_argument("--report", default=None, help="Optional report.json to include")
    p_ec = p_e.add_parser(
        "compare",
        help=(
            "Compare two eval reports/bundles and fail if suite digests, "
            "policy digests, or seed protocols differ"
        ),
    )
    p_ec.add_argument("left", help="Left report.json, eval_run.json, or bundle directory")
    p_ec.add_argument("right", help="Right report.json, eval_run.json, or bundle directory")
    p_ec.add_argument("--json", action="store_true")

    p_release = sub.add_parser(
        "release",
        help="Evaluation bundles and signed 1.0 release-evidence commands",
    )
    p_rel = p_release.add_subparsers(dest="release_command", required=True)
    p_rb = p_rel.add_parser(
        "build",
        help="Build an evaluation release bundle (locked eval digests/artifacts)",
    )
    p_rb.add_argument("--eval", required=True, dest="eval_run", help="Eval run directory")
    p_rb.add_argument("--out", required=True)
    p_rb.add_argument("--report", default=None)
    p_ra = p_rel.add_parser(
        "assemble",
        help=(
            "Assemble signed-ready release-evidence index from R-01..R-14 gates "
            "and exact release artifacts (not eval-bundle build)"
        ),
    )
    p_ra.add_argument("--release", required=True)
    p_ra.add_argument("--tag", required=True)
    p_ra.add_argument("--commit", required=True)
    p_ra.add_argument(
        "--gate",
        action="append",
        default=[],
        metavar="R-NN=PATH",
        help="Repeat once for every mandatory release gate",
    )
    p_ra.add_argument("--artifact", action="append", default=[], required=True)
    p_ra.add_argument(
        "--eval-bundle",
        action="append",
        default=[],
        dest="eval_bundle",
        metavar="PATH",
        help=(
            "Optional eval bundle directory or bundle.json; records evaluation_digest "
            "when present (repeatable)"
        ),
    )
    p_ra.add_argument("--out", required=True)
    p_rs = p_rel.add_parser("sign", help="Sign a release evidence index or current ledger")
    p_rs.add_argument("document")
    p_rs.add_argument("--key", required=True)
    p_rs.add_argument("--out", required=True)
    p_rv = p_rel.add_parser("verify", help="Verify a signed Arena release-evidence index")
    p_rv.add_argument("evidence_index")
    p_rv.add_argument("--signature", required=True)
    p_rv.add_argument("--key", required=True)
    release_mode = p_rv.add_mutually_exclusive_group(required=True)
    release_mode.add_argument("--at-release", action="store_true")
    release_mode.add_argument("--current-ledger", default=None)
    p_rv.add_argument("--ledger-signature", default=None)
    p_rv.add_argument("--ledger-key", default=None)

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
    p_aq.add_argument("--peer", default=None, help="Peer task YAML for equivalence qualification")
    p_aq.add_argument("--trace-suite", default=None, help="Trace suite for task qualification")

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

    p_attest = sub.add_parser(
        "attest",
        help=(
            "Detached lab attestations with user-owned Ed25519 keys "
            "(no CA, no publisher account)"
        ),
    )
    p_at = p_attest.add_subparsers(dest="attest_command", required=True)
    p_ak = p_at.add_parser(
        "keygen",
        help="Generate a user-owned Ed25519 keypair (private 0600, public 0644)",
    )
    p_ak.add_argument(
        "--private",
        required=True,
        metavar="PATH",
        help="Write PKCS8 private key PEM here (refuses overwrite)",
    )
    p_ak.add_argument(
        "--public",
        required=True,
        metavar="PATH",
        help="Write SPKI public key PEM here (refuses overwrite)",
    )
    p_ak.add_argument("--json", action="store_true")
    p_as = p_at.add_parser(
        "sign",
        help="Sign an artifact identity into a detached attestation JSON",
    )
    p_as.add_argument(
        "source",
        help="Local artifact path (file or directory) whose identity is signed",
    )
    p_as.add_argument(
        "--key",
        required=True,
        metavar="PRIVATE.pem",
        help="Ed25519 private key PEM (lab-supplied; unencrypted PKCS8)",
    )
    p_as.add_argument(
        "--issuer",
        required=True,
        metavar="NAME",
        help="Non-empty issuer label recorded in the attestation predicate",
    )
    p_as.add_argument(
        "--out",
        required=True,
        metavar="ATTEST.json",
        help="Detached attestation output path (refuses overwrite)",
    )
    p_as.add_argument("--json", action="store_true")
    p_av = p_at.add_parser(
        "verify",
        help="Rehash artifact identity and verify the detached Ed25519 attestation",
    )
    p_av.add_argument(
        "source",
        help="Local artifact path to rehash and compare against the attestation subject",
    )
    p_av.add_argument(
        "attestation",
        metavar="ATTEST.json",
        help="Detached attestation JSON produced by `arena attest sign`",
    )
    p_av.add_argument(
        "--key",
        required=True,
        metavar="PUBLIC.pem",
        help="Trusted Ed25519 public key PEM (independently supplied; not a CA)",
    )
    p_av.add_argument("--json", action="store_true")

    p_push = sub.add_parser("push", help="Mirror an Arena artifact without changing identity")
    p_push.add_argument("source", help="Artifact path, object digest, or local ref")
    p_push.add_argument(
        "destination",
        help=(
            "file://, hf://, oci://, wandb://, or mlflow:// store URI "
            "(push returns the same URI with #sha256:… identity fragment)"
        ),
    )
    p_push.add_argument(
        "--verify",
        action="store_true",
        help="After push, re-read the store and check blob digests match the descriptor",
    )
    p_push.add_argument("--json", action="store_true")

    p_pull = sub.add_parser("pull", help="Restore a mirrored Arena artifact")
    p_pull.add_argument(
        "source",
        help="Artifact URI from `arena push` (must include #sha256:… identity fragment)",
    )
    p_pull.add_argument("--out", default=None, help="Restore directory (defaults from identity)")
    p_pull.add_argument(
        "--verify",
        action="store_true",
        help="After restore, recompute artifact identity and require it matches the URI",
    )
    p_pull.add_argument("--json", action="store_true")

    p_store = sub.add_parser(
        "store",
        help="External-store qualification commands",
    )
    p_s = p_store.add_subparsers(dest="store_command", required=True)
    p_sq = p_s.add_parser(
        "qualify",
        help="Produce machine-readable verified push/pull identity evidence",
    )
    p_sq.add_argument("source", help="Local Arena artifact")
    p_sq.add_argument("destination", help="Registered store URI")
    p_sq.add_argument("--out", required=True, help="Qualification report JSON")

    parse_output = io.StringIO()
    try:
        if global_json:
            with contextlib.redirect_stdout(parse_output):
                args = parser.parse_args(clean_argv)
        else:
            args = parser.parse_args(clean_argv)
        args.json = bool(global_json)
        args.debug = bool(debug)
        if global_json:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result_code = _dispatch(args)
            rendered = output.getvalue().strip()
            try:
                data = json.loads(rendered) if rendered else {}
            except json.JSONDecodeError:
                data = {"output": rendered}
            if result_code != 0:
                diagnostic = {
                    "schema": "arena.diagnostic/v1",
                    "ok": False,
                    "code": "COMMAND_FAILED",
                    "category": "schema_compatibility",
                    "message": "The command completed with a non-success result.",
                    "cause": data,
                    "repair": "Inspect the structured command result and apply its suggested repairs.",
                    "docs_url": "https://github.com/almanzalex/Arena/blob/main/docs/errors.md#command_failed",
                    "context": {"result": data},
                    "command": command_label,
                }
                print(json.dumps(diagnostic, ensure_ascii=False, default=str))
                return result_code if result_code in {2, 3, 4, 5, 6, 70} else 3
            envelope = {
                "schema": "arena.cli-result/v1",
                "ok": True,
                "command": command_label,
                "data": data,
                "warnings": [],
            }
            # Preserve the pre-1.0 top-level result keys as additive aliases while
            # the versioned envelope becomes the stable automation contract.
            if isinstance(data, dict):
                for key, value in data.items():
                    envelope.setdefault(key, value)
            print(json.dumps(envelope, ensure_ascii=False, default=str))
            return 0
        return _dispatch(args)
    except SystemExit as exc:
        if exc.code in {0, None}:
            if global_json:
                print(
                    json.dumps(
                        {
                            "schema": "arena.cli-result/v1",
                            "ok": True,
                            "command": command_label,
                            "data": {"help": parse_output.getvalue()},
                            "warnings": [],
                        },
                        ensure_ascii=False,
                    )
                )
            return 0
        if isinstance(exc.code, str):
            error: BaseException = CliUsageError(
                exc.code,
                repair=f"Run `arena {command_label} --help` for valid usage.",
            )
        else:
            error = CliUsageError(
                f"command exited before completion (status {exc.code})",
                repair=f"Run `arena {command_label} --help` for valid usage.",
            )
        return _emit_error(error, as_json=global_json, debug=debug, command=command_label)
    except Exception as e:  # noqa: BLE001
        return _emit_error(e, as_json=global_json, debug=debug, command=command_label)


def _emit_error(
    error: BaseException,
    *,
    as_json: bool,
    debug: bool,
    command: str,
) -> int:
    diagnostic = diagnostic_from_exception(error, command=command, debug=debug)
    if as_json:
        print(json.dumps(diagnostic, ensure_ascii=False, default=str))
    else:
        print(
            f"error [{diagnostic['code']}]: {diagnostic['message']}",
            file=sys.stderr,
        )
        if diagnostic.get("cause"):
            print(f"cause: {diagnostic['cause']}", file=sys.stderr)
        if diagnostic.get("repair"):
            print(f"repair: {diagnostic['repair']}", file=sys.stderr)
        if debug:
            print(str(redact(traceback.format_exc())), file=sys.stderr)
    return exit_code_for_exception(error)


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "doctor":
        return cmd_doctor(args)
    if args.command == "demo" and args.demo_command == "handoff":
        return cmd_demo_handoff(args)
    if args.command == "demo" and args.demo_command == "multiagent":
        return cmd_demo_multiagent(args)
    if args.command == "schema" and args.schema_command == "list":
        return cmd_schema_list(args)
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
    if args.command == "task":
        if args.task_command == "import":
            return cmd_task_import(args)
        if args.task_command == "verify-equivalence":
            return cmd_task_verify_equivalence(args)
    if args.command == "data":
        if args.data_command == "inspect":
            return cmd_data_inspect(args)
        if args.data_command == "select":
            return cmd_data_select(args)
        if args.data_command == "materialize":
            return cmd_data_materialize(args)
    if args.command == "train":
        return cmd_train(args)
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
        if args.eval_command == "compare":
            return cmd_eval_compare(args)
    if args.command == "release":
        if args.release_command == "build":
            return cmd_release_build(args)
        if args.release_command == "assemble":
            return cmd_release_assemble(args)
        if args.release_command == "sign":
            return cmd_release_sign(args)
        if args.release_command == "verify":
            return cmd_release_verify(args)
    if args.command == "adapter" and args.adapter_command == "qualify":
        return cmd_adapter_qualify(args)
    if args.command == "capture":
        return cmd_capture(args)
    if args.command == "attest":
        return cmd_attest(args)
    if args.command == "push":
        return cmd_push(args)
    if args.command == "pull":
        return cmd_pull(args)
    if args.command == "store" and args.store_command == "qualify":
        return cmd_store_qualify(args)
    raise CliUsageError(f"unknown command {args.command!r}")


def cmd_doctor(args: argparse.Namespace) -> int:
    from arena.core.support import doctor_report, format_doctor_human

    report = doctor_report(args.capability)
    if args.json:
        _print(report, as_json=True)
    else:
        print(format_doctor_human(report), end="")
    return 0


def cmd_schema_list(args: argparse.Namespace) -> int:
    from arena.core.support import load_schema_registry

    _print(load_schema_registry(), as_json=bool(args.json))
    return 0


def cmd_demo_handoff(args: argparse.Namespace) -> int:
    from arena.adapters.policy_custom_torch import (
        build_module,
        export_from_checkpoint,
        verify_bundle_self,
    )
    from arena.core.identity import canonical_json, digest_uri, sha256_bytes
    from arena.core.io import publish_directory
    from arena.core.manifests import dump_json
    from arena.core.mirror import pull_artifact, push_artifact
    from arena.core.sdk import Policy

    destination = Path(args.out).resolve()
    result: dict[str, Any] = {}

    def build(stage: Path) -> None:
        import torch

        architecture = {
            "type": "mlp_categorical",
            "observation_dim": 4,
            "hidden_dims": [16, 16],
            "action_n": 3,
        }
        torch.manual_seed(0)
        checkpoint = stage / ".reference-checkpoint.pt"
        torch.save(build_module(architecture).state_dict(), checkpoint)
        source = export_from_checkpoint(
            source=checkpoint,
            out=stage / "source-policy.arena",
            role="agent",
            name="arena-quickstart-reference",
            architecture=architecture,
            observation={
                "type": "Box",
                "shape": [4],
                "dtype": "float32",
                "low": -10.0,
                "high": 10.0,
            },
            action={
                "type": "Discrete",
                "n": 3,
                "dtype": "int64",
                "masks": "none",
            },
            preprocessing={"id": "normalize_v0", "mean": 0.0, "std": 1.0},
        )
        checkpoint.unlink()
        verification = verify_bundle_self(source)
        source_policy = Policy.load(source)
        pushed = push_artifact(source, (stage / "store").as_uri(), verify=True)
        restored_path = stage / "restored-policy.arena"
        pulled = pull_artifact(pushed["uri"], restored_path, verify=True)
        restored_policy = Policy.load(restored_path)
        if restored_policy.digest != source_policy.digest:
            raise ArenaError(
                "quickstart handoff changed policy identity",
                code="DEMO_IDENTITY_MISMATCH",
                repair="Report this as an Arena integrity defect; do not use the restored artifact.",
            )
        intent = {
            "schema": "arena.evaluation-intent/v1",
            "task": {
                "env": "arena/quickstart-reference-v1",
                "interaction": "offline-reference",
            },
            "policy": source_policy.digest,
            "metric": "reference-case-conformance",
            "tolerance": {"atol": 1e-6, "rtol": 1e-6},
        }
        result.update(
            {
                "schema": "arena.demo-handoff/v1",
                "ok": True,
                "source_policy": "source-policy.arena",
                "restored_policy": "restored-policy.arena",
                "source_digest": source_policy.digest,
                "restored_digest": restored_policy.digest,
                "verification": verification,
                "evaluation_intent_digest": digest_uri(
                    sha256_bytes(canonical_json(intent))
                ),
                "capabilities": {
                    "runtime": "native:stable",
                    "store": "file:stable",
                    "provider": "reference-conformance:stable",
                },
                "next": (
                    "Run `arena doctor --capability openenv` before the external "
                    "runtime equivalence journey."
                ),
                "pull": {
                    "identity": pulled["identity"],
                    "kind": pulled["kind"],
                    "out": "restored-policy.arena",
                    "files": [
                        {
                            **entry,
                            "path": str(
                                Path("restored-policy.arena")
                                / Path(entry["path"]).relative_to(restored_path)
                            ),
                        }
                        for entry in pulled["files"]
                    ],
                    "verified": pulled["verified"],
                    "content_verified": pulled["content_verified"],
                    "identity_verified": pulled["identity_verified"],
                },
            }
        )
        dump_json(result, stage / "result.json")

    def verify(stage: Path) -> None:
        restored = Policy.load(stage / "restored-policy.arena")
        if restored.digest != result.get("source_digest"):
            raise ArenaError("staged quickstart result failed final identity verification")

    publish_directory(destination, build, verify=verify, replace=True)
    rendered = {**result, "out": str(destination)}
    _print(rendered, as_json=bool(args.json))
    return 0


def cmd_demo_multiagent(args: argparse.Namespace) -> int:
    """Run the packaged PettingZoo classic/rps_v2 multi-agent demo."""
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "examples" / "multiagent" / "run_demo.py"
    spec = importlib.util.spec_from_file_location("arena_multiagent_demo", path)
    if spec is None or spec.loader is None:
        raise ArenaError(f"cannot load multiagent demo from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    summary = module.run_multiagent_demo(out=Path(args.out))
    _print(summary, as_json=bool(args.json))
    return 0 if summary.get("ok") else 1



def cmd_init(args: argparse.Namespace) -> int:
    from arena.core.store import LocalStore

    store = LocalStore(args.path)
    path = store.init(force=args.force)
    print(f"Initialized Arena workspace at {path}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    from arena.core.manifests import load_manifest, resolve_artifact_path
    from arena.core.sdk import Policy

    artifact = Path(args.artifact)
    if artifact.is_dir() and (artifact / "bundle.yaml").exists():
        from arena.runtime.trajectory import inspect_trajectory

        info = inspect_trajectory(artifact)
        _print(info, as_json=args.json)
        return 0

    path = resolve_artifact_path(artifact)
    data = load_manifest(path)
    schema = data.get("schema", "")
    if schema.startswith("arena.policy"):
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
    elif schema.startswith("arena.match"):
        info = {"kind": "match", **data}
    else:
        info = {"kind": "manifest", **data}
    _print(info, as_json=args.json)
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    from arena.core.errors import CompatibilityError
    from arena.core.sdk import Policy, check

    config = _parse_json_arg(args.config) if getattr(args, "config", None) else None
    if config is not None and not isinstance(config, dict):
        raise SystemExit("--config must be a JSON/YAML object")
    task = _load_task_arg(args.task, config=config)
    policy = Policy.load(args.policy)
    report = check(task, policy.as_role(args.role), action_mode=args.action_mode)
    if not report.ok:
        raise CompatibilityError(
            report.format_human(),
            context=report.to_dict(),
        )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.format_human())
    return 0


def cmd_policy_export(args: argparse.Namespace) -> int:
    from arena.adapters.policy_custom_torch import (
        export_from_checkpoint,
        export_module_from_checkpoint,
    )
    from arena.core.manifests import load_manifest

    if args.adapter != "custom-pytorch":
        from arena.core.errors import SchemaError

        raise SchemaError(
            (
                f"unsupported adapter kind {args.adapter!r}. Supported: ['custom-pytorch']. "
                "Use --adapter custom-pytorch, or extend Arena with a registered policy adapter "
                "and run `arena adapter qualify` before claiming support. Arena will not silently "
                "coerce adapters."
            ),
            code="UNKNOWN_KIND",
            cause="unsupported policy export adapter",
            repair="Pass --adapter custom-pytorch (currently the only supported export adapter).",
            context={"adapter": args.adapter, "known": ["custom-pytorch"]},
        )

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
        from arena.adapters.policy_custom_torch import (
            _extract_state_dict,
            load_checkpoint_file,
        )
        from arena.core.identity import digest_uri, sha256_file
        from arena.plugins.payloads import export_trusted_source_bundle

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
    from arena.adapters.policy_custom_torch import verify_bundle_self

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
    from arena.core.errors import IncompleteExecutionError
    from arena.core.manifests import expand_seeds, load_manifest, validate_match_manifest
    from arena.core.sdk import Match, Policy, Task

    path = Path(args.match_manifest)
    data = validate_match_manifest(load_manifest(path))
    if isinstance(data["task"], dict):
        task_spec = dict(data["task"])
    else:
        task_ref = Path(str(data["task"]))
        if not task_ref.is_absolute():
            task_ref = (path.parent / task_ref).resolve()
        task_spec = dict(Task.load(task_ref if task_ref.exists() else data["task"]).spec)
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
    if result["outcome"]["failure_count"]:
        raise IncompleteExecutionError(
            "match did not complete every requested episode",
            code="MATCH_INCOMPLETE",
            context={
                "run_id": result["run_id"],
                "outcome": result["outcome"],
                "output": out,
            },
        )
    print(f"Match complete: {result['run_id']}")
    print(f"  completed={result['outcome']['episodes_completed']} failures=0")
    if out:
        print(f"  output={out}")
    return 0


def cmd_task_import(args: argparse.Namespace) -> int:
    from arena.core.manifests import TASK_SCHEMA, dump_yaml, task_content_digest

    out = args.out or (args.name.replace(":", "-").replace("@", "-") + ".yaml")
    if args.source.startswith("openenv://"):
        from arena.core.tasks import import_openenv_task

        manifest = import_openenv_task(
            args.source,
            name=args.name,
            out=out,
            contract_path=args.contract,
            source_revision=args.source_revision,
            timeout_seconds=args.timeout,
        )
    elif args.source.startswith("openspiel://"):
        from arena.adapters.task_openspiel import OpenSpielPackager, interaction_for_game

        manifest = {
            "schema": TASK_SCHEMA,
            "name": args.name,
            "adapter": "openspiel",
            "env": args.source,
            "interaction": interaction_for_game(args.source),
            "packaging": {"kind": "openspiel"},
        }
        description = OpenSpielPackager().describe_task(manifest)
        manifest["version"] = description["version"]
        if args.source_revision:
            manifest["source_revision"] = args.source_revision
        manifest["digest"] = task_content_digest(manifest)
        dump_yaml(manifest, out)
    else:
        from arena.core.registry import TASK_PACKAGERS, ensure_plugins_loaded

        ensure_plugins_loaded()
        scheme = args.source.split(":", 1)[0]
        TASK_PACKAGERS.get(scheme)
        raise SystemExit(f"task import for registered scheme {scheme!r} has no importer")
    _print(manifest, as_json=args.json)
    return 0


def cmd_task_verify_equivalence(args: argparse.Namespace) -> int:
    from arena.core.manifests import dump_json, load_manifest
    from arena.core.tasks import load_task_spec, verify_task_equivalence

    suite = load_manifest(args.trace_suite)
    result = verify_task_equivalence(
        load_task_spec(args.left),
        load_task_spec(args.right) if args.right else None,
        suite,
    )
    if args.out:
        dump_json(result, args.out)
    _print(result, as_json=args.json)
    return 0


def cmd_data_inspect(args: argparse.Namespace) -> int:
    from arena.core.errors import IncompleteExecutionError
    from arena.runtime.trajectory import inspect_trajectory

    info = inspect_trajectory(args.trajectory)
    if not info.get("completeness", {}).get("ok", True):
        raise IncompleteExecutionError(
            "trajectory is incomplete",
            code="TRAJECTORY_INCOMPLETE",
            context=info,
        )
    _print(info, as_json=args.json)
    return 0


def cmd_data_select(args: argparse.Namespace) -> int:
    from arena.core.dataset import select_episodes

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


def cmd_data_materialize(args: argparse.Namespace) -> int:
    from arena.core.dataset import materialize_dataset

    splits: dict[str, float] | None = None
    if args.split:
        splits = {}
        for item in args.split:
            if "=" not in item:
                raise SystemExit("--split must be NAME=WEIGHT")
            name, raw_weight = item.split("=", 1)
            if name in splits:
                raise SystemExit(f"duplicate --split name {name!r}")
            try:
                splits[name] = float(raw_weight)
            except ValueError as exc:
                raise SystemExit(f"--split weight must be numeric: {item!r}") from exc
    dataset = materialize_dataset(
        args.dataset,
        out_dir=args.out,
        splits=splits,
        split_seed=args.split_seed,
    )
    _print(dataset, as_json=args.json)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from arena.runtime.training import run_training_recipe

    run = run_training_recipe(args.recipe, out_dir=args.out)
    _print(run, as_json=args.json)
    return 0


def cmd_population_create(args: argparse.Namespace) -> int:
    from arena.core.population import create_population_from_yaml, write_population_yaml
    from arena.core.store import LocalStore

    store = LocalStore.find()
    pop = create_population_from_yaml(args.manifest, store=store, ref=args.ref)
    if args.out:
        write_population_yaml(pop, args.out)
    _print(pop, as_json=args.json)
    return 0


def cmd_population_inspect(args: argparse.Namespace) -> int:
    from arena.core.manifests import population_content_digest
    from arena.core.population import load_population
    from arena.core.store import LocalStore

    store = LocalStore.find()
    pop = load_population(args.ref, store=store)
    pop = {**pop, "digest": pop.get("digest") or population_content_digest(pop)}
    _print(pop, as_json=args.json)
    return 0


def cmd_eval_validate(args: argparse.Namespace) -> int:
    from arena.core.manifests import (
        evaluation_content_digest,
        evaluation_intent_digest,
    )
    from arena.core.population import load_population
    from arena.core.sdk import Policy
    from arena.core.store import LocalStore
    from arena.runtime.evaluation import load_evaluation, validate_evaluation

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
    identity_assignments: dict[str, Any] = {}
    for role, spec in suite["assignments"].items():
        if isinstance(spec, str):
            candidate = Path(spec) if Path(spec).is_absolute() else base / spec
            identity_assignments[role] = (
                Policy.load(candidate).digest if candidate.exists() else spec
            )
        elif isinstance(spec, dict) and spec.get("kind", "policy") == "policy":
            key = "policy" if "policy" in spec else "ref"
            ref = str(spec[key])
            candidate = Path(ref) if Path(ref).is_absolute() else base / ref
            identity_assignments[role] = {
                **spec,
                key: Policy.load(candidate).digest if candidate.exists() else ref,
            }
        elif isinstance(spec, dict) and spec.get("kind") in {
            "population",
            "crossplay",
        }:
            ref = str(spec["population"])
            population = populations.get(ref)
            identity_assignments[role] = {
                **spec,
                "population": population["digest"] if population else ref,
            }
        else:
            identity_assignments[role] = spec
    identity_suite = {**suite, "assignments": identity_assignments}
    validate_evaluation(identity_suite, populations=populations)
    _print(
        {
            "ok": True,
            "evaluation_digest": evaluation_content_digest(identity_suite),
            "evaluation_intent_digest": evaluation_intent_digest(identity_suite),
        },
        as_json=bool(args.json),
    )
    return 0


def cmd_eval_run(args: argparse.Namespace) -> int:
    from pathlib import Path

    from arena.core.errors import IncompleteExecutionError
    from arena.core.population import load_population
    from arena.core.store import LocalStore
    from arena.runtime.evaluation import load_evaluation, run_evaluation

    suite_path = Path(args.suite)
    suite = load_evaluation(suite_path)
    base = suite_path.parent.resolve()
    policy_index: dict[str, Path] = {}
    for item in args.policy:
        if "=" not in item:
            from arena.core.errors import CliUsageError

            raise CliUsageError(
                (
                    "--policy must be digest=path or name=path "
                    f"(got {item!r}). Example: sha256:<64-hex>=./policy_bundle"
                ),
                code="USAGE_INVALID",
                cause="policy binding is not digest=path or name=path",
                repair=(
                    "Pass --policy as digest=path or name=path, e.g. "
                    "`sha256:<64-hex>=./my_policy` using the digest from policy export."
                ),
                context={"value": item},
            )
        key, path = item.split("=", 1)
        policy_index[key] = Path(path)
        from arena.core.sdk import Policy

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
        elif isinstance(spec, str):
            candidate = Path(spec) if Path(spec).is_absolute() else (base / spec)
            assigns[role] = str(candidate.resolve()) if candidate.exists() else spec
        elif isinstance(spec, dict) and spec.get("kind", "policy") == "policy":
            pref = str(spec.get("policy") or spec.get("ref"))
            candidate = Path(pref) if Path(pref).is_absolute() else (base / pref)
            resolved = str(candidate.resolve()) if candidate.exists() else pref
            key = "policy" if "policy" in spec else "ref"
            assigns[role] = {**spec, key: resolved}
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
        provider=args.provider,
    )
    summary = {
        "run_id": result["run_id"],
        "run_dir": result["run_dir"],
        "evaluation_digest": result["evaluation_digest"],
        "evaluation_intent_digest": result.get("evaluation_intent_digest"),
        "execution_binding_digest": result.get("execution_binding_digest"),
        "semantic_result_digest": result.get("semantic_result_digest"),
        "state": result.get("state", "complete"),
        "denominators": result.get("denominators"),
        "cells": len(result["cells"]),
        "sampling_ledger": result["sampling_ledger"],
    }
    if result.get("state", "complete") != "complete":
        raise IncompleteExecutionError(
            "evaluation did not complete every declared attempt",
            code="EVALUATION_INCOMPLETE",
            cause=str(result.get("state")),
            repair=(
                f"Inspect {result['run_dir']}/eval_run.json, repair the recorded "
                "failures, and retry to a new output path."
            ),
            context=summary,
        )
    _print(summary, as_json=args.json)
    return 0


def cmd_eval_report(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from arena.core.manifests import dump_json, dump_yaml
    from arena.runtime.evaluation import build_eval_report

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
        from arena.core.manifests import load_manifest

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

    from arena.core.eval_bundle import build_eval_bundle

    report = None
    if args.report:
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    bundle = build_eval_bundle(eval_run_dir=args.run_dir, report=report, out_dir=args.out)
    print(json.dumps({"digest": bundle["digest"], "out": args.out}, indent=2))
    return 0


def cmd_eval_compare(args: argparse.Namespace) -> int:
    from arena.core.eval_compare import compare_eval_claims

    result = compare_eval_claims(args.left, args.right)
    _print(result, as_json=args.json)
    return 0


def cmd_release_build(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from arena.core.eval_bundle import build_eval_bundle

    report = None
    if args.report:
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    bundle = build_eval_bundle(eval_run_dir=args.eval_run, report=report, out_dir=args.out)
    print(json.dumps({"digest": bundle["digest"], "out": args.out}, indent=2))
    return 0


def cmd_release_verify(args: argparse.Namespace) -> int:
    from arena.core.release import verify_release_evidence

    result = verify_release_evidence(
        args.evidence_index,
        signature=args.signature,
        public_key=args.key,
        current_ledger=args.current_ledger,
        current_ledger_signature=args.ledger_signature,
        current_ledger_key=args.ledger_key,
    )
    _print(result, as_json=bool(args.json))
    return 0


def cmd_release_assemble(args: argparse.Namespace) -> int:
    from arena.core.release import assemble_release_evidence

    gates: dict[str, str] = {}
    for item in args.gate:
        if "=" not in item:
            raise CliUsageError("--gate must be R-NN=PATH")
        gate_id, path = item.split("=", 1)
        if gate_id in gates:
            raise CliUsageError(f"duplicate --gate {gate_id}")
        gates[gate_id] = path
    result = assemble_release_evidence(
        release=args.release,
        tag=args.tag,
        commit=args.commit,
        gates=gates,
        artifacts=args.artifact,
        out=args.out,
        eval_bundles=args.eval_bundle or None,
    )
    _print(result, as_json=bool(args.json))
    return 0


def cmd_release_sign(args: argparse.Namespace) -> int:
    from arena.core.manifests import load_manifest
    from arena.core.release import (
        CURRENT_SCHEMA,
        EVIDENCE_SCHEMA,
        sign_qualification_ledger,
        sign_release_evidence,
    )

    schema = load_manifest(args.document).get("schema")
    if schema == EVIDENCE_SCHEMA:
        result = sign_release_evidence(
            args.document,
            private_key=args.key,
            out=args.out,
        )
    elif schema == CURRENT_SCHEMA:
        result = sign_qualification_ledger(
            args.document,
            private_key=args.key,
            out=args.out,
        )
    else:
        raise ArenaError(
            f"release sign does not support schema {schema!r}",
            code="RELEASE_DOCUMENT_UNSUPPORTED",
        )
    _print(result, as_json=bool(args.json))
    return 0


def cmd_adapter_qualify(args: argparse.Namespace) -> int:
    from arena.conformance.qualification import (
        qualify_adapter_fixture,
        qualify_task_fixture,
    )
    from arena.core.errors import ConformanceError

    if args.trace_suite:
        report = qualify_task_fixture(
            args.fixture,
            peer=args.peer,
            trace_suite=args.trace_suite,
            report_path=args.out,
        )
    else:
        report = qualify_adapter_fixture(args.fixture, report_path=args.out)
    if not report["ok"]:
        raise ConformanceError(
            "adapter qualification failed",
            code="ADAPTER_QUALIFICATION_FAILED",
            context=report,
        )
    _print(report, as_json=bool(args.json))
    return 0


def cmd_capture(args: argparse.Namespace) -> int:
    from arena.adapters.task_pettingzoo.adapter import make_env
    from arena.core.capture import capture_draft_from_env

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


def cmd_attest(args: argparse.Namespace) -> int:
    from arena.core.attestation import (
        generate_signing_keypair,
        sign_artifact,
        verify_artifact_attestation,
    )

    if args.attest_command == "keygen":
        result = generate_signing_keypair(
            private_key=args.private,
            public_key=args.public,
        )
    elif args.attest_command == "sign":
        result = sign_artifact(
            args.source,
            private_key=args.key,
            out=args.out,
            issuer=args.issuer,
        )
    elif args.attest_command == "verify":
        result = verify_artifact_attestation(
            args.source,
            attestation=args.attestation,
            public_key=args.key,
        )
    else:
        raise CliUsageError(f"unknown attest command {args.attest_command!r}")
    _print(result, as_json=bool(args.json))
    return 0


def cmd_push(args: argparse.Namespace) -> int:
    from arena.core.mirror import push_artifact

    result = push_artifact(args.source, args.destination, verify=args.verify)
    _print(result, as_json=args.json)
    return 0


def cmd_pull(args: argparse.Namespace) -> int:
    from urllib.parse import urldefrag

    from arena.core.mirror import pull_artifact

    _base, identity = urldefrag(args.source)
    out = args.out or f"pulled-{identity.removeprefix('sha256:')[:12]}.arena"
    result = pull_artifact(args.source, out, verify=args.verify)
    _print(result, as_json=args.json)
    return 0


def cmd_store_qualify(args: argparse.Namespace) -> int:
    from arena.conformance.qualification import qualify_store
    from arena.core.errors import ConformanceError

    report = qualify_store(
        args.source,
        destination=args.destination,
        report_path=args.out,
    )
    if not report["ok"]:
        raise ConformanceError(
            "store qualification failed",
            code="STORE_QUALIFICATION_FAILED",
            context=report,
        )
    _print(report, as_json=bool(args.json))
    return 0


def _load_task_arg(task_arg: str, *, config: dict[str, Any] | None = None) -> Any:
    """Load a task from env id, task YAML, or match.yaml (using its nested task).

    Optional ``config`` is merged into ``task.spec["config"]`` so ``arena check``
    validates against the configured spaces (not just env defaults).
    """
    from arena.core.manifests import MATCH_SCHEMA, load_manifest
    from arena.core.sdk import Task

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
    elif task_arg.startswith(("pettingzoo://", "openspiel://")):
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
