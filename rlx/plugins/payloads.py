"""Actor payload cases: template | torchscript | trusted_source."""

from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Protocol

from rlx.core.errors import ConformanceError, SchemaError
from rlx.core.identity import digest_uri, sha256_file
from rlx.core.registry import PAYLOAD_LOADERS

TRUSTED_SOURCE_WARNING = (
    "WARNING: payload tier 'trusted_source' executes digest-pinned Python from the "
    "bundle. This is NOT sandboxed. Only opt in with --trust-source for code you "
    "already trust. Prefer TorchScript (tier=torchscript) whenever the actor is "
    "scriptable."
)


class PayloadCase(Protocol):
    kind: str

    def required_payload_keys(self) -> tuple[str, ...]: ...

    def load_module(
        self,
        *,
        manifest: dict[str, Any],
        root: Path,
        trust_source: bool = False,
    ) -> Any: ...


def register_payload_case(kind: str, case: PayloadCase, *, replace: bool = False) -> PayloadCase:
    return PAYLOAD_LOADERS.register(kind, case, replace=replace)


def _bundle_payload_path(root: Path, raw_path: Any) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise SchemaError("payload path must be a non-empty relative string")
    candidate = root / raw_path
    try:
        resolved_root = root.resolve()
        resolved_candidate = candidate.resolve()
    except OSError as e:
        raise SchemaError(f"cannot resolve payload path {raw_path!r}") from e
    if Path(raw_path).is_absolute() or resolved_root not in resolved_candidate.parents:
        raise SchemaError(f"payload path escapes policy bundle: {raw_path!r}")
    return resolved_candidate


def _verify_digest(manifest: dict[str, Any], key: str, path: Path) -> None:
    entry = manifest.get("payloads", {}).get(key, {})
    declared = entry.get("digest") if isinstance(entry, dict) else None
    if not declared:
        raise SchemaError(f"policy {key} payload missing or lacks a digest: {path}")
    if not path.exists():
        raise SchemaError(f"policy {key} payload missing: {path}")
    actual = digest_uri(sha256_file(path))
    if actual != declared:
        raise ConformanceError(
            f"policy bundle integrity check failed: {key} payload digest mismatch "
            f"(declared {declared}, actual {actual})"
        )


class TemplatePayload:
    kind = "template"

    def required_payload_keys(self) -> tuple[str, ...]:
        return ("weights",)

    def load_module(
        self,
        *,
        manifest: dict[str, Any],
        root: Path,
        trust_source: bool = False,
    ) -> Any:
        del trust_source
        from rlx.adapters.policy_custom_torch import _require_torch, build_module

        torch, _, _ = _require_torch()
        entry = manifest["payloads"]["weights"]
        weights_path = _bundle_payload_path(root, entry["path"])
        _verify_digest(manifest, "weights", weights_path)
        module = build_module(manifest["architecture"])
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        module.load_state_dict(state)
        module.eval()
        return module


class TorchScriptPayload:
    kind = "torchscript"

    def required_payload_keys(self) -> tuple[str, ...]:
        return ("model",)

    def load_module(
        self,
        *,
        manifest: dict[str, Any],
        root: Path,
        trust_source: bool = False,
    ) -> Any:
        del trust_source
        from rlx.adapters.policy_custom_torch import _require_torch

        torch, _, _ = _require_torch()
        entry = manifest["payloads"]["model"]
        model_path = _bundle_payload_path(root, entry["path"])
        _verify_digest(manifest, "model", model_path)
        module = torch.jit.load(str(model_path), map_location="cpu")
        module.eval()
        return module


class TrustedSourcePayload:
    """Opt-in digest-pinned Python source tier (NOT a sandbox; not the default)."""

    kind = "trusted_source"

    def required_payload_keys(self) -> tuple[str, ...]:
        return ("weights", "inference_src")

    def load_module(
        self,
        *,
        manifest: dict[str, Any],
        root: Path,
        trust_source: bool = False,
    ) -> Any:
        if not trust_source:
            raise SchemaError(
                "payload tier 'trusted_source' refused by default. "
                "This path executes digest-pinned Python and is NOT sandboxed. "
                "Prefer TorchScript. To opt in explicitly, pass trust_source=True "
                "/ --trust-source after reviewing the pinned source digests."
            )
        warnings.warn(TRUSTED_SOURCE_WARNING, UserWarning, stacklevel=2)
        from rlx.adapters.policy_custom_torch import _require_torch

        torch, _, _ = _require_torch()
        src_entry = manifest["payloads"]["inference_src"]
        src_path = _bundle_payload_path(root, src_entry["path"])
        _verify_digest(manifest, "inference_src", src_path)
        weights_entry = manifest["payloads"]["weights"]
        weights_path = _bundle_payload_path(root, weights_entry["path"])
        _verify_digest(manifest, "weights", weights_path)

        factory_attr = (
            manifest.get("runtime", {}).get("trusted_source", {}).get("factory")
            or manifest.get("architecture", {}).get("factory")
            or "build_actor"
        )
        module_obj = _import_pinned_source(src_path, attr=factory_attr)
        if isinstance(module_obj, type) and issubclass(module_obj, torch.nn.Module):
            actor = module_obj()
        elif callable(module_obj):
            actor = module_obj()
        else:
            raise SchemaError(
                f"trusted_source factory {factory_attr!r} must be an nn.Module subclass "
                "or a zero-arg factory"
            )
        if not isinstance(actor, torch.nn.Module):
            raise SchemaError(
                f"trusted_source factory returned {type(actor).__name__}, expected nn.Module"
            )
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        actor.load_state_dict(state)
        actor.eval()
        return actor


def _import_pinned_source(src_path: Path, *, attr: str) -> Any:
    """Import a single digest-verified .py file without adding trainer repos to path."""
    if src_path.suffix != ".py":
        raise SchemaError(
            f"trusted_source inference_src must be a single .py file, got {src_path.name!r}"
        )
    mod_name = f"rlx_trusted_source_{sha256_file(src_path)[:12]}"
    spec = importlib.util.spec_from_file_location(mod_name, src_path)
    if spec is None or spec.loader is None:
        raise SchemaError(f"cannot import trusted_source module from {src_path}")
    mod = importlib.util.module_from_spec(spec)
    # Isolate from ambient trainer packages: do not mutate sys.path.
    sys.modules[mod_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # noqa: BLE001
        sys.modules.pop(mod_name, None)
        raise SchemaError(f"trusted_source module failed to import: {e}") from e
    if not hasattr(mod, attr):
        raise SchemaError(f"trusted_source module has no attribute {attr!r}")
    return getattr(mod, attr)


def export_trusted_source_bundle(
    *,
    out_dir: Path | str,
    name: str,
    roles: list[str],
    source_py: Path | str,
    state_dict: dict[str, Any],
    observation: dict[str, Any],
    action: dict[str, Any],
    factory: str = "build_actor",
    io: dict[str, Any] | None = None,
    preprocessing: dict[str, Any] | None = None,
    reference_cases: list[dict[str, Any]] | None = None,
    source_act_fn: Any | None = None,
    trust_source: bool = False,
    lineage: dict[str, Any] | None = None,
) -> Path:
    """Publish a minimal trusted_source bundle (requires explicit trust at export).

    Export still requires ``trust_source=True`` so authors cannot accidentally
    publish an arbitrary-code tier. TorchScript remains the preferred path.
    """
    if not trust_source:
        raise SchemaError(
            "refusing to export trusted_source payload without trust_source=True / "
            "--trust-source. Prefer TorchScript. trusted_source is NOT sandboxed."
        )
    warnings.warn(TRUSTED_SOURCE_WARNING, UserWarning, stacklevel=2)
    from rlx.adapters.policy_custom_torch import (
        PROVENANCE_SELF,
        PROVENANCE_SOURCE,
        _publish_staging,
        _require_torch,
        pack_reference_cases,
    )
    from rlx.adapters.policy_custom_torch.preprocess import canonical_pipeline
    from rlx.core.action_cases import action_type, box_distribution
    from rlx.core.contracts import validate_architecture_spaces, validate_reference_case_action
    from rlx.core.manifests import (
        POLICY_SCHEMA,
        dump_yaml,
        policy_content_digest,
        validate_policy_manifest,
    )

    torch, _, _ = _require_torch()
    out_dir = Path(out_dir)
    source_py = Path(source_py)
    if not source_py.is_file() or source_py.suffix != ".py":
        raise SchemaError("trusted_source export requires a single .py source file")
    cases = list(reference_cases or [])
    if not cases:
        raise SchemaError(
            "trusted_source export requires source-captured reference_cases; "
            "refusing to mint unverifiable cases"
        )
    io = dict(io or {})
    architecture = {
        "type": "trusted_source_module",
        "factory": factory,
        "io": io,
    }
    action = dict(action)
    action.setdefault("masks", "none")
    # Reuse serialized-module space rules (Box obs + complete action cases).
    architecture_for_spaces = {"type": "serialized_module", "io": io}
    validate_architecture_spaces(
        observation=observation,
        action=action,
        architecture=architecture_for_spaces,
        adapter="custom-pytorch",
    )

    parent = out_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".rlx-export-", dir=str(parent)))
    try:
        payloads = staging / "payloads"
        payloads.mkdir()
        src_dest = payloads / "inference.py"
        shutil.copy2(source_py, src_dest)
        weights_path = payloads / "weights.pt"
        torch.save(state_dict, weights_path)
        prep = canonical_pipeline(preprocessing, observation)
        prep_path = payloads / "preprocess.json"
        import json

        prep_path.write_text(json.dumps(prep, sort_keys=True), encoding="utf-8")

        provenance = PROVENANCE_SOURCE if source_act_fn is not None else PROVENANCE_SELF
        if source_act_fn is not None:
            captured: list[dict[str, Any]] = []
            for case in cases:
                case = dict(case)
                if "seed" in case and "rng" not in case:
                    import numpy as np

                    case = {**case, "rng": np.random.default_rng(int(case["seed"]))}
                result = source_act_fn(case)
                if isinstance(result, tuple):
                    case["expected_action"], case["expected_logits"] = result
                else:
                    case["expected_action"] = result
                case.pop("rng", None)
                captured.append(case)
            cases = captured
        for i, case in enumerate(cases):
            validate_reference_case_action(case, action=action, index=i)
        cases_path = payloads / "reference_cases.json"
        cases_path.write_text(
            json.dumps(pack_reference_cases(cases, provenance=provenance), indent=2),
            encoding="utf-8",
        )
        modes = (
            ["deterministic"]
            if action_type(action) == "Box" and box_distribution(action) == "deterministic"
            else ["deterministic", "stochastic"]
        )
        manifest = {
            "schema": POLICY_SCHEMA,
            "name": name,
            "roles": {"allowed": roles},
            "runtime": {
                "adapter": "custom-pytorch",
                "tier": "trusted_source",
                "trusted_source": {"factory": factory, "sandboxed": False},
                "torch": str(torch.__version__),
                "python": "3.12",
            },
            "observation": observation,
            "action": action,
            "state": {"recurrent": bool(io.get("recurrent")), "reset_on": io.get("reset_on", [])},
            "inference": {"modes": modes},
            "preprocessing": prep,
            "architecture": architecture,
            "payloads": {
                "weights": {
                    "path": "payloads/weights.pt",
                    "digest": digest_uri(sha256_file(weights_path)),
                },
                "inference_src": {
                    "path": "payloads/inference.py",
                    "digest": digest_uri(sha256_file(src_dest)),
                },
                "preprocess": {
                    "path": "payloads/preprocess.json",
                    "digest": digest_uri(sha256_file(prep_path)),
                },
                "reference_cases": {
                    "path": "payloads/reference_cases.json",
                    "digest": digest_uri(sha256_file(cases_path)),
                },
            },
            "conformance": {
                "status": "source-captured" if provenance == PROVENANCE_SOURCE else "insufficient-evidence",
                "provenance": provenance,
            },
        }
        if lineage:
            manifest["lineage"] = dict(lineage)
        validate_policy_manifest(manifest)
        dump_yaml(manifest, staging / "policy.yaml")
        (staging / "DIGEST").write_text(policy_content_digest(manifest) + "\n", encoding="utf-8")
        return _publish_staging(staging, out_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def register_builtins() -> None:
    register_payload_case("template", TemplatePayload(), replace=True)
    register_payload_case("torchscript", TorchScriptPayload(), replace=True)
    register_payload_case("trusted_source", TrustedSourcePayload(), replace=True)
