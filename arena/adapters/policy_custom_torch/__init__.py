"""Custom PyTorch policy adapter: declarative architectures, export, verify, load."""

from __future__ import annotations

import json
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from arena.adapters.policy_custom_torch.preprocess import PreprocessPipeline, canonical_pipeline
from arena.core.action_cases import (
    action_type,
    actions_equal,
    box_distribution,
    decode_action_from_params,
    validate_runtime_action,
)
from arena.core.contracts import (
    check_observation_vector,
    validate_architecture_spaces,
    validate_reference_case_action,
)
from arena.core.errors import missing_extra, ArenaError, ConformanceError, SchemaError
from arena.core.identity import digest_uri, sha256_file
from arena.core.manifests import (
    POLICY_SCHEMA,
    dump_yaml,
    policy_content_digest,
    validate_policy_manifest,
)
from arena.core.spaces import normalize_space_descriptor

ADAPTER_NAME = "custom-pytorch"

# Provenance labels for embedded reference cases / verify results.
PROVENANCE_SELF = "self-consistency"
PROVENANCE_SOURCE = "source-conformance"

_TRAINING_CHECKPOINT_KEYS = (
    "state_dict",
    "model",
    "policy",
    "actor",
    "network",
    "model_state_dict",
    "policy_state_dict",
    "optimizer",
    "optimizer_state_dict",
    "scheduler",
    "epoch",
    "global_step",
    "iteration",
    "loss",
    "rng_state",
    "amp",
    "scaler",
    "config",
    "hyper_parameters",
    "hparams",
)


def _verify_weights_digest(manifest: dict[str, Any], weights_path: Path) -> None:
    """Content-addressed integrity: the weights file must hash to the manifest digest.

    A tampered/corrupted ``weights.pt`` (or a manifest pointing at swapped bytes)
    must be detected here rather than silently trusted at inference time.
    """
    entry = manifest.get("payloads", {}).get("weights", {})
    declared = entry.get("digest") if isinstance(entry, dict) else None
    if not declared:
        return
    if not weights_path.exists():
        raise SchemaError(f"policy weights payload missing: {weights_path}")
    actual = digest_uri(sha256_file(weights_path))
    if actual != declared:
        raise ConformanceError(
            "policy bundle integrity check failed: weights payload does not match "
            f"manifest digest (declared {declared}, actual {actual}). The bundle "
            "may be corrupted or tampered with."
        )


def _verify_payload_digest(manifest: dict[str, Any], key: str, path: Path) -> None:
    entry = manifest.get("payloads", {}).get(key, {})
    declared = entry.get("digest") if isinstance(entry, dict) else None
    if not declared or not path.exists():
        raise SchemaError(f"policy {key} payload missing or lacks a digest: {path}")
    actual = digest_uri(sha256_file(path))
    if actual != declared:
        raise ConformanceError(
            f"policy bundle integrity check failed: {key} payload digest mismatch "
            f"(declared {declared}, actual {actual})"
        )


def _bundle_payload_path(root: Path, raw_path: Any) -> Path:
    """Resolve a declared payload without permitting a bundle escape."""
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


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as e:
        raise missing_extra(
            "torch",
            feature="custom-pytorch adapter",
            capability="torch",
        ) from e
    return torch, nn, F


class Preprocess:
    def __init__(self, spec: dict[str, Any], *, obs_dim: int | None = None) -> None:
        self.spec = spec
        self.mean = np.asarray(spec.get("mean", 0.0), dtype=np.float32)
        self.std = np.asarray(spec.get("std", 1.0), dtype=np.float32)
        self.clip = spec.get("clip")
        self.obs_dim = obs_dim
        if obs_dim is not None:
            for name, arr in (("mean", self.mean), ("std", self.std)):
                if arr.ndim == 0 or arr.size == 1:
                    continue
                if int(arr.size) != int(obs_dim):
                    raise SchemaError(
                        f"preprocessing.{name} length {arr.size} != "
                        f"architecture.observation_dim {obs_dim}"
                    )

    def __call__(self, obs: Any) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32)
        if self.obs_dim is not None and x.reshape(-1).size != self.obs_dim:
            raise ConformanceError(
                f"observation length {x.reshape(-1).size} != "
                f"architecture.observation_dim {self.obs_dim} at preprocess"
            )
        try:
            x = (x - self.mean) / np.maximum(self.std, 1e-8)
        except ValueError as e:
            raise ConformanceError(
                f"preprocessing mean/std broadcast failed for observation shape "
                f"{x.shape}: {e}"
            ) from e
        if self.clip is not None:
            lo, hi = self.clip
            x = np.clip(x, lo, hi)
        return x


def build_module(architecture: dict[str, Any]):
    torch, nn, F = _require_torch()
    atype = architecture["type"]
    obs_dim = int(architecture["observation_dim"])
    action_n = int(architecture["action_n"])
    hidden = [int(h) for h in architecture.get("hidden_dims", [64, 64])]

    class MLPCategorical(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            prev = obs_dim
            for h in hidden:
                layers.extend([nn.Linear(prev, h), nn.Tanh()])
                prev = h
            layers.append(nn.Linear(prev, action_n))
            self.net = nn.Sequential(*layers)

        def forward(self, x, hidden=None):
            logits = self.net(x)
            return logits, hidden

        def initial_state(self, batch: int = 1):
            return None

    class GRUCategorical(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hidden_size = int(architecture.get("rnn_hidden_size", 32))
            self.input = nn.Linear(obs_dim, self.hidden_size)
            self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
            self.head = nn.Linear(self.hidden_size, action_n)

        def forward(self, x, hidden=None):
            # x: (B, obs) or (B, T, obs)
            if x.dim() == 2:
                x = x.unsqueeze(1)
            z = torch.tanh(self.input(x))
            if hidden is None:
                out, h = self.gru(z)
            else:
                out, h = self.gru(z, hidden)
            logits = self.head(out[:, -1, :])
            return logits, h

        def initial_state(self, batch: int = 1):
            return torch.zeros(1, batch, self.hidden_size)

    if atype == "mlp_categorical":
        return MLPCategorical()
    if atype == "gru_categorical":
        return GRUCategorical()
    raise SchemaError(f"unknown architecture type: {atype!r}")


class TorchPolicyRuntime:
    """Executable policy loaded from an Arena bundle (no training-repo imports)."""

    def __init__(
        self,
        manifest: dict[str, Any],
        weights_path: Path,
        *,
        bundle_root: Path | None = None,
        trust_source: bool = False,
    ) -> None:
        torch, _, _ = _require_torch()
        from arena.core.registry import PAYLOAD_LOADERS, ensure_plugins_loaded

        ensure_plugins_loaded()
        self.manifest = validate_policy_manifest(manifest)
        validate_architecture_spaces(
            observation=self.manifest["observation"],
            action=self.manifest["action"],
            architecture=self.manifest["architecture"],
            adapter=self.manifest.get("runtime", {}).get("adapter", ADAPTER_NAME),
        )
        self.tier = self.manifest.get("runtime", {}).get("tier", "template")
        self.trust_source = bool(trust_source)
        root = Path(bundle_root) if bundle_root is not None else Path(weights_path).parent.parent
        payload = PAYLOAD_LOADERS.get(str(self.tier))
        self.module = payload.load_module(
            manifest=self.manifest, root=root, trust_source=self.trust_source
        )
        if self.tier in {"torchscript", "trusted_source"}:
            self.preprocess = PreprocessPipeline(
                canonical_pipeline(self.manifest.get("preprocessing"), self.manifest["observation"]),
                self.manifest["observation"],
            )
        else:
            obs_dim = int(self.manifest["architecture"]["observation_dim"])
            self.preprocess = Preprocess(self.manifest.get("preprocessing", {}), obs_dim=obs_dim)
        self.module.eval()
        self._hidden: dict[str, Any] = {}
        self._last_logits: np.ndarray | None = None
        self.masks_mode = self.manifest["action"]["masks"]
        self.recurrent = bool(self.manifest["state"]["recurrent"])
        self.reset_on = set(self.manifest["state"].get("reset_on", []))
        self.digest = policy_content_digest(self.manifest)

    @property
    def last_logits(self) -> np.ndarray | None:
        """Logits from the most recent ``act()`` forward pass (same step as the action)."""
        return None if self._last_logits is None else self._last_logits.copy()

    def reset(self, agent_id: str = "default") -> None:
        if self.recurrent:
            if self._serialized_io:
                shape = self.manifest["architecture"].get("io", {}).get("hidden_shape")
                if not shape:
                    raise ConformanceError(
                        "serialized recurrent actor requires architecture.io.hidden_shape"
                    )
                self._hidden[agent_id] = _require_torch()[0].zeros(*shape)
            else:
                self._hidden[agent_id] = self.module.initial_state(1)
        else:
            self._hidden.pop(agent_id, None)
        if isinstance(self.preprocess, PreprocessPipeline):
            self.preprocess.reset(agent_id)

    def reset_agent(self, agent_id: str) -> None:
        if self.recurrent and "agent_termination" in self.reset_on:
            self._hidden[agent_id] = self.module.initial_state(1)

    @property
    def _serialized_io(self) -> bool:
        return self.tier in {"torchscript", "trusted_source"}

    def _encode_obs(self, observation: Any) -> np.ndarray:
        if self._serialized_io:
            return self.preprocess(observation, agent_id="default")
        obs_dim = int(self.manifest["architecture"]["observation_dim"])
        x = check_observation_vector(
            observation,
            obs_space=self.manifest["observation"],
            obs_dim=obs_dim,
        )
        return self.preprocess(x)

    def act(
        self,
        observation: Any,
        *,
        mode: str = "deterministic",
        action_mask: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
        agent_id: str = "default",
    ) -> Any:
        torch, _, _ = _require_torch()
        if self.masks_mode == "required" and action_mask is None:
            raise ConformanceError("action mask required but missing")
        x = (
            self.preprocess(observation, agent_id)
            if self._serialized_io
            else self._encode_obs(observation)
        )
        xt = torch.as_tensor(x, dtype=torch.float32)
        if not self._serialized_io:
            xt = xt.view(1, -1)
        else:
            xt = xt.unsqueeze(0)
        hidden = self._hidden.get(agent_id)
        with torch.no_grad():
            output = self._forward(xt, hidden, action_mask, agent_id)
            if isinstance(output, tuple):
                logits, new_hidden = output[0], output[1] if len(output) > 1 else hidden
            else:
                logits, new_hidden = output, hidden
            if self.recurrent:
                self._hidden[agent_id] = new_hidden
            if not bool(torch.isfinite(logits).all()):
                raise ConformanceError(
                    "policy produced non-finite logits (NaN/Inf); refusing to emit an action"
                )
            # Stash raw actor params/logits from THIS forward so verify can compare
            # without re-stepping RNNs. Masking is applied inside decode for Discrete
            # / MultiDiscrete cases so last_logits stays the unmasked head output.
            self._last_logits = logits.detach().cpu().numpy().reshape(-1).copy()
            action = decode_action_from_params(
                self._last_logits,
                action=self.manifest["action"],
                mode=mode,
                rng=rng,
                action_mask=action_mask,
            )
        validate_runtime_action(action, action=self.manifest["action"])
        return action

    def _forward(self, xt, hidden, action_mask, agent_id: str = "default"):
        """Invoke only the explicitly declared serialized actor contract."""
        if not self._serialized_io:
            return self.module(xt, hidden)
        io = self.manifest["architecture"].get("io", {})
        needs_hidden = bool(io.get("recurrent", False))
        needs_mask = bool(io.get("mask_in_graph", False))
        if needs_mask and action_mask is None:
            raise ConformanceError("action mask required by serialized actor graph but missing")
        args = [xt]
        if needs_hidden:
            if hidden is None:
                self.reset(agent_id)
                hidden = self._hidden[agent_id]
            args.append(hidden)
        if needs_mask:
            args.append(_require_torch()[0].as_tensor(action_mask, dtype=_require_torch()[0].bool).view(1, -1))
        return self.module(*args)

    def logits(
        self,
        observation: Any,
        *,
        action_mask: np.ndarray | None = None,
        agent_id: str = "default",
        update_hidden: bool = True,
    ) -> np.ndarray:
        torch, _, _ = _require_torch()
        x = (
            self.preprocess(observation, agent_id)
            if self._serialized_io
            else self._encode_obs(observation)
        )
        xt = torch.as_tensor(x, dtype=torch.float32)
        xt = xt.unsqueeze(0) if self._serialized_io else xt.view(1, -1)
        hidden = self._hidden.get(agent_id)
        with torch.no_grad():
            output = self._forward(xt, hidden, action_mask, agent_id)
            logits, new_hidden = output if isinstance(output, tuple) else (output, hidden)
            if update_hidden and self.recurrent:
                self._hidden[agent_id] = new_hidden
            if action_mask is not None:
                mask_t = torch.as_tensor(np.asarray(action_mask, dtype=np.bool_), dtype=torch.bool)
                logits = logits.masked_fill(~mask_t.view(1, -1), -1e9)
        return logits.cpu().numpy().reshape(-1)


def _write_bundle_into(
    staging: Path,
    *,
    name: str,
    roles: list[str],
    observation: dict[str, Any],
    action: dict[str, Any],
    architecture: dict[str, Any],
    state_dict: dict[str, Any],
    preprocessing: dict[str, Any] | None = None,
    recurrent: bool = False,
    reset_on: list[str] | None = None,
    modes: list[str] | None = None,
    lineage: dict[str, Any] | None = None,
    reference_cases: dict[str, Any] | None = None,
) -> None:
    """Write bundle contents into an existing staging directory (not published yet)."""
    torch, _, _ = _require_torch()
    payloads = staging / "payloads"
    payloads.mkdir(parents=True, exist_ok=True)
    weights_path = payloads / "weights.pt"
    torch.save(state_dict, weights_path)
    weights_digest = digest_uri(sha256_file(weights_path))

    prep = {
        "included": True,
        "id": (preprocessing or {}).get("id", "normalize_v0"),
        "mean": (preprocessing or {}).get("mean", 0.0),
        "std": (preprocessing or {}).get("std", 1.0),
    }
    if preprocessing and preprocessing.get("clip") is not None:
        prep["clip"] = preprocessing["clip"]

    observation = normalize_space_descriptor(observation)
    action = normalize_space_descriptor(action)
    action.setdefault("masks", "none")

    # None → defaults; explicit [] must stay empty (do not coerce via `or`).
    if reset_on is None:
        resolved_reset_on = (
            ["episode_start", "agent_termination"] if recurrent else []
        )
    else:
        resolved_reset_on = list(reset_on)

    validate_architecture_spaces(
        observation=observation,
        action=action,
        architecture=architecture,
        adapter=ADAPTER_NAME,
    )

    manifest: dict[str, Any] = {
        "schema": POLICY_SCHEMA,
        "name": name,
        "roles": {"allowed": list(roles)},
        "runtime": {"adapter": ADAPTER_NAME, "python": "3.12"},
        "observation": observation,
        "action": action,
        "state": {
            "recurrent": bool(recurrent),
            "reset_on": resolved_reset_on,
        },
        "inference": {"modes": list(modes or ["deterministic", "stochastic"])},
        "preprocessing": prep,
        "architecture": architecture,
        "payloads": {
            "weights": {"digest": weights_digest, "path": "payloads/weights.pt"},
        },
    }
    if lineage:
        manifest["lineage"] = lineage

    if reference_cases is not None:
        case_list = (
            reference_cases["cases"]
            if isinstance(reference_cases, dict) and "cases" in reference_cases
            else reference_cases
        )
        if isinstance(case_list, list):
            for i, case in enumerate(case_list):
                if isinstance(case, dict):
                    validate_reference_case_action(case, action=action, index=i)
        cases_path = payloads / "reference_cases.json"
        cases_path.write_text(json.dumps(reference_cases, indent=2), encoding="utf-8")
        manifest["payloads"]["reference_cases"] = {
            "digest": digest_uri(sha256_file(cases_path)),
            "path": "payloads/reference_cases.json",
        }

    manifest["conformance"] = {"status": "unverified"}
    validate_policy_manifest(manifest)
    dump_yaml(manifest, staging / "policy.yaml")
    (staging / "DIGEST").write_text(policy_content_digest(manifest) + "\n", encoding="utf-8")


def _publish_staging(staging: Path, out_dir: Path) -> Path:
    """Atomically publish a staging directory to ``out_dir`` (no partial final bundle)."""
    out_dir = Path(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    staging.replace(out_dir)
    return out_dir


def export_policy(
    *,
    out_dir: Path | str,
    name: str,
    roles: list[str],
    observation: dict[str, Any],
    action: dict[str, Any],
    architecture: dict[str, Any],
    state_dict: dict[str, Any],
    preprocessing: dict[str, Any] | None = None,
    recurrent: bool = False,
    reset_on: list[str] | None = None,
    modes: list[str] | None = None,
    lineage: dict[str, Any] | None = None,
    reference_cases: dict[str, Any] | None = None,
) -> Path:
    """Write a portable policy bundle directory atomically (no partial bundle on failure)."""
    _require_torch()
    out_dir = Path(out_dir)
    parent = out_dir.parent if str(out_dir.parent) else Path(".")
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".arena-export-", dir=str(parent)))
    try:
        _write_bundle_into(
            staging,
            name=name,
            roles=roles,
            observation=observation,
            action=action,
            architecture=architecture,
            state_dict=state_dict,
            preprocessing=preprocessing,
            recurrent=recurrent,
            reset_on=reset_on,
            modes=modes,
            lineage=lineage,
            reference_cases=reference_cases,
        )
        return _publish_staging(staging, out_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def export_module_policy(
    *,
    out_dir: Path | str,
    name: str,
    roles: list[str],
    module: Any,
    observation: dict[str, Any],
    action: dict[str, Any],
    io: dict[str, Any] | None = None,
    preprocessing: dict[str, Any] | None = None,
    reference_cases: list[dict[str, Any]] | None = None,
    source_act_fn: Any | None = None,
    allow_trace: bool = False,
    lineage: dict[str, Any] | None = None,
) -> Path:
    """Export a BYO ``nn.Module`` as a trainer-free TorchScript actor.

    The actor contract is explicit: ``forward(obs[, hidden][, action_mask])`` and
    returns actions/logits (and optional next hidden). Trace is opt-in because it
    freezes Python control flow; callers must provide representative examples.

    ``lineage`` may record exporter-side provenance (source revision, checkpoint
    digest, wrapper identity notes). It is stored on the manifest but excluded
    from the content digest so receiver identity stays content-addressed.
    """
    torch, _, _ = _require_torch()
    out_dir = Path(out_dir)
    io = dict(io or {})
    architecture = {"type": "serialized_module", "io": io}
    action = dict(action)
    action.setdefault("masks", "none")
    validate_architecture_spaces(
        observation=observation, action=action, architecture=architecture, adapter=ADAPTER_NAME
    )
    cases = list(reference_cases or [])
    if not cases:
        raise SchemaError(
            "BYO module export requires source-captured reference_cases; refusing "
            "to mint unverifiable random/self cases"
        )
    parent = out_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".arena-export-", dir=str(parent)))
    try:
        module.eval()
        tier = "torchscript-script"
        try:
            portable = torch.jit.script(module)
        except Exception as script_error:
            if not allow_trace:
                raise SchemaError(
                    "TorchScript scripting failed. Refactor this actor to an explicit "
                    "tensor-only forward contract that torch.jit.script can preserve. "
                    "Trace is available only with allow_trace=True plus representative "
                    "cases, but is unsafe for Python/data-dependent control flow. "
                    "torch.export and bundled Python source are deliberately not "
                    "runtime tiers in this release, so Arena will not silently ship "
                    "trainer code. Script error: "
                    f"{script_error}"
                ) from script_error
            first = cases[0]
            x = np.asarray(first["observation"], dtype=np.float32)
            x = PreprocessPipeline(canonical_pipeline(preprocessing, observation), observation)(x)
            args: list[Any] = [torch.as_tensor(x).unsqueeze(0)]
            if io.get("recurrent"):
                shape = io.get("hidden_shape")
                if not shape:
                    raise SchemaError("traced recurrent actor requires io.hidden_shape")
                args.append(torch.zeros(*shape))
            if io.get("mask_in_graph"):
                mask = first.get("action_mask")
                if mask is None:
                    raise SchemaError("traced masked actor requires action_mask in reference cases")
                args.append(torch.as_tensor(mask, dtype=torch.bool).view(1, -1))
            portable = torch.jit.trace(module, tuple(args), check_trace=True)
            tier = "torchscript-trace"
        payloads = staging / "payloads"
        payloads.mkdir()
        model_path = payloads / "model.pt"
        torch.jit.save(portable, str(model_path))
        prep = canonical_pipeline(preprocessing, observation)
        prep_path = payloads / "preprocess.json"
        prep_path.write_text(json.dumps(prep, sort_keys=True), encoding="utf-8")
        provenance = PROVENANCE_SOURCE if source_act_fn is not None else PROVENANCE_SELF
        if source_act_fn is not None:
            captured: list[dict[str, Any]] = []
            for case in cases:
                case = dict(case)
                if "seed" in case and "rng" not in case:
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
        manifest = {
            "schema": POLICY_SCHEMA,
            "name": name,
            "roles": {"allowed": roles},
            "runtime": {
                "adapter": ADAPTER_NAME,
                "tier": "torchscript",
                "serialization": tier,
                "torch": str(torch.__version__),
                "python": "3.12",
            },
            "observation": observation,
            "action": action,
            "state": {"recurrent": bool(io.get("recurrent")), "reset_on": io.get("reset_on", [])},
            "inference": {
                "modes": (
                    ["deterministic"]
                    if action_type(action) == "Box"
                    and box_distribution(action) == "deterministic"
                    else ["deterministic", "stochastic"]
                )
            },
            "preprocessing": prep,
            "architecture": architecture,
            "payloads": {
                "model": {"path": "payloads/model.pt", "digest": digest_uri(sha256_file(model_path))},
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


def load_runtime(bundle: Path | str, *, trust_source: bool = False) -> TorchPolicyRuntime:
    from arena.core.manifests import load_manifest
    from arena.core.registry import PAYLOAD_LOADERS, ensure_plugins_loaded

    ensure_plugins_loaded()
    bundle = Path(bundle)
    manifest_path = bundle / "policy.yaml" if bundle.is_dir() else bundle
    root = manifest_path.parent
    # Every declared payload contributes to policy behavior or its evidence.
    # Authenticate all of them before loading any executable artifact.
    verify_bundle_integrity(root)
    manifest = load_manifest(manifest_path)
    tier = str(manifest.get("runtime", {}).get("tier", "template"))
    payload = PAYLOAD_LOADERS.get(tier)
    # Prefer the primary executable payload key for the constructor path arg.
    key = payload.required_payload_keys()[0]
    entry = manifest["payloads"].get(key)
    if not isinstance(entry, dict) or "path" not in entry:
        raise SchemaError(f"policy payloads.{key}.path is required")
    return TorchPolicyRuntime(
        manifest,
        _bundle_payload_path(root, entry["path"]),
        bundle_root=root,
        trust_source=trust_source,
    )


def verify_bundle_integrity(bundle: Path | str) -> dict[str, Any]:
    """Verify every declared payload digest in a bundle against its bytes on disk.

    Returns a summary dict. Raises ConformanceError on the first mismatch so a
    tampered/corrupted bundle is never silently accepted.
    """
    from arena.core.manifests import load_manifest

    bundle = Path(bundle)
    manifest_path = bundle / "policy.yaml" if bundle.is_dir() else bundle
    root = manifest_path.parent
    manifest = load_manifest(manifest_path)
    checked: list[str] = []
    for key, entry in manifest.get("payloads", {}).items():
        if not isinstance(entry, dict) or "digest" not in entry or "path" not in entry:
            continue
        payload_path = _bundle_payload_path(root, entry["path"])
        if not payload_path.exists():
            raise ConformanceError(f"payload {key!r} missing on disk: {payload_path}")
        actual = digest_uri(sha256_file(payload_path))
        if actual != entry["digest"]:
            raise ConformanceError(
                f"payload {key!r} integrity check failed: declared {entry['digest']}, "
                f"actual {actual} ({payload_path})"
            )
        checked.append(key)
    return {"ok": True, "payloads_checked": checked}


def _is_tensor_like(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "device")


def _looks_like_state_dict(obj: Any) -> bool:
    """True when ``obj`` is a mapping of parameter-name → tensor (or nested tensors)."""
    if not isinstance(obj, dict) or not obj:
        return False
    if not all(isinstance(k, str) for k in obj):
        return False
    for value in obj.values():
        if _is_tensor_like(value):
            continue
        if isinstance(value, dict) and _looks_like_state_dict(value):
            continue
        return False
    return True


def _extract_state_dict(
    obj: Any,
    source: Path,
    *,
    prefer_ema: bool = True,
) -> dict[str, Any]:
    """Pull a tensor state_dict out of a checkpoint object with actionable errors.

    EMA is the default whenever a usable ``ema_state_dict`` is present. Base
    weights require explicit ``prefer_ema=False``.
    """
    if not isinstance(obj, dict):
        raise SchemaError(
            f"unsupported checkpoint format at {source}: expected a dict/state_dict, "
            f"got {type(obj).__name__}. Save torch tensors via "
            f"torch.save(model.state_dict(), path) or a dict with a 'state_dict'/'model' key."
        )

    ema = obj.get("ema_state_dict")
    has_ema = isinstance(ema, dict) and _looks_like_state_dict(ema)
    if prefer_ema:
        if has_ema:
            return ema

    for key in ("state_dict", "model", "policy", "actor", "network", "model_state_dict"):
        nested = obj.get(key)
        if isinstance(nested, dict) and _looks_like_state_dict(nested):
            if has_ema:
                warnings.warn(
                    f"checkpoint at {source} contains ema_state_dict but base weights "
                    f"({key!r}) were explicitly selected.",
                    stacklevel=2,
                )
            return nested

    if has_ema:
        # No plain state_dict — EMA is the only usable weight blob.
        warnings.warn(
            f"checkpoint at {source} has ema_state_dict but no plain state_dict; "
            "using ema_state_dict.",
            stacklevel=2,
        )
        return ema

    if _looks_like_state_dict(obj):
        return obj

    keys = sorted(str(k) for k in obj.keys())
    trainingish = [k for k in keys if k in _TRAINING_CHECKPOINT_KEYS or k.endswith("state_dict")]
    hint_keys = ", ".join(keys[:12]) + ("…" if len(keys) > 12 else "")
    if trainingish or any(isinstance(obj.get(k), dict) for k in obj):
        raise SchemaError(
            f"checkpoint at {source} looks like a training checkpoint dict "
            f"(top-level keys: [{hint_keys}]), not a raw state_dict. "
            f"Extract the weight tensors (e.g. torch.save(ckpt['state_dict'], path) "
            f"or ckpt['model']) and re-export. Common weight keys: "
            f"state_dict, model, policy, actor, network, ema_state_dict."
        )
    raise SchemaError(
        f"unsupported checkpoint format at {source}: dict keys [{hint_keys}] do not "
        f"contain tensor weights. Provide a state_dict of tensors."
    )


def load_checkpoint_file(
    source: Path | str,
    *,
    allow_unsafe_checkpoint: bool = False,
) -> Any:
    """Load a checkpoint with the pickle safety boundary first.

    Attempts ``weights_only=True`` (no arbitrary pickle globals). Falls back to an
    unsafe load only when ``allow_unsafe_checkpoint=True`` and emits a warning.
    """
    torch, _, _ = _require_torch()
    source = Path(source)
    try:
        return torch.load(source, map_location="cpu", weights_only=True)
    except Exception as safe_err:
        if not allow_unsafe_checkpoint:
            raise SchemaError(
                f"refusing to load checkpoint {source} with unsafe pickle execution "
                f"({type(safe_err).__name__}: {safe_err}). "
                f"Re-save as a tensor-only state_dict "
                f"(torch.save(model.state_dict(), path)), or pass "
                f"allow_unsafe_checkpoint=True / --allow-unsafe-checkpoint to opt in "
                f"(trusted sources only)."
            ) from safe_err
        warnings.warn(
            f"Loading {source} with weights_only=False (unsafe pickle). "
            "Only use this for checkpoints you fully trust.",
            stacklevel=2,
        )
        # Explicit trusted-source opt-in; the safe weights-only path is always first.
        return torch.load(  # nosec B614
            source, map_location="cpu", weights_only=False
        )


def _jsonable_action(action: Any) -> Any:
    if isinstance(action, np.ndarray):
        return action.tolist()
    if isinstance(action, (np.integer,)):
        return int(action)
    if isinstance(action, (np.floating,)):
        return float(action)
    if isinstance(action, dict):
        return {k: _jsonable_action(v) for k, v in action.items()}
    if isinstance(action, (list, tuple)):
        return [_jsonable_action(v) for v in action]
    return action


def generate_reference_cases(
    runtime: TorchPolicyRuntime,
    *,
    observation: dict[str, Any],
    action: dict[str, Any],
    n_deterministic: int = 8,
    stochastic_seeds: tuple[int, ...] = (7, 11, 19),
    agent_id: str = "default",
) -> list[dict[str, Any]]:
    """Capture reference cases for later ``arena policy verify``.

    Logits are taken from the same ``act()`` forward pass (via ``last_logits``) so
    recurrent policies are not double-stepped. When the runtime is recurrent, cases
    carry hidden state across steps and also cross an explicit reset boundary so
    verify exercises real RNN behavior (not only per-step resets).
    """
    obs_type = observation.get("type")
    action_n = int(action.get("n", runtime.manifest["architecture"].get("action_n", 0)) or 0)
    masks_mode = action.get("masks", "none")
    masks_needed = masks_mode in {"required", "optional"}
    rng = np.random.default_rng(0)

    def _sample_obs(i: int) -> Any:
        if obs_type == "Discrete":
            n = int(observation.get("n", 1))
            return int(i % max(n, 1))
        shape = observation.get("shape") or [
            int(runtime.manifest["architecture"]["observation_dim"])
        ]
        size = int(np.prod(shape))
        return rng.normal(size=size).astype(np.float32).tolist()

    def _mask_for(i: int, *, force: bool = False) -> list[bool] | None:
        if not action_n:
            return None
        if masks_mode == "required" or force:
            m = np.zeros(action_n, dtype=bool)
            m[i % action_n] = True
            m[(i + 1) % action_n] = True
            return m.tolist()
        if masks_mode == "optional" and (force or i % 2 == 0):
            m = np.zeros(action_n, dtype=bool)
            m[i % action_n] = True
            m[(i + 1) % action_n] = True
            return m.tolist()
        return None

    def _record(
        *,
        obs: Any,
        mode: str,
        hidden_reset: bool,
        mask: list[bool] | None,
        seed: int | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        mask_arr = np.asarray(mask) if mask is not None else None
        if hidden_reset:
            runtime.reset(agent_id)
        if mode == "stochastic":
            assert seed is not None
            act = runtime.act(
                obs, mode="stochastic", rng=np.random.default_rng(seed),
                action_mask=mask_arr, agent_id=agent_id,
            )
        else:
            act = runtime.act(
                obs, mode="deterministic", action_mask=mask_arr, agent_id=agent_id
            )
        case: dict[str, Any] = {
            "observation": obs,
            "mode": mode,
            "expected_action": _jsonable_action(act),
            "hidden_reset": hidden_reset,
            "agent_id": agent_id,
        }
        if runtime.last_logits is not None and mode == "deterministic":
            case["expected_logits"] = runtime.last_logits.tolist()
        if mask is not None:
            case["action_mask"] = mask
        if seed is not None:
            case["seed"] = int(seed)
        if note is not None:
            case["note"] = note
        return case

    cases: list[dict[str, Any]] = []

    if runtime.recurrent:
        # Carry hidden across a multi-step stream (this is what verify must exercise).
        stream_len = max(n_deterministic, 4)
        for i in range(stream_len):
            cases.append(
                _record(
                    obs=_sample_obs(i),
                    mode="deterministic",
                    hidden_reset=(i == 0),
                    mask=_mask_for(i),
                    note="recurrent_carry" if i > 0 else "recurrent_start",
                )
            )
        # Cross a reset boundary: after carry, reset and confirm fresh-start behavior.
        cases.append(
            _record(
                obs=_sample_obs(0),
                mode="deterministic",
                hidden_reset=True,
                mask=_mask_for(0),
                note="recurrent_reset_boundary",
            )
        )
        cases.append(
            _record(
                obs=_sample_obs(1),
                mode="deterministic",
                hidden_reset=False,
                mask=_mask_for(1),
                note="recurrent_after_reset",
            )
        )
    else:
        for i in range(n_deterministic):
            cases.append(
                _record(
                    obs=_sample_obs(i),
                    mode="deterministic",
                    hidden_reset=True,
                    mask=_mask_for(i),
                )
            )

    if "stochastic" in runtime.manifest["inference"].get("modes", []):
        for seed in stochastic_seeds:
            cases.append(
                _record(
                    obs=_sample_obs(seed),
                    mode="stochastic",
                    hidden_reset=True,
                    mask=_mask_for(seed, force=masks_mode == "required"),
                    seed=int(seed),
                )
            )

    # Guaranteed masked coverage when masks are required or optional.
    if masks_needed and action_n and not any(c.get("action_mask") is not None for c in cases):
        cases.append(
            _record(
                obs=_sample_obs(0),
                mode="deterministic",
                hidden_reset=True,
                mask=_mask_for(0, force=True),
                note="masked_path",
            )
        )
    return cases


def pack_reference_cases(
    cases: list[dict[str, Any]],
    *,
    provenance: str = PROVENANCE_SOURCE,
) -> dict[str, Any]:
    """Wrap cases with an explicit verify-mode provenance label."""
    if provenance not in {PROVENANCE_SELF, PROVENANCE_SOURCE}:
        raise SchemaError(f"unknown reference-case provenance: {provenance!r}")
    return {"provenance": provenance, "cases": cases}


def export_from_checkpoint(
    *,
    source: Path | str,
    out: Path | str,
    role: str,
    architecture: dict[str, Any],
    observation: dict[str, Any],
    action: dict[str, Any],
    name: str | None = None,
    preprocessing: dict[str, Any] | None = None,
    recurrent: bool = False,
    entrypoint_module: Any | None = None,
    make_reference_cases: bool = True,
    allow_unsafe_checkpoint: bool = False,
    prefer_ema: bool = True,
    source_runtime: TorchPolicyRuntime | None = None,
) -> Path:
    """Export from a torch checkpoint into a portable bundle (atomic on failure).

    Checkpoint loading tries ``weights_only=True`` first (fail closed). Pass
    ``allow_unsafe_checkpoint=True`` only for trusted legacy pickles.

    A usable ``ema_state_dict`` is selected by default. Set ``prefer_ema=False``
    only to explicitly select plain/base weights.

    Reference cases are captured from a source runtime (caller-supplied
    ``source_runtime``, or a freshly built module loaded with the checkpoint
    weights) *before* the final bundle is published, and labeled
    ``source-conformance``. That is stronger than replaying the exported bundle
    against itself (self-consistency), but still not a substitute for trainer-side
    golden cases when the trainer's forward path differs from this adapter.
    """
    torch, _, _ = _require_torch()
    source = Path(source)
    out = Path(out)
    parent = out.parent if str(out.parent) else Path(".")
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".arena-export-", dir=str(parent)))

    try:
        obj = load_checkpoint_file(source, allow_unsafe_checkpoint=allow_unsafe_checkpoint)
        state_dict = _extract_state_dict(obj, source, prefer_ema=prefer_ema)

        if entrypoint_module is not None:
            pass

        _write_bundle_into(
            staging,
            name=name or f"{role}-policy",
            roles=[role],
            observation=observation,
            action=action,
            architecture=architecture,
            state_dict=state_dict,
            preprocessing=preprocessing,
            recurrent=recurrent,
            lineage={
                "source_checkpoint": source.name,
                "weight_selection": (
                    "ema_state_dict"
                    if prefer_ema
                    and isinstance(obj, dict)
                    and _looks_like_state_dict(obj.get("ema_state_dict"))
                    else "base_state_dict"
                ),
            },
        )

        if make_reference_cases:
            if source_runtime is not None:
                case_runtime = source_runtime
                provenance = PROVENANCE_SOURCE
            else:
                # Template policies have a fixed, audited source implementation.
                # Capture from that source runtime before publishing the final bundle.
                case_runtime = load_runtime(staging)
                provenance = PROVENANCE_SOURCE
            cases = generate_reference_cases(
                case_runtime, observation=observation, action=action
            )
            _embed_reference_cases(
                staging, cases, provenance=provenance, rewrite_digest=True
            )

        return _publish_staging(staging, out)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _embed_reference_cases(
    bundle: Path | str,
    cases: list[dict[str, Any]],
    *,
    provenance: str = PROVENANCE_SELF,
    rewrite_digest: bool = True,
) -> None:
    from arena.core.manifests import dump_yaml, load_manifest

    bundle = Path(bundle)
    manifest = load_manifest(bundle / "policy.yaml")
    action = manifest.get("action") or {}
    for i, case in enumerate(cases):
        validate_reference_case_action(case, action=action, index=i)
    path = bundle / "payloads" / "reference_cases.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = pack_reference_cases(cases, provenance=provenance)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    manifest["payloads"]["reference_cases"] = {
        "digest": digest_uri(sha256_file(path)),
        "path": "payloads/reference_cases.json",
    }
    dump_yaml(manifest, bundle / "policy.yaml")
    if rewrite_digest:
        (bundle / "DIGEST").write_text(
            policy_content_digest(manifest) + "\n", encoding="utf-8"
        )


def verify_against_source(
    bundle: Path | str,
    source_act_fn,
    cases: list[dict[str, Any]],
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    """Compare exported runtime to a source callable on reference cases.

    Each case: {observation, mode, seed?, action_mask?, agent_id?, hidden_reset?}
    source_act_fn(case) -> action (and optionally logits via source_logits_fn)

    This is true ``source-conformance`` (external callable vs export).
    """
    del atol, rtol  # reserved for optional logits compare by callers
    runtime = load_runtime(bundle)
    failures: list[dict[str, Any]] = []
    for i, case in enumerate(cases):
        agent = case.get("agent_id", "default")
        if case.get("hidden_reset", True):
            runtime.reset(agent)
        rng = np.random.default_rng(case["seed"]) if "seed" in case else None
        mode = case.get("mode", "deterministic")
        mask = case.get("action_mask")
        obs = case["observation"]
        exported = runtime.act(obs, mode=mode, action_mask=mask, rng=rng, agent_id=agent)
        rng2 = np.random.default_rng(case["seed"]) if "seed" in case else None
        expected = source_act_fn({**case, "rng": rng2})
        if not actions_equal(exported, expected, action=runtime.manifest["action"]):
            failures.append(
                {
                    "index": i,
                    "exported": exported if not hasattr(exported, "tolist") else exported.tolist(),
                    "expected": expected if not hasattr(expected, "tolist") else expected.tolist(),
                    "case": {k: v for k, v in case.items() if k != "observation"},
                }
            )
    if failures:
        raise ConformanceError(f"verify failed on {len(failures)} case(s): {failures[0]}")
    return {
        "ok": True,
        "cases": len(cases),
        "verify_mode": PROVENANCE_SOURCE,
    }


def verify_bundle_self(
    bundle: Path | str,
    cases_path: Path | str | None = None,
    *,
    allow_self_consistency: bool = False,
    trust_source: bool = False,
) -> dict[str, Any]:
    """Verify a bundle against embedded/provided reference cases.

    Returns ``verify_mode`` of ``source-conformance`` or ``self-consistency``.
    Self-consistency only proves the bundle replays its own embedded cases (or
    cases generated from the export path) — it does **not** by itself prove the
    export matches an external trainer. When mode is self-consistency, a
    ``warning`` field is included.

    ``trust_source`` is required to load ``runtime.tier=trusted_source`` bundles
    (not sandboxed; prefer TorchScript).
    """
    bundle = Path(bundle)
    runtime = load_runtime(bundle, trust_source=trust_source)
    if cases_path is None:
        cases_path = bundle / "payloads" / "reference_cases.json"
    cases_path = Path(cases_path)
    if not cases_path.exists():
        raise SchemaError(f"reference cases not found: {cases_path}")
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    provenance = PROVENANCE_SELF
    if isinstance(cases, dict) and "cases" in cases:
        case_list = cases["cases"]
        provenance = str(cases.get("provenance") or PROVENANCE_SELF)
    else:
        case_list = cases
    # External --source-test files without provenance are treated as source cases.
    if cases_path.resolve() != (bundle / "payloads" / "reference_cases.json").resolve():
        if isinstance(cases, dict) and "provenance" not in cases:
            provenance = PROVENANCE_SOURCE
        elif not isinstance(cases, dict):
            provenance = PROVENANCE_SOURCE

    action_contract = runtime.manifest["action"]
    # Reject illegal expected_action before stamping verified — even if the
    # runtime would coincidentally emit the same illegal index under a desynced head.
    for i, case in enumerate(case_list):
        validate_reference_case_action(case, action=action_contract, index=i)

    failures = []
    for i, case in enumerate(case_list):
        agent = case.get("agent_id", "default")
        if case.get("hidden_reset", True):
            runtime.reset(agent)
        rng = np.random.default_rng(case["seed"]) if "seed" in case else None
        got = runtime.act(
            case["observation"],
            mode=case.get("mode", "deterministic"),
            action_mask=(
                np.asarray(case["action_mask"]) if case.get("action_mask") is not None else None
            ),
            rng=rng,
            agent_id=agent,
        )
        expected_action = case.get("expected_action")
        if expected_action is not None and not actions_equal(
            got, expected_action, action=runtime.manifest["action"]
        ):
            failures.append(
                {
                    "index": i,
                    "got": got if not hasattr(got, "tolist") else got.tolist(),
                    "expected": case["expected_action"],
                }
            )
        if "expected_logits" in case:
            # Use logits from the same act() forward — never re-step a recurrent net.
            if runtime.last_logits is None:
                failures.append({"index": i, "logits_mismatch": True, "reason": "missing_last_logits"})
            else:
                exp = np.asarray(case["expected_logits"], dtype=np.float32)
                if not np.allclose(runtime.last_logits, exp, atol=1e-5, rtol=1e-5):
                    failures.append(
                        {
                            "index": i,
                            "logits_mismatch": True,
                            "max_abs_err": float(
                                np.max(np.abs(runtime.last_logits - exp))
                            ),
                        }
                    )
    if failures:
        raise ConformanceError(f"self-verify failed: {failures[0]}")

    if provenance != PROVENANCE_SOURCE and not allow_self_consistency:
        raise ConformanceError(
            "verify has insufficient evidence: these cases are self-consistency only, "
            "not source-captured conformance. Provide source-captured cases or pass "
            "--allow-self-consistency to acknowledge the weaker result."
        )

    # Masked-path honesty: when masks are required/optional, at least one case
    # must exercise an action_mask (otherwise verify never covers masked logits).
    masks_mode = action_contract.get("masks", "none")
    if masks_mode in {"required", "optional"}:
        if not any(c.get("action_mask") is not None for c in case_list):
            raise ConformanceError(
                f"verify refused: action.masks={masks_mode!r} but no reference case "
                "includes action_mask. Regenerate cases (export) or add a masked case."
            )

    manifest_path = bundle / "policy.yaml"
    from arena.core.manifests import dump_yaml, load_manifest

    manifest = load_manifest(manifest_path)
    manifest["conformance"] = {
        "status": "verified" if provenance == PROVENANCE_SOURCE else "self-consistency-only",
        "cases": len(case_list),
        "verify_mode": provenance,
    }
    dump_yaml(manifest, manifest_path)
    result: dict[str, Any] = {
        "ok": True,
        "cases": len(case_list),
        "verify_mode": provenance,
    }
    if provenance != PROVENANCE_SOURCE:
        result["warning"] = (
            "Only self-consistency was checked. Embedded/reference cases were not "
            "labeled as source-conformance, so this does not prove the export matches "
            "an external trainer. Capture cases from the source model at export time "
            "(default for `arena policy export`) or pass --source-test cases from the "
            "trainer for source-conformance."
        )
    return result


def resolve_export_factory(module_ref: str) -> Any:
    """Import ``pkg.mod:attr`` for BYO TorchScript export (exporter-side only).

    The receiver never imports this path — only the TorchScript payload is shipped.
    ``attr`` may be an ``nn.Module`` subclass (called with kwargs) or a factory
    callable returning a module instance.
    """
    import importlib

    if ":" not in module_ref:
        raise SchemaError(
            f"--module must be 'package.module:factory' (got {module_ref!r})"
        )
    mod_path, attr = module_ref.rsplit(":", 1)
    if not mod_path or not attr:
        raise SchemaError(f"invalid --module reference {module_ref!r}")
    try:
        mod = importlib.import_module(mod_path)
    except ImportError as e:
        raise SchemaError(
            f"cannot import module {mod_path!r} for BYO export: {e}. "
            "This path is exporter-side only; the receiver loads TorchScript."
        ) from e
    if not hasattr(mod, attr):
        raise SchemaError(f"module {mod_path!r} has no attribute {attr!r}")
    return getattr(mod, attr)


def build_module_for_export(
    factory: Any,
    *,
    module_args: dict[str, Any] | None = None,
) -> Any:
    """Instantiate a BYO actor from a class or zero/kwargs factory."""
    torch, _, _ = _require_torch()
    kwargs = dict(module_args or {})
    if isinstance(factory, type) and issubclass(factory, torch.nn.Module):
        module = factory(**kwargs)
    elif callable(factory):
        module = factory(**kwargs) if kwargs else factory()
    else:
        raise SchemaError(
            "BYO --module target must be an nn.Module subclass or a callable factory"
        )
    if not isinstance(module, torch.nn.Module):
        raise SchemaError(
            f"BYO factory returned {type(module).__name__}, expected torch.nn.Module"
        )
    return module


def export_module_from_checkpoint(
    *,
    module_ref: str,
    out_dir: Path | str,
    role: str,
    observation: dict[str, Any],
    action: dict[str, Any],
    source: Path | str | None = None,
    name: str | None = None,
    io: dict[str, Any] | None = None,
    preprocessing: dict[str, Any] | None = None,
    reference_cases: list[dict[str, Any]] | None = None,
    module_args: dict[str, Any] | None = None,
    allow_trace: bool = False,
    allow_unsafe_checkpoint: bool = False,
    prefer_ema: bool = True,
    source_revision: str | None = None,
    wrappers_identity: str | None = None,
    roles: list[str] | None = None,
) -> Path:
    """CLI/SDK helper: resolve factory → optional weights → TorchScript bundle.

    Source-captured reference cases are required. When cases lack expected
    actions, they are filled by running the live (pre-script) module through the
    declared preprocess pipeline so verify proves trainer-side fidelity.
    """
    torch, _, _ = _require_torch()
    factory = resolve_export_factory(module_ref)
    module = build_module_for_export(factory, module_args=module_args)

    lineage: dict[str, Any] = {
        "export_path": "byo-torchscript",
        "module_ref": module_ref,
    }
    if source_revision:
        lineage["source_revision"] = source_revision
    if wrappers_identity:
        lineage["wrappers_identity"] = wrappers_identity
    if module_args:
        lineage["module_args"] = module_args

    if source is not None:
        source_path = Path(source)
        obj = load_checkpoint_file(source_path, allow_unsafe_checkpoint=allow_unsafe_checkpoint)
        state_dict = _extract_state_dict(obj, source_path, prefer_ema=prefer_ema)
        missing, unexpected = module.load_state_dict(state_dict, strict=True)
        if missing or unexpected:  # pragma: no cover — strict=True raises before this
            raise SchemaError(
                f"state_dict mismatch loading {source_path}: "
                f"missing={list(missing)} unexpected={list(unexpected)}"
            )
        lineage["source_checkpoint"] = source_path.name
        lineage["checkpoint_digest"] = digest_uri(sha256_file(source_path))
        lineage["weight_selection"] = (
            "ema_state_dict"
            if prefer_ema
            and isinstance(obj, dict)
            and _looks_like_state_dict(obj.get("ema_state_dict"))
            else "base_state_dict"
        )

    cases = list(reference_cases or [])
    if not cases:
        raise SchemaError(
            "BYO TorchScript export requires --reference-cases (JSON list of "
            "observation cases). Refusing to mint unverifiable random cases."
        )

    prep = canonical_pipeline(preprocessing, observation)
    module.eval()

    def _source_act(case: dict[str, Any]) -> tuple[Any, list[float]]:
        raw = case["observation"]
        x = PreprocessPipeline(prep, observation)(raw)
        xt = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)
        args: list[Any] = [xt]
        io_cfg = dict(io or {})
        if io_cfg.get("recurrent"):
            shape = io_cfg.get("hidden_shape")
            if not shape:
                raise SchemaError("recurrent BYO export requires io.hidden_shape")
            args.append(torch.zeros(*shape))
        if io_cfg.get("mask_in_graph"):
            mask = case.get("action_mask")
            if mask is None:
                raise SchemaError("masked BYO actor requires action_mask in reference cases")
            args.append(torch.as_tensor(mask, dtype=torch.bool).view(1, -1))
        with torch.no_grad():
            out = module(*args)
            logits = out[0] if isinstance(out, tuple) else out
            logits_np = logits.detach().cpu().numpy().reshape(-1)
        if action.get("type") == "Box":
            return logits_np.tolist(), logits_np.tolist()
        return int(np.argmax(logits_np)), logits_np.tolist()

    # Always capture expected actions/logits from the live pre-script module so
    # verify is source-conformance, not a self-replay of the TorchScript archive.
    return export_module_policy(
        out_dir=out_dir,
        name=name or f"{role}-policy",
        roles=list(roles) if roles else [role],
        module=module,
        observation=observation,
        action=action,
        io=io,
        preprocessing=preprocessing,
        reference_cases=cases,
        source_act_fn=_source_act,
        allow_trace=allow_trace,
        lineage=lineage,
    )
