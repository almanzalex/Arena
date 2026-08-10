"""Small, reproducible training recipes over Arena trajectory datasets."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from arena.adapters.policy_custom_torch import build_module, export_policy
from arena.core.dataset import dataset_content_digest
from arena.core.errors import missing_extra, ArenaError, ConformanceError, SchemaError
from arena.core.identity import canonical_json, digest_uri, sha256_bytes, sha256_file
from arena.core.manifests import dump_json, dump_yaml, load_manifest, validate_dataset_manifest
from arena.core.sdk import Policy

TRAIN_RECIPE_SCHEMA = "arena.train/v1"
TRAIN_RUN_SCHEMA = "arena.train-run/v1"
TRAIN_CHECKPOINT_SCHEMA = "arena.train-checkpoint/v1"


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise missing_extra("torch", feature="arena train", capability="torch") from exc
    return torch


def validate_train_recipe(recipe: dict[str, Any]) -> dict[str, Any]:
    if recipe.get("schema") != TRAIN_RECIPE_SCHEMA:
        raise SchemaError(
            f"expected training recipe schema {TRAIN_RECIPE_SCHEMA}, got {recipe.get('schema')!r}"
        )
    for key in ("name", "algorithm", "dataset", "role", "observation", "action"):
        if key not in recipe:
            raise SchemaError(f"training recipe missing required field: {key}")
    for key in ("name", "algorithm", "dataset", "role"):
        if not isinstance(recipe[key], str) or not recipe[key].strip():
            raise SchemaError(f"training recipe {key} must be a non-empty string")
    if recipe.get("algorithm_config") is not None and not isinstance(
        recipe["algorithm_config"], dict
    ):
        raise SchemaError("training recipe algorithm_config must be a mapping")
    if recipe.get("dataset_split") is not None and (
        not isinstance(recipe["dataset_split"], str)
        or not recipe["dataset_split"].strip()
    ):
        raise SchemaError("training recipe dataset_split must be a non-empty string")
    if recipe.get("resume_from") is not None and (
        not isinstance(recipe["resume_from"], str)
        or not recipe["resume_from"].strip()
    ):
        raise SchemaError("training recipe resume_from must be a non-empty path string")
    from arena.core.registry import TRAINERS, ensure_plugins_loaded

    ensure_plugins_loaded()
    TRAINERS.get(str(recipe["algorithm"])).validate(recipe)
    return recipe


def _encode_observation(observation: Any, contract: dict[str, Any]) -> np.ndarray:
    if contract["type"] == "Discrete":
        n = int(contract["n"])
        if isinstance(observation, bool) or not isinstance(
            observation, (int, np.integer)
        ):
            raise ConformanceError(
                f"Discrete observation must be an integer, got {observation!r}"
            )
        index = int(observation)
        if index < 0 or index >= n:
            raise ConformanceError(f"Discrete observation {index} outside [0, {n})")
        encoded = np.zeros(n, dtype=np.float32)
        encoded[index] = 1.0
        return encoded
    value = np.asarray(observation, dtype=np.float32).reshape(-1)
    shape = tuple(int(item) for item in contract.get("shape") or [])
    expected = int(np.prod(shape)) if shape else value.size
    if value.size != expected:
        raise ConformanceError(
            f"Box observation has {value.size} elements, expected {expected} from {shape}"
        )
    if not np.all(np.isfinite(value)):
        raise ConformanceError("Box observation contains non-finite values")
    return value


def _apply_preprocess(value: np.ndarray, preprocessing: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(preprocessing.get("mean", 0.0), dtype=np.float32)
    std = np.asarray(preprocessing.get("std", 1.0), dtype=np.float32)
    if np.any(std <= 0):
        raise SchemaError("training preprocessing.std must be positive")
    result = (value - mean) / np.maximum(std, 1e-8)
    if preprocessing.get("clip") is not None:
        low, high = preprocessing["clip"]
        result = np.clip(result, float(low), float(high))
    return np.asarray(result, dtype=np.float32)


def _load_samples(
    dataset_path: Path,
    dataset: dict[str, Any],
    *,
    role: str,
    observation: dict[str, Any],
    dataset_split: str | None = None,
) -> tuple[list[Any], np.ndarray, np.ndarray, np.ndarray]:
    raw_observations: list[Any] = []
    encoded: list[np.ndarray] = []
    actions: list[int] = []
    episode_returns: list[float] = []
    for index, entry in enumerate(dataset["episodes"]):
        if not isinstance(entry, dict) or not entry.get("path") or not entry.get("digest"):
            raise SchemaError(f"dataset.episodes[{index}] requires path and digest")
        if dataset_split is not None and entry.get("split") != dataset_split:
            continue
        episode_path = Path(str(entry["path"]))
        if not episode_path.is_absolute():
            episode_path = (dataset_path.parent / episode_path).resolve()
        if not episode_path.is_file():
            raise SchemaError(f"training episode not found: {episode_path}")
        actual = digest_uri(sha256_file(episode_path))
        if actual != entry["digest"]:
            raise ConformanceError(
                f"training dataset mutation detected for {episode_path}: "
                f"declared {entry['digest']}, actual {actual}"
            )
        episode = json.loads(episode_path.read_text(encoding="utf-8"))
        episode_return = (episode.get("returns") or {}).get(role)
        if episode_return is None:
            episode_return = sum(
                float((step.get("rewards") or {}).get(role, 0.0))
                for step in episode.get("steps") or []
            )
        for step in episode.get("steps") or []:
            if role not in (step.get("observations") or {}) or role not in (step.get("actions") or {}):
                continue
            raw = step["observations"][role]
            raw_action = step["actions"][role]
            if isinstance(raw_action, bool) or not isinstance(
                raw_action, (int, np.integer)
            ):
                raise ConformanceError(
                    f"Discrete training action must be an integer, got {raw_action!r}"
                )
            action = int(raw_action)
            if not math.isfinite(float(episode_return)):
                raise ConformanceError(
                    f"episode return for role {role!r} must be finite"
                )
            raw_observations.append(raw)
            encoded.append(_encode_observation(raw, observation))
            actions.append(action)
            episode_returns.append(float(episode_return))
    if not encoded:
        suffix = f" in split {dataset_split!r}" if dataset_split is not None else ""
        raise SchemaError(
            f"dataset contains no observation/action samples for role {role!r}{suffix}"
        )
    return (
        raw_observations,
        np.stack(encoded),
        np.asarray(actions, dtype=np.int64),
        np.asarray(episode_returns, dtype=np.float32),
    )


def run_training_recipe(recipe_path: Path | str, *, out_dir: Path | str) -> dict[str, Any]:
    """Dispatch a versioned recipe through the registered trainer axis."""
    recipe_file = Path(recipe_path).resolve()
    recipe = validate_train_recipe(load_manifest(recipe_file))
    from arena.core.registry import TRAINERS, ensure_plugins_loaded

    ensure_plugins_loaded()
    trainer = TRAINERS.get(str(recipe["algorithm"]))
    return trainer.run(
        recipe,
        recipe_path=recipe_file,
        out_dir=Path(out_dir),
    )


def _sample_weights(
    signal: np.ndarray,
    *,
    weighting: str,
    algorithm_config: dict[str, Any],
) -> np.ndarray:
    if weighting == "uniform":
        return np.ones(len(signal), dtype=np.float32)
    if weighting != "episode_return":
        raise SchemaError(f"unknown built-in training weighting {weighting!r}")
    temperature = float(algorithm_config.get("temperature", 1.0))
    max_weight = float(algorithm_config.get("max_weight", 20.0))
    centered = (signal - float(np.max(signal))) / temperature
    weights = np.exp(np.clip(centered, -50.0, 0.0))
    weights = weights / max(float(np.mean(weights)), 1e-12)
    return np.clip(weights, 0.0, max_weight).astype(np.float32)


def _run_categorical_recipe(
    recipe: dict[str, Any],
    *,
    recipe_path: Path,
    out_dir: Path,
    weighting: str,
) -> dict[str, Any]:
    """Shared deterministic categorical engine used by qualified trainer cases."""
    torch = _require_torch()
    recipe_file = Path(recipe_path).resolve()
    dataset_path = Path(str(recipe["dataset"]))
    if not dataset_path.is_absolute():
        dataset_path = (recipe_file.parent / dataset_path).resolve()
    dataset = validate_dataset_manifest(load_manifest(dataset_path))
    dataset_digest = dataset_content_digest(dataset)
    if dataset.get("digest") is not None and dataset["digest"] != dataset_digest:
        raise ConformanceError(
            f"training dataset digest mismatch: declared {dataset['digest']}, actual {dataset_digest}"
        )

    observation = dict(recipe["observation"])
    action = dict(recipe["action"])
    action.setdefault("masks", "none")
    role = str(recipe["role"])
    dataset_split = (
        str(recipe["dataset_split"]) if recipe.get("dataset_split") is not None else None
    )
    raw, features, targets, return_signal = _load_samples(
        dataset_path,
        dataset,
        role=role,
        observation=observation,
        dataset_split=dataset_split,
    )
    if np.any(targets < 0) or np.any(targets >= int(action["n"])):
        bad = int(targets[(targets < 0) | (targets >= int(action["n"]))][0])
        raise ConformanceError(
            f"training action {bad} is outside declared Discrete range [0, {action['n']})"
        )
    preprocessing = dict(recipe.get("preprocessing") or {"id": "normalize_v0"})
    features = np.stack([_apply_preprocess(item, preprocessing) for item in features])
    if not np.all(np.isfinite(features)):
        raise ConformanceError("training features contain non-finite values")
    algorithm_config = dict(recipe.get("algorithm_config") or {})
    sample_weights = _sample_weights(
        return_signal,
        weighting=weighting,
        algorithm_config=algorithm_config,
    )

    architecture = dict(recipe.get("architecture") or {})
    architecture.setdefault("type", "mlp_categorical")
    architecture.setdefault("observation_dim", int(features.shape[1]))
    architecture.setdefault("hidden_dims", [32, 32])
    architecture.setdefault("action_n", int(action["n"]))
    if int(architecture["observation_dim"]) != int(features.shape[1]):
        raise SchemaError(
            f"architecture.observation_dim={architecture['observation_dim']} does not match "
            f"encoded dataset width {features.shape[1]}"
        )
    if int(architecture["action_n"]) != int(action["n"]):
        raise SchemaError("architecture.action_n must match action.n")

    seed = int(recipe.get("seed", 0))
    epochs = int(recipe.get("epochs", 25))
    batch_size = int(recipe.get("batch_size", 32))
    learning_rate = float(recipe.get("learning_rate", 1e-2))
    if (
        epochs < 1
        or batch_size < 1
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise SchemaError("epochs/batch_size must be >=1 and learning_rate must be positive")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    module = build_module(architecture)
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate)
    x = torch.as_tensor(features, dtype=torch.float32)
    y = torch.as_tensor(targets, dtype=torch.long)
    weights = torch.as_tensor(sample_weights, dtype=torch.float32)

    def loss_value() -> Any:
        logits, _hidden = module(x)
        per_sample = torch.nn.functional.cross_entropy(logits, y, reduction="none")
        return (per_sample * weights).sum() / weights.sum()

    training_contract = {
        "algorithm": str(recipe["algorithm"]),
        "algorithm_config": algorithm_config,
        "dataset_digest": dataset_digest,
        "dataset_split": dataset_split,
        "role": role,
        "observation": observation,
        "action": action,
        "architecture": architecture,
        "preprocessing": preprocessing,
        "seed": seed,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weighting": weighting,
    }
    training_contract_digest = digest_uri(
        sha256_bytes(canonical_json(training_contract))
    )
    start_epoch = 0
    resumed_from: str | None = None
    epoch_losses: list[float] = []
    module.train()
    with torch.no_grad():
        initial_loss = float(loss_value().item())

    if recipe.get("resume_from") is not None:
        resume_path = Path(str(recipe["resume_from"]))
        if not resume_path.is_absolute():
            resume_path = (recipe_file.parent / resume_path).resolve()
        checkpoint_meta_path = (
            resume_path / "checkpoint.json" if resume_path.is_dir() else resume_path
        )
        checkpoint_meta = load_manifest(checkpoint_meta_path)
        if checkpoint_meta.get("schema") != TRAIN_CHECKPOINT_SCHEMA:
            raise SchemaError(
                f"expected {TRAIN_CHECKPOINT_SCHEMA}, got "
                f"{checkpoint_meta.get('schema')!r}"
            )
        if checkpoint_meta.get("training_contract_digest") != training_contract_digest:
            raise ConformanceError(
                "training resume contract mismatch: dataset/algorithm/model/optimizer "
                "configuration changed"
            )
        payload_path = checkpoint_meta_path.parent / str(checkpoint_meta["payload"])
        actual_payload = digest_uri(sha256_file(payload_path))
        if actual_payload != checkpoint_meta.get("payload_digest"):
            raise ConformanceError(
                f"training checkpoint mutation detected: declared "
                f"{checkpoint_meta.get('payload_digest')}, actual {actual_payload}"
            )
        try:
            checkpoint_state = torch.load(
                payload_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError as exc:  # pragma: no cover - torch floor supports it
            raise SchemaError(
                "safe checkpoint resume requires torch.load(weights_only=True)"
            ) from exc
        module.load_state_dict(checkpoint_state["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
        rng.bit_generator.state = checkpoint_meta["numpy_rng_state"]
        start_epoch = int(checkpoint_meta["epochs_completed"])
        epoch_losses = [float(value) for value in checkpoint_meta.get("loss_epochs") or []]
        initial_loss = float(checkpoint_meta["initial_loss"])
        resumed_from = str(checkpoint_meta_path)
        if len(epoch_losses) != start_epoch:
            raise ConformanceError(
                "training checkpoint loss history does not match epochs_completed"
            )
        if epochs <= start_epoch:
            raise SchemaError(
                f"recipe epochs={epochs} must exceed resumed epochs={start_epoch}"
            )

    for _epoch in range(start_epoch, epochs):
        order = rng.permutation(len(features))
        losses: list[float] = []
        for start in range(0, len(order), batch_size):
            indexes = torch.as_tensor(order[start : start + batch_size], dtype=torch.long)
            logits, _hidden = module(x[indexes])
            per_sample = torch.nn.functional.cross_entropy(
                logits, y[indexes], reduction="none"
            )
            batch_weights = weights[indexes]
            loss = (per_sample * batch_weights).sum() / batch_weights.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().item()))
        epoch_losses.append(float(np.mean(losses)))
    module.eval()
    with torch.no_grad():
        final_loss = float(loss_value().item())

    cases: list[dict[str, Any]] = []
    with torch.no_grad():
        for raw_observation, feature in zip(raw[:8], features[:8], strict=False):
            logits, _hidden = module(torch.as_tensor(feature).view(1, -1))
            values = logits.detach().cpu().numpy().reshape(-1)
            cases.append(
                {
                    "observation": raw_observation,
                    "mode": "deterministic",
                    "expected_action": int(np.argmax(values)),
                    "expected_logits": values.tolist(),
                }
            )

    out_path = Path(out_dir)
    if out_path.exists() and (not out_path.is_dir() or any(out_path.iterdir())):
        raise SchemaError(f"training output must be empty or absent: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".arena-train-", dir=str(out_path.parent)))
    try:
        recipe_digest = digest_uri(sha256_bytes(canonical_json(recipe)))
        roles = list(recipe.get("roles") or [role])
        bundle = export_policy(
            out_dir=staging / "policy.arena",
            name=str(recipe.get("policy_name") or recipe["name"]),
            roles=roles,
            observation=observation,
            action=action,
            architecture=architecture,
            state_dict=module.state_dict(),
            preprocessing=preprocessing,
            lineage={
                "algorithm": str(recipe["algorithm"]),
                "algorithm_config": algorithm_config,
                "dataset_digest": dataset_digest,
                "dataset_split": dataset_split,
                "recipe_digest": recipe_digest,
                "seed": seed,
                "training_examples": len(features),
                "training_contract_digest": training_contract_digest,
            },
            reference_cases={"provenance": "source-conformance", "cases": cases},
        )
        policy = Policy.load(bundle)
        checkpoint_payload = staging / "checkpoint.pt"
        torch.save(
            {
                "model_state_dict": module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            checkpoint_payload,
        )
        checkpoint = {
            "schema": TRAIN_CHECKPOINT_SCHEMA,
            "algorithm": str(recipe["algorithm"]),
            "training_contract_digest": training_contract_digest,
            "epochs_completed": epochs,
            "initial_loss": initial_loss,
            "loss_epochs": epoch_losses,
            "numpy_rng_state": rng.bit_generator.state,
            "payload": "checkpoint.pt",
            "payload_digest": digest_uri(sha256_file(checkpoint_payload)),
        }
        dump_json(checkpoint, staging / "checkpoint.json")
        run = {
            "schema": TRAIN_RUN_SCHEMA,
            "name": recipe["name"],
            "algorithm": str(recipe["algorithm"]),
            "algorithm_config": algorithm_config,
            "recipe_digest": recipe_digest,
            "dataset_digest": dataset_digest,
            "dataset_split": dataset_split,
            "training_contract_digest": training_contract_digest,
            "role": role,
            "seed": seed,
            "examples": len(features),
            "epochs": epochs,
            "start_epoch": start_epoch,
            "resumed_from": resumed_from,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "sample_weights": {
                "kind": weighting,
                "min": float(np.min(sample_weights)),
                "max": float(np.max(sample_weights)),
                "mean": float(np.mean(sample_weights)),
            },
            "loss": {
                "initial": initial_loss,
                "final": final_loss,
                "epochs": epoch_losses,
            },
            "output_policy": {"path": "policy.arena", "digest": policy.digest},
            "checkpoint": {
                "path": "checkpoint.json",
                "payload_digest": checkpoint["payload_digest"],
                "epochs_completed": epochs,
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        dump_yaml(recipe, staging / "recipe.yaml")
        dump_yaml(run, staging / "train.yaml")
        dump_json(run, staging / "train.json")
        if out_path.exists():
            out_path.rmdir()
        staging.replace(out_path)
        return {**run, "out": str(out_path), "policy": str(out_path / "policy.arena")}
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
