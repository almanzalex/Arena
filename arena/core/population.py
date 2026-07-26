"""Policy populations: content-addressed sets of 0.1 policy digests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from arena.core.errors import CompatibilityError, SchemaError, StoreError
from arena.core.identity import canonical_json
from arena.core.manifests import (
    POPULATION_SCHEMA,
    dump_yaml,
    load_manifest,
    population_content_digest,
    validate_population_manifest,
)
from arena.core.sdk import Policy
from arena.core.store import LocalStore


def _resolve_member_policy(
    policy_ref: Any, *, store: LocalStore, base_dir: Path | None = None
) -> tuple[str, dict[str, Any]]:
    text = str(policy_ref).strip()
    if text.startswith("sha256:"):
        return text, {"roles": {"allowed": []}}
    path = Path(text)
    if not path.is_absolute() and base_dir is not None:
        path = (base_dir / path).resolve()
    if path.exists():
        policy = Policy.load(path)
        return policy.digest, policy.manifest
    try:
        digest = store.get_ref(text)
        return digest, {"roles": {"allowed": []}}
    except StoreError:
        pass
    raise SchemaError(f"cannot resolve policy ref: {policy_ref!r}")


def create_population(
    *,
    name: str,
    members: list[dict[str, Any]],
    store: LocalStore,
    ref: str | None = None,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    """Resolve member policy paths to digests and store an immutable population object."""
    resolved: list[dict[str, Any]] = []
    for i, raw in enumerate(members):
        if not isinstance(raw, dict):
            raise SchemaError(f"members[{i}] must be a mapping")
        policy_ref = raw.get("policy")
        if policy_ref is None:
            raise SchemaError(f"members[{i}].policy is required")
        digest, manifest = _resolve_member_policy(policy_ref, store=store, base_dir=base_dir)
        member: dict[str, Any] = {
            "policy": digest,
            "weight": float(raw.get("weight", 1.0)),
        }
        if "generation" in raw:
            member["generation"] = raw["generation"]
        if "tags" in raw:
            member["tags"] = list(raw["tags"])
        if "roles" in raw:
            member["roles"] = {"allowed": list(raw["roles"].get("allowed", []))}
        elif manifest.get("roles", {}).get("allowed"):
            member["roles"] = {"allowed": list(manifest["roles"]["allowed"])}
        resolved.append(member)

    population = validate_population_manifest(
        {
            "schema": POPULATION_SCHEMA,
            "name": name,
            "members": resolved,
        }
    )
    digest = population_content_digest(population)
    population["digest"] = digest
    object_digest = store.put_bytes(canonical_json(population))
    ref_name = ref or f"populations/{name}"
    store.set_ref(ref_name, object_digest)
    store.set_ref(f"populations/by-digest/{digest.split(':', 1)[-1][:16]}", object_digest)
    population["object_digest"] = object_digest
    population["ref"] = ref_name
    return population


def load_population(ref: str | Path, *, store: LocalStore | None = None) -> dict[str, Any]:
    path = Path(str(ref))
    if path.exists():
        data = load_manifest(path)
        if data.get("schema") != POPULATION_SCHEMA:
            raise SchemaError(f"expected {POPULATION_SCHEMA}")
        # Allow authoring YAMLs with local bundle paths; resolve to digests in-memory.
        if any(
            not str(m.get("policy", "")).startswith("sha256:")
            for m in (data.get("members") or [])
        ):
            if store is None:
                try:
                    store = LocalStore.find()
                except StoreError:
                    store = LocalStore(path.parent)
                    if not (store.arena / "workspace.toml").exists():
                        store.init()
            return create_population(
                name=str(data["name"]),
                members=list(data["members"]),
                store=store,
                ref=None,
                base_dir=path.parent.resolve(),
            )
        return validate_population_manifest(data)
    if store is None:
        store = LocalStore.find()
    text = str(ref)
    if text.startswith("sha256:"):
        data = json.loads(store.get_bytes(text).decode("utf-8"))
        return validate_population_manifest(data)
    for candidate in (
        text,
        f"populations/{text}",
        text if text.startswith("populations/") else f"populations/{text}",
    ):
        try:
            digest = store.get_ref(candidate)
            data = json.loads(store.get_bytes(digest).decode("utf-8"))
            return validate_population_manifest(data)
        except StoreError:
            continue
    raise StoreError(f"population ref not found: {ref}")


def member_allowed_for_role(member: dict[str, Any], role: str) -> bool:
    roles = member.get("roles")
    if not roles:
        return True
    allowed = roles.get("allowed")
    if not allowed:
        return True
    return role in allowed


def assert_members_compatible_with_role(population: dict[str, Any], role: str) -> None:
    bad = [m["policy"] for m in population["members"] if not member_allowed_for_role(m, role)]
    if bad:
        raise CompatibilityError(
            f"population members incompatible with role {role!r}: {bad}. "
            "Repair: adjust members[].roles.allowed or choose a different population."
        )


def create_population_from_yaml(
    path: Path | str, *, store: LocalStore, ref: str | None = None
) -> dict[str, Any]:
    path = Path(path)
    data = load_manifest(path)
    if data.get("schema") != POPULATION_SCHEMA:
        raise SchemaError(f"expected {POPULATION_SCHEMA}")
    return create_population(
        name=str(data["name"]),
        members=list(data["members"]),
        store=store,
        ref=ref,
        base_dir=path.parent.resolve(),
    )


def write_population_yaml(population: dict[str, Any], path: Path | str) -> None:
    out = {
        "schema": POPULATION_SCHEMA,
        "name": population["name"],
        "members": population["members"],
        "digest": population.get("digest") or population_content_digest(population),
    }
    dump_yaml(out, path)
