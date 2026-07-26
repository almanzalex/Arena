"""Detached Ed25519 attestations for Arena artifact identity.

Signatures prove possession of an explicitly trusted key. Arena deliberately does
not invent accounts, certificate authorities, revocation, or transparency logs.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from arena.core.errors import ArenaError, ConformanceError, SchemaError
from arena.core.identity import canonical_json, digest_uri, sha256_bytes
from arena.core.io import atomic_write_bytes
from arena.core.manifests import dump_json, load_manifest
from arena.core.mirror import build_mirror_artifact

ATTESTATION_SCHEMA = "arena.attestation/v1"


def _crypto() -> tuple[Any, Any, Any, Any]:
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
            Ed25519PublicKey,
        )
    except ImportError as exc:
        raise ArenaError(
            "Arena signing requires cryptography; install with: pip install 'arena[signing]'"
        ) from exc
    return serialization, Ed25519PrivateKey, Ed25519PublicKey, InvalidSignature


def _refuse_existing(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise SchemaError(f"refusing to overwrite existing path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def generate_signing_keypair(
    *,
    private_key: Path | str,
    public_key: Path | str,
) -> dict[str, Any]:
    serialization, Ed25519PrivateKey, _public_cls, _invalid = _crypto()
    private_path = Path(private_key)
    public_path = Path(public_key)
    if private_path.resolve() == public_path.resolve():
        raise SchemaError("private and public key paths must be different")
    _refuse_existing(private_path)
    _refuse_existing(public_path)
    key = Ed25519PrivateKey.generate()
    private_bytes = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_bytes = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    # Publish the non-secret half first. A process death can at worst leave an
    # orphan public key; the private key is a single fsync+rename operation.
    atomic_write_bytes(public_path, public_bytes, mode=0o644)
    try:
        atomic_write_bytes(private_path, private_bytes, mode=0o600)
    except BaseException:
        public_path.unlink(missing_ok=True)
        raise
    key_id = public_key_id(public_path)
    return {
        "algorithm": "ed25519",
        "key_id": key_id,
        "private_key": str(private_path),
        "public_key": str(public_path),
    }


def public_key_id(public_key: Path | str) -> str:
    serialization, _private_cls, public_cls, _invalid = _crypto()
    key = serialization.load_pem_public_key(Path(public_key).read_bytes())
    if not isinstance(key, public_cls):
        raise SchemaError("trusted public key must be Ed25519")
    der = key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return digest_uri(sha256_bytes(der))


def _statement(
    *,
    identity: str,
    kind: str,
    issuer: str,
    key_id: str,
    created_at: str,
) -> dict[str, Any]:
    return {
        "schema": ATTESTATION_SCHEMA,
        "subject": {"identity": identity, "kind": kind},
        "predicate": {
            "type": "arena.artifact-authenticity/v1",
            "issuer": issuer,
            "key_id": key_id,
            "created_at": created_at,
        },
    }


def sign_artifact(
    source: Path | str,
    *,
    private_key: Path | str,
    out: Path | str,
    issuer: str,
) -> dict[str, Any]:
    if not issuer.strip():
        raise SchemaError("attestation issuer must be non-empty")
    serialization, private_cls, _public_cls, _invalid = _crypto()
    key = serialization.load_pem_private_key(
        Path(private_key).read_bytes(),
        password=None,
    )
    if not isinstance(key, private_cls):
        raise SchemaError("signing private key must be Ed25519")
    public_der = key.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    key_id = digest_uri(sha256_bytes(public_der))
    artifact = build_mirror_artifact(source)
    statement = _statement(
        identity=artifact.identity,
        kind=artifact.kind,
        issuer=issuer,
        key_id=key_id,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    signature = key.sign(canonical_json(statement))
    attestation = {
        **statement,
        "signature": {
            "algorithm": "ed25519",
            "value": base64.b64encode(signature).decode("ascii"),
        },
    }
    out_path = Path(out)
    _refuse_existing(out_path)
    dump_json(attestation, out_path)
    return {**attestation, "out": str(out_path)}


def verify_artifact_attestation(
    source: Path | str,
    *,
    attestation: Path | str,
    public_key: Path | str,
) -> dict[str, Any]:
    serialization, _private_cls, _public_cls, InvalidSignature = _crypto()
    document = load_manifest(attestation)
    if document.get("schema") != ATTESTATION_SCHEMA:
        raise SchemaError(
            f"expected attestation schema {ATTESTATION_SCHEMA}, "
            f"got {document.get('schema')!r}"
        )
    signature = document.get("signature")
    if not isinstance(signature, dict) or signature.get("algorithm") != "ed25519":
        raise SchemaError("attestation signature.algorithm must be ed25519")
    if not isinstance(signature.get("value"), str) or not signature["value"]:
        raise SchemaError("attestation signature.value must be non-empty base64")
    subject = document.get("subject")
    predicate = document.get("predicate")
    if not isinstance(subject, dict):
        raise SchemaError("attestation subject must be a mapping")
    if not isinstance(predicate, dict):
        raise SchemaError("attestation predicate must be a mapping")
    for field in ("issuer", "key_id", "created_at"):
        if not isinstance(predicate.get(field), str) or not predicate[field]:
            raise SchemaError(f"attestation predicate.{field} must be non-empty")
    statement = {
        "schema": document["schema"],
        "subject": subject,
        "predicate": predicate,
    }
    artifact = build_mirror_artifact(source)
    expected_subject = {"identity": artifact.identity, "kind": artifact.kind}
    if statement["subject"] != expected_subject:
        raise ConformanceError(
            f"attestation subject mismatch: expected {expected_subject}, "
            f"got {statement['subject']}"
        )
    key = serialization.load_pem_public_key(Path(public_key).read_bytes())
    if not isinstance(key, _public_cls):
        raise SchemaError("trusted public key must be Ed25519")
    key_id = public_key_id(public_key)
    if (statement.get("predicate") or {}).get("key_id") != key_id:
        raise ConformanceError("attestation key_id does not match trusted public key")
    try:
        key.verify(
            base64.b64decode(str(signature["value"]), validate=True),
            canonical_json(statement),
        )
    except (InvalidSignature, ValueError) as exc:
        raise ConformanceError("attestation signature verification failed") from exc
    return {
        "schema": "arena.attestation-verification/v1",
        "ok": True,
        "identity": artifact.identity,
        "kind": artifact.kind,
        "issuer": statement["predicate"]["issuer"],
        "key_id": key_id,
        "attestation": str(Path(attestation).resolve()),
    }
