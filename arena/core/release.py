"""Signed, durable release-evidence verification."""

from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Any

from arena.core.attestation import _crypto, public_key_id
from arena.core.errors import ConformanceError, SchemaError
from arena.core.identity import canonical_json, digest_uri, parse_digest, sha256_bytes, sha256_file
from arena.core.manifests import dump_json, load_manifest

EVIDENCE_SCHEMA = "arena.release-evidence/v1"
SIGNATURE_SCHEMA = "arena.detached-signature/v1"
CURRENT_SCHEMA = "arena.qualification-ledger/v1"
MANDATORY_GATES = {f"R-{index:02d}" for index in range(1, 15)}
NON_WAIVABLE_GATES = {
    "R-01",
    "R-02",
    "R-03",
    "R-04",
    "R-05",
    "R-06",
    "R-07",
    "R-08",
    "R-09",
    "R-10",
    "R-11",
    "R-12",
    "R-13",
    "R-14",
}
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_TAG_RE = re.compile(r"^v?[0-9]+\.[0-9]+\.[0-9]+(?:[A-Za-z0-9.-]+)?$")


def assemble_release_evidence(
    *,
    release: str,
    tag: str,
    commit: str,
    gates: dict[str, Path | str],
    artifacts: list[Path | str],
    out: Path | str,
) -> dict[str, Any]:
    out_path = Path(out)
    if out_path.exists() or out_path.is_symlink():
        raise SchemaError(f"refusing to overwrite release evidence: {out_path}")
    gate_entries = []
    for gate_id, raw_path in sorted(gates.items()):
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise SchemaError(f"release gate evidence is unavailable: {path}")
        gate_entries.append(
            {
                "id": gate_id,
                "status": "pass",
                "evidence_digest": digest_uri(sha256_file(path)),
                "evidence_path": str(path),
            }
        )
    artifact_entries = []
    for raw_path in artifacts:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise SchemaError(f"release artifact is unavailable: {path}")
        artifact_entries.append(
            {
                "path": str(path),
                "digest": digest_uri(sha256_file(path)),
                "size": path.stat().st_size,
            }
        )
    document = {
        "schema": EVIDENCE_SCHEMA,
        "release": release,
        "tag": tag,
        "commit": commit,
        "gates": gate_entries,
        "artifacts": artifact_entries,
    }
    _validate_evidence(document)
    dump_json(document, out_path)
    return document


def _validate_evidence(document: dict[str, Any]) -> None:
    if document.get("schema") != EVIDENCE_SCHEMA:
        raise SchemaError(
            f"expected evidence schema {EVIDENCE_SCHEMA}, got {document.get('schema')!r}"
        )
    if not _TAG_RE.fullmatch(str(document.get("tag", ""))):
        raise SchemaError("release evidence tag must be a concrete version tag")
    if not _COMMIT_RE.fullmatch(str(document.get("commit", ""))):
        raise SchemaError("release evidence commit must be a full lowercase Git commit")
    release = str(document.get("release", ""))
    if not release or release not in str(document["tag"]):
        raise SchemaError("release evidence release must match its tag")
    gates = document.get("gates")
    if not isinstance(gates, list):
        raise SchemaError("release evidence gates must be a list")
    by_id: dict[str, dict[str, Any]] = {}
    for gate in gates:
        if not isinstance(gate, dict) or not isinstance(gate.get("id"), str):
            raise SchemaError("each release gate must be a mapping with an id")
        gate_id = gate["id"]
        if gate_id in by_id:
            raise SchemaError(f"duplicate release gate: {gate_id}")
        if gate.get("status") not in {"pass", "downgraded", "waived"}:
            raise SchemaError(f"release gate {gate_id} has invalid status")
        if not isinstance(gate.get("evidence_digest"), str):
            raise SchemaError(f"release gate {gate_id} requires evidence_digest")
        parse_digest(gate["evidence_digest"])
        by_id[gate_id] = gate
    missing = sorted(MANDATORY_GATES - set(by_id))
    if missing:
        raise ConformanceError(
            "release evidence is missing mandatory gates: " + ", ".join(missing)
        )
    invalid = sorted(
        gate_id
        for gate_id in NON_WAIVABLE_GATES
        if by_id[gate_id].get("status") != "pass"
    )
    if invalid:
        raise ConformanceError(
            "non-waivable release gates did not pass: " + ", ".join(invalid)
        )
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise SchemaError("release evidence requires at least one release artifact")
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise SchemaError("release artifacts must be mappings")
        parse_digest(str(artifact.get("digest", "")))


def sign_release_evidence(
    evidence_index: Path | str,
    *,
    private_key: Path | str,
    out: Path | str,
) -> dict[str, Any]:
    out_path = Path(out)
    if out_path.exists() or out_path.is_symlink():
        raise SchemaError(f"refusing to overwrite existing signature: {out_path}")
    document = load_manifest(evidence_index)
    _validate_evidence(document)
    serialization, private_cls, _public_cls, _invalid = _crypto()
    key = serialization.load_pem_private_key(Path(private_key).read_bytes(), password=None)
    if not isinstance(key, private_cls):
        raise SchemaError("release signing private key must be Ed25519")
    statement = canonical_json(document)
    public_der = key.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    signature = {
        "schema": SIGNATURE_SCHEMA,
        "algorithm": "ed25519",
        "key_id": digest_uri(sha256_bytes(public_der)),
        "subject_digest": digest_uri(sha256_bytes(statement)),
        "value": base64.b64encode(key.sign(statement)).decode("ascii"),
    }
    dump_json(signature, out_path)
    return signature


def sign_qualification_ledger(
    ledger: Path | str,
    *,
    private_key: Path | str,
    out: Path | str,
) -> dict[str, Any]:
    out_path = Path(out)
    if out_path.exists() or out_path.is_symlink():
        raise SchemaError(f"refusing to overwrite existing signature: {out_path}")
    document = load_manifest(ledger)
    _validate_qualification_ledger(document)
    serialization, private_cls, _public_cls, _invalid = _crypto()
    key = serialization.load_pem_private_key(Path(private_key).read_bytes(), password=None)
    if not isinstance(key, private_cls):
        raise SchemaError("release signing private key must be Ed25519")
    statement = canonical_json(document)
    public_der = key.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    signature = {
        "schema": SIGNATURE_SCHEMA,
        "algorithm": "ed25519",
        "key_id": digest_uri(sha256_bytes(public_der)),
        "subject_digest": digest_uri(sha256_bytes(statement)),
        "value": base64.b64encode(key.sign(statement)).decode("ascii"),
    }
    dump_json(signature, out_path)
    return signature


def _verify_signature(
    document: dict[str, Any],
    *,
    signature_path: Path | str,
    public_key: Path | str,
) -> str:
    serialization, _private_cls, public_cls, InvalidSignature = _crypto()
    signature = load_manifest(signature_path)
    if signature.get("schema") != SIGNATURE_SCHEMA:
        raise SchemaError("unsupported release signature schema")
    if signature.get("algorithm") != "ed25519":
        raise SchemaError("release signature algorithm must be ed25519")
    statement = canonical_json(document)
    expected_subject = digest_uri(sha256_bytes(statement))
    if signature.get("subject_digest") != expected_subject:
        raise ConformanceError("release signature subject digest does not match evidence")
    key = serialization.load_pem_public_key(Path(public_key).read_bytes())
    if not isinstance(key, public_cls):
        raise SchemaError("release verification key must be Ed25519")
    key_id = public_key_id(public_key)
    if signature.get("key_id") != key_id:
        raise ConformanceError("release signature key_id does not match trusted key")
    try:
        key.verify(
            base64.b64decode(str(signature.get("value", "")), validate=True),
            statement,
        )
    except (InvalidSignature, ValueError) as exc:
        raise ConformanceError("release evidence signature verification failed") from exc
    return key_id


def _verify_local_artifacts(document: dict[str, Any], *, root: Path) -> list[str]:
    checked: list[str] = []
    for artifact in document["artifacts"]:
        raw_path = artifact.get("path")
        if not raw_path:
            continue
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise ConformanceError(f"release artifact is unavailable: {path}")
        actual = digest_uri(sha256_file(path))
        if actual != artifact["digest"]:
            raise ConformanceError(
                f"release artifact digest mismatch for {path}: "
                f"expected {artifact['digest']}, got {actual}"
            )
        checked.append(str(path))
    return checked


def _verify_local_gate_evidence(document: dict[str, Any], *, root: Path) -> list[str]:
    checked: list[str] = []
    for gate in document["gates"]:
        raw_path = gate.get("evidence_path")
        if not raw_path:
            continue
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise ConformanceError(
                f"release gate evidence is unavailable for {gate['id']}: {path}"
            )
        actual = digest_uri(sha256_file(path))
        if actual != gate["evidence_digest"]:
            raise ConformanceError(
                f"release gate evidence digest mismatch for {gate['id']} at {path}: "
                f"expected {gate['evidence_digest']}, got {actual}"
            )
        checked.append(str(path))
    return checked


def _validate_qualification_ledger(ledger: dict[str, Any]) -> list[dict[str, Any]]:
    if ledger.get("schema") != CURRENT_SCHEMA:
        raise SchemaError(f"expected current ledger schema {CURRENT_SCHEMA}")
    if not isinstance(ledger.get("release"), str) or not ledger["release"]:
        raise SchemaError("qualification ledger release must be non-empty")
    records = ledger.get("records")
    if not isinstance(records, list) or not records:
        raise SchemaError("qualification ledger records must be a non-empty list")
    capabilities: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise SchemaError(
                f"qualification ledger record {index} must be a mapping"
            )
        capability = record.get("capability")
        if not isinstance(capability, str) or not capability.strip():
            raise SchemaError(
                f"qualification ledger record {index} requires capability"
            )
        if capability in capabilities:
            raise SchemaError(
                f"duplicate qualification ledger capability: {capability}"
            )
        capabilities.add(capability)
        if record.get("status") not in {"pass", "stale", "failed"}:
            raise SchemaError(
                f"qualification ledger capability {capability} has invalid status"
            )
        validated.append(record)
    return validated


def _apply_current_ledger(
    document: dict[str, Any],
    *,
    ledger_path: Path | str,
    signature_path: Path | str,
    public_key: Path | str,
) -> dict[str, Any]:
    ledger = load_manifest(ledger_path)
    records = _validate_qualification_ledger(ledger)
    if ledger.get("release") != document.get("release"):
        raise ConformanceError("current ledger release does not match evidence release")
    ledger_key_id = _verify_signature(
        ledger,
        signature_path=signature_path,
        public_key=public_key,
    )
    stale = [
        item["capability"]
        for item in records
        if item["status"] in {"stale", "failed"}
    ]
    if stale:
        raise ConformanceError(
            "current qualification ledger has stale or failed capabilities: "
            + ", ".join(str(item) for item in stale)
        )
    return {
        "records": len(records),
        "stale_or_failed": stale,
        "key_id": ledger_key_id,
    }


def verify_release_evidence(
    evidence_index: Path | str,
    *,
    signature: Path | str,
    public_key: Path | str,
    current_ledger: Path | str | None = None,
    current_ledger_signature: Path | str | None = None,
    current_ledger_key: Path | str | None = None,
) -> dict[str, Any]:
    index_path = Path(evidence_index)
    document = load_manifest(index_path)
    _validate_evidence(document)
    key_id = _verify_signature(
        document,
        signature_path=signature,
        public_key=public_key,
    )
    artifacts = _verify_local_artifacts(document, root=index_path.parent)
    gate_evidence = _verify_local_gate_evidence(document, root=index_path.parent)
    current = None
    if current_ledger is not None:
        if current_ledger_signature is None or current_ledger_key is None:
            raise SchemaError(
                "current verification requires --ledger-signature and --ledger-key"
            )
        current = _apply_current_ledger(
            document,
            ledger_path=current_ledger,
            signature_path=current_ledger_signature,
            public_key=current_ledger_key,
        )
    return {
        "schema": "arena.release-verification/v1",
        "ok": True,
        "release": document["release"],
        "tag": document["tag"],
        "commit": document["commit"],
        "key_id": key_id,
        "evidence_digest": digest_uri(sha256_bytes(canonical_json(document))),
        "gates": len(document["gates"]),
        "local_gate_evidence_checked": gate_evidence,
        "local_artifacts_checked": artifacts,
        "mode": "current" if current_ledger is not None else "at-release",
        "current": current,
    }
