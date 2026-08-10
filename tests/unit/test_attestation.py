from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("cryptography")

from arena.cli.main import main
from arena.core.attestation import (
    generate_signing_keypair,
    sign_artifact,
    verify_artifact_attestation,
)
from arena.core.errors import ConformanceError, SchemaError
from arena.core.mirror import pull_artifact, push_artifact


def _lab_artifact(root: Path, *, name: str = "lab.arena", body: str = "lab-payload\n") -> Path:
    """Hermetic mirrorable artifact — no examples/, training/, or eval demos."""
    artifact = root / name
    artifact.mkdir()
    (artifact / "payload.txt").write_text(body, encoding="utf-8")
    return artifact


def test_hermetic_sign_verify_tamper_fails(tmp_path: Path) -> None:
    """Lab keypair: sign → verify → byte/signature tamper must fail closed."""
    private_key = tmp_path / "lab-private.pem"
    public_key = tmp_path / "lab-public.pem"
    keys = generate_signing_keypair(private_key=private_key, public_key=public_key)
    assert keys["algorithm"] == "ed25519"
    assert keys["key_id"].startswith("sha256:")
    assert private_key.stat().st_mode & 0o077 == 0

    source = _lab_artifact(tmp_path)
    attestation = tmp_path / "lab.attestation.json"
    signed = sign_artifact(
        source,
        private_key=private_key,
        out=attestation,
        issuer="hermetic-lab",
    )
    assert signed["signature"]["algorithm"] == "ed25519"
    assert signed["predicate"]["issuer"] == "hermetic-lab"
    assert attestation.is_file()

    verified = verify_artifact_attestation(
        source,
        attestation=attestation,
        public_key=public_key,
    )
    assert verified["ok"] is True
    assert verified["issuer"] == "hermetic-lab"
    assert verified["key_id"] == keys["key_id"]
    assert verified["identity"] == signed["subject"]["identity"]

    # Tamper the artifact bytes → identity changes → subject mismatch.
    (source / "payload.txt").write_text("tampered-payload\n", encoding="utf-8")
    with pytest.raises(ConformanceError, match="subject mismatch"):
        verify_artifact_attestation(
            source,
            attestation=attestation,
            public_key=public_key,
        )

    # Restore artifact, then corrupt the detached signature blob.
    (source / "payload.txt").write_text("lab-payload\n", encoding="utf-8")
    document = json.loads(attestation.read_text(encoding="utf-8"))
    document["signature"]["value"] = "AAAA"
    attestation.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ConformanceError, match="signature verification failed"):
        verify_artifact_attestation(
            source,
            attestation=attestation,
            public_key=public_key,
        )


def test_detached_signature_survives_identity_preserving_mirror(
    tmp_path: Path,
) -> None:
    private_key = tmp_path / "lab-private.pem"
    public_key = tmp_path / "lab-public.pem"
    keys = generate_signing_keypair(
        private_key=private_key,
        public_key=public_key,
    )
    assert keys["key_id"].startswith("sha256:")
    assert private_key.stat().st_mode & 0o077 == 0

    source = _lab_artifact(tmp_path, name="mirror.arena")
    attestation = tmp_path / "mirror.attestation.json"
    signed = sign_artifact(
        source,
        private_key=private_key,
        out=attestation,
        issuer="example-lab",
    )
    assert signed["signature"]["algorithm"] == "ed25519"
    verified = verify_artifact_attestation(
        source,
        attestation=attestation,
        public_key=public_key,
    )
    assert verified["ok"] is True
    assert verified["issuer"] == "example-lab"

    pushed = push_artifact(source, (tmp_path / "mirror").as_uri(), verify=True)
    restored = tmp_path / "restored.arena"
    pull_artifact(pushed["uri"], restored, verify=True)
    assert verify_artifact_attestation(
        restored,
        attestation=attestation,
        public_key=public_key,
    )["identity"] == verified["identity"]


def test_attestation_refuses_wrong_subject_and_wrong_key(tmp_path: Path) -> None:
    private_key = tmp_path / "private.pem"
    public_key = tmp_path / "public.pem"
    other_private = tmp_path / "other-private.pem"
    other_public = tmp_path / "other-public.pem"
    generate_signing_keypair(private_key=private_key, public_key=public_key)
    generate_signing_keypair(private_key=other_private, public_key=other_public)

    alpha = _lab_artifact(tmp_path, name="alpha.arena", body="alpha\n")
    beta = _lab_artifact(tmp_path, name="beta.arena", body="beta\n")
    attestation = tmp_path / "artifact.json"
    sign_artifact(
        alpha,
        private_key=private_key,
        out=attestation,
        issuer="example-lab",
    )
    with pytest.raises(ConformanceError, match="subject mismatch"):
        verify_artifact_attestation(
            beta,
            attestation=attestation,
            public_key=public_key,
        )
    with pytest.raises(ConformanceError, match="key_id does not match"):
        verify_artifact_attestation(
            alpha,
            attestation=attestation,
            public_key=other_public,
        )


def test_attestation_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    private_key = tmp_path / "private.pem"
    public_key = tmp_path / "public.pem"
    attestation = tmp_path / "artifact.json"
    source = str(_lab_artifact(tmp_path, name="cli.arena"))
    assert main(
        [
            "attest",
            "keygen",
            "--private",
            str(private_key),
            "--public",
            str(public_key),
        ]
    ) == 0
    keygen_out = capsys.readouterr().out
    assert "key_id" in keygen_out
    assert "ed25519" in keygen_out

    assert main(
        [
            "--json",
            "attest",
            "sign",
            source,
            "--key",
            str(private_key),
            "--issuer",
            "cli-lab",
            "--out",
            str(attestation),
        ]
    ) == 0
    signed = json.loads(capsys.readouterr().out)
    assert signed["predicate"]["issuer"] == "cli-lab"
    assert signed["out"] == str(attestation)

    assert main(
        [
            "--json",
            "attest",
            "verify",
            source,
            str(attestation),
            "--key",
            str(public_key),
        ]
    ) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["ok"] is True
    assert verified["issuer"] == "cli-lab"


def test_keygen_refuses_same_output_and_malformed_attestation(
    tmp_path: Path,
) -> None:
    same = tmp_path / "same.pem"
    with pytest.raises(SchemaError, match="must be different"):
        generate_signing_keypair(private_key=same, public_key=same)
    assert not same.exists()

    private_key = tmp_path / "private.pem"
    public_key = tmp_path / "public.pem"
    generate_signing_keypair(private_key=private_key, public_key=public_key)
    source = _lab_artifact(tmp_path, name="malformed-subject.arena")
    malformed = tmp_path / "malformed.json"
    malformed.write_text(
        json.dumps(
            {
                "schema": "arena.attestation/v1",
                "subject": {},
                "predicate": {},
                "signature": {"algorithm": "ed25519"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SchemaError, match="signature.value"):
        verify_artifact_attestation(
            source,
            attestation=malformed,
            public_key=public_key,
        )
