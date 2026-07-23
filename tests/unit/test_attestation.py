from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("cryptography")

from rlx.cli.main import main
from rlx.core.attestation import (
    generate_signing_keypair,
    sign_artifact,
    verify_artifact_attestation,
)
from rlx.core.errors import ConformanceError, SchemaError
from rlx.core.mirror import pull_artifact, push_artifact


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

    source = Path("examples/eval/demo/rock.rlx").resolve()
    attestation = tmp_path / "rock.attestation.json"
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
    restored = tmp_path / "restored.rlx"
    pull_artifact(pushed["uri"], restored, verify=True)
    assert verify_artifact_attestation(
        restored,
        attestation=attestation,
        public_key=public_key,
    )["identity"] == verified["identity"]


def test_attestation_refuses_wrong_subject_and_tampered_signature(
    tmp_path: Path,
) -> None:
    private_key = tmp_path / "private.pem"
    public_key = tmp_path / "public.pem"
    generate_signing_keypair(private_key=private_key, public_key=public_key)
    rock = Path("examples/eval/demo/rock.rlx").resolve()
    paper = Path("examples/eval/demo/paper.rlx").resolve()
    attestation = tmp_path / "artifact.json"
    sign_artifact(
        rock,
        private_key=private_key,
        out=attestation,
        issuer="example-lab",
    )
    with pytest.raises(ConformanceError, match="subject mismatch"):
        verify_artifact_attestation(
            paper,
            attestation=attestation,
            public_key=public_key,
        )

    document = json.loads(attestation.read_text())
    document["signature"]["value"] = "AAAA"
    attestation.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ConformanceError, match="signature verification failed"):
        verify_artifact_attestation(
            rock,
            attestation=attestation,
            public_key=public_key,
        )


def test_attestation_cli(tmp_path: Path) -> None:
    private_key = tmp_path / "private.pem"
    public_key = tmp_path / "public.pem"
    attestation = tmp_path / "artifact.json"
    source = "examples/eval/demo/rock.rlx"
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
    assert main(
        [
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
    assert main(
        [
            "attest",
            "verify",
            source,
            str(attestation),
            "--key",
            str(public_key),
        ]
    ) == 0


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
    malformed = tmp_path / "malformed.json"
    malformed.write_text(
        json.dumps(
            {
                "schema": "rlx.attestation/v1",
                "subject": {},
                "predicate": {},
                "signature": {"algorithm": "ed25519"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SchemaError, match="signature.value"):
        verify_artifact_attestation(
            "examples/eval/demo/rock.rlx",
            attestation=malformed,
            public_key=public_key,
        )
