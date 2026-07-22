"""Unit tests for store and manifests."""

from pathlib import Path

from rlx.core.identity import digest_uri, sha256_bytes
from rlx.core.manifests import (
    POLICY_SCHEMA,
    dump_yaml,
    load_manifest,
    policy_content_digest,
    validate_policy_manifest,
)
from rlx.core.store import LocalStore


def test_init_and_put_get(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    store.init()
    assert (tmp_path / ".rlx" / "workspace.toml").exists()
    digest = store.put_bytes(b"hello-rlx")
    assert digest.startswith("sha256:")
    assert store.get_bytes(digest) == b"hello-rlx"
    store.set_ref("policy/demo", digest)
    assert store.get_ref("policy/demo") == digest


def test_policy_manifest_roundtrip(tmp_path: Path) -> None:
    manifest = {
        "schema": POLICY_SCHEMA,
        "name": "demo",
        "roles": {"allowed": ["player_0"]},
        "runtime": {"adapter": "custom-pytorch", "python": "3.12"},
        "observation": {"type": "Discrete", "n": 4},
        "action": {"type": "Discrete", "n": 3, "masks": "none"},
        "state": {"recurrent": False, "reset_on": []},
        "inference": {"modes": ["deterministic"]},
        "preprocessing": {"included": True, "id": "normalize_v0", "mean": 0.0, "std": 1.0},
        "architecture": {
            "type": "mlp_categorical",
            "observation_dim": 4,
            "hidden_dims": [8],
            "action_n": 3,
        },
        "payloads": {"weights": {"digest": digest_uri(sha256_bytes(b"x")), "path": "payloads/weights.pt"}},
    }
    validate_policy_manifest(manifest)
    path = tmp_path / "policy.yaml"
    dump_yaml(manifest, path)
    loaded = load_manifest(path)
    assert policy_content_digest(loaded) == policy_content_digest(manifest)
