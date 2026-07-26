"""Claim 1 (adversarial): content-addressing integrity.

Attacks:
  * identical artifacts -> identical digest; any payload change -> different digest
  * irrelevant metadata (lineage / conformance / local paths) must NOT change identity
  * corrupt/tamper a stored object -> DETECTED (never silently trusted)
  * corrupt/tamper a policy bundle payload -> DETECTED on load
  * path-traversal / absolute / unicode ref names
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from arena.core.errors import StoreError
from arena.core.identity import parse_digest, sha256_bytes
from arena.core.manifests import policy_content_digest
from arena.core.store import LocalStore

torch = pytest.importorskip("torch")
pytest.importorskip("pettingzoo")

from _adv_envs import make_discrete_policy  # noqa: E402

from arena.adapters.policy_custom_torch import (  # noqa: E402
    load_runtime,
    verify_bundle_integrity,
)
from arena.core.errors import ConformanceError  # noqa: E402
from arena.core.manifests import load_manifest  # noqa: E402


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_identity_stable_and_metadata_irrelevant(tmp_path: Path) -> None:
    """Two byte-identical exports share a digest; lineage/conformance/name-of-file
    do not participate, but any executable-content change flips the digest."""
    a = make_discrete_policy(tmp_path / "a", role="player_0", seed=123)
    b = make_discrete_policy(tmp_path / "b", role="player_0", seed=123)
    man_a = load_manifest(a / "policy.yaml")
    man_b = load_manifest(b / "policy.yaml")
    assert policy_content_digest(man_a) == policy_content_digest(man_b)

    # Irrelevant metadata must NOT change identity.
    tampered = copy.deepcopy(man_a)
    tampered["lineage"] = {"source_run": "run:secret", "source_checkpoint": "/home/me/ckpt-999"}
    tampered["conformance"] = {"status": "verified", "cases": 999, "notes": "whatever"}
    assert policy_content_digest(tampered) == policy_content_digest(man_a)

    # A change to any *executable* contract field must change identity.
    for mutate in (
        lambda m: m["action"].__setitem__("n", 99),
        lambda m: m["observation"].__setitem__("n", 99),
        lambda m: m["architecture"].__setitem__("hidden_dims", [1, 2, 3]),
        lambda m: m["preprocessing"].__setitem__("std", 7.0),
        lambda m: m["roles"].__setitem__("allowed", ["evil"]),
    ):
        mut = copy.deepcopy(man_a)
        mutate(mut)
        assert policy_content_digest(mut) != policy_content_digest(man_a)


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_different_weights_different_digest(tmp_path: Path) -> None:
    a = make_discrete_policy(tmp_path / "a", role="player_0", seed=1)
    b = make_discrete_policy(tmp_path / "b", role="player_0", seed=2)
    da = policy_content_digest(load_manifest(a / "policy.yaml"))
    db = policy_content_digest(load_manifest(b / "policy.yaml"))
    assert da != db, "different weights must yield different content identity"


def test_object_store_tamper_detected(tmp_path: Path) -> None:
    """Flipping bytes of a stored content-addressed object is detected, not trusted."""
    store = LocalStore(tmp_path)
    store.init()
    digest = store.put_bytes(b"important-artifact-bytes")
    assert store.get_bytes(digest) == b"important-artifact-bytes"

    obj_path = store.objects / parse_digest(digest)[:2] / parse_digest(digest)[2:]
    obj_path.write_bytes(b"TAMPERED-PAYLOAD")
    with pytest.raises(StoreError, match="integrity check failed"):
        store.get_bytes(digest)
    with pytest.raises(StoreError):
        store.verify_object(digest)


@pytest.mark.requires_torch
@pytest.mark.requires_pettingzoo
def test_bundle_weights_tamper_detected_on_load(tmp_path: Path) -> None:
    """A bundle whose weights.pt no longer matches its manifest digest must fail
    loudly at load time (not silently serve a different policy)."""
    bundle = make_discrete_policy(tmp_path / "pol", role="player_0", seed=5)
    assert verify_bundle_integrity(bundle)["ok"]
    load_runtime(bundle)  # clean load works

    # Swap the weights for a different network's weights.
    torch.manual_seed(9999)
    from arena.adapters.policy_custom_torch import build_module

    other = build_module(
        {"type": "mlp_categorical", "observation_dim": 4, "hidden_dims": [16], "action_n": 3}
    ).state_dict()
    torch.save(other, bundle / "payloads" / "weights.pt")

    with pytest.raises(ConformanceError, match="integrity check failed"):
        load_runtime(bundle)
    with pytest.raises(ConformanceError):
        verify_bundle_integrity(bundle)


def test_ref_path_traversal_blocked(tmp_path: Path) -> None:
    """Weird ref names must never escape the refs/ directory."""
    store = LocalStore(tmp_path)
    store.init()
    digest = store.put_bytes(b"x")

    escape_marker = tmp_path / "PWNED"
    for evil in ("../../PWNED", "../PWNED", "a/../../PWNED"):
        with pytest.raises(StoreError, match="traversal"):
            store.set_ref(evil, digest)
    assert not escape_marker.exists()

    # Absolute paths (posix and would-be windows drive) are rejected.
    with pytest.raises(StoreError):
        store.set_ref(str(tmp_path / "abs_escape"), digest)
    assert not (tmp_path / "abs_escape").exists()

    # Empty / whitespace names rejected.
    for bad in ("", "   "):
        with pytest.raises(StoreError):
            store.set_ref(bad, digest)


def test_ref_unicode_and_tag_roundtrip(tmp_path: Path) -> None:
    """Legitimate (including unicode + colon/at tag) ref names round-trip correctly."""
    store = LocalStore(tmp_path)
    store.init()
    d1 = store.put_bytes(b"one")
    d2 = store.put_bytes(b"two")

    store.set_ref("policy:évader/π@sha256:abc", d1)
    assert store.get_ref("policy:évader/π@sha256:abc") == d1

    store.set_ref("policy:pursuer/candidate@latest", d2)
    # Re-open the workspace from scratch: ref resolution survives a fresh handle.
    reopened = LocalStore(tmp_path)
    assert reopened.get_ref("policy:pursuer/candidate@latest") == d2

    with pytest.raises(StoreError, match="ref not found"):
        store.get_ref("policy:does-not-exist@1")


def test_put_get_content_addressed_roundtrip(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    store.init()
    payload = b"deterministic-content"
    d1 = store.put_bytes(payload)
    d2 = store.put_bytes(payload)
    assert d1 == d2 == f"sha256:{sha256_bytes(payload)}"
    assert store.get_bytes(d1) == payload
