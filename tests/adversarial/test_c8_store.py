"""Claim 8 (adversarial): store robustness.

Attacks: simulate an interrupted/partial write and confirm the store is not left
corrupted (atomic-write claim); confirm ref resolution + re-open works; confirm a
leftover temp file never masquerades as a real object.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from arena.core.errors import StoreError
from arena.core.identity import parse_digest, sha256_bytes
from arena.core.store import LocalStore


def test_interrupted_write_leaves_no_partial_object(tmp_path: Path, monkeypatch) -> None:
    """If the final atomic rename fails mid-write, no half-written object is published
    and no temp turd is left behind; the store remains fully usable afterwards."""
    store = LocalStore(tmp_path)
    store.init()

    payload = b"a" * 4096
    digest_hex = sha256_bytes(payload)
    obj_dir = store.objects / digest_hex[:2]

    real_link = os.link

    def boom_link(source, target):  # noqa: ANN001
        del source, target
        raise OSError("simulated crash before atomic publish")

    monkeypatch.setattr(os, "link", boom_link)
    with pytest.raises(OSError):
        store.put_bytes(payload)
    monkeypatch.setattr(os, "link", real_link)

    # The object was never published...
    dest = store.objects / digest_hex[:2] / digest_hex[2:]
    assert not dest.exists(), "partial object was published despite a failed write"
    # ...and no leftover temp files remain in the object shard directory.
    if obj_dir.exists():
        leftovers = list(obj_dir.iterdir())
        assert leftovers == [], f"leftover temp files: {leftovers}"

    # The store still works: a real write now succeeds and reads back cleanly.
    digest = store.put_bytes(payload)
    assert store.get_bytes(digest) == payload


def test_idempotent_put_and_reopen(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    store.init()
    d1 = store.put_bytes(b"same-bytes")
    d2 = store.put_bytes(b"same-bytes")
    assert d1 == d2

    store.set_ref("policy:demo@1", d1)
    # A fresh handle (as a second researcher would open it) resolves the same ref/object.
    reopened = LocalStore.find(tmp_path)
    assert reopened.get_ref("policy:demo@1") == d1
    assert reopened.get_bytes(d1) == b"same-bytes"


def test_leftover_temp_file_not_resolved_as_object(tmp_path: Path) -> None:
    """A stray temp file in the shard dir must never be mistaken for a real object."""
    store = LocalStore(tmp_path)
    store.init()
    digest = store.put_bytes(b"genuine")
    shard = store.objects / parse_digest(digest)[:2]
    (shard / "tmp_garbage_leftover").write_bytes(b"junk")

    # The genuine object still resolves and verifies.
    assert store.get_bytes(digest) == b"genuine"
    # A bogus digest (the temp file's name) is not resolvable as an object.
    with pytest.raises(StoreError, match="invalid object digest"):
        store.get_bytes("sha256:tmp_garbage_leftover")


def test_find_without_workspace_errors(tmp_path: Path) -> None:
    with pytest.raises(StoreError, match="no .arena"):
        LocalStore.find(tmp_path)
