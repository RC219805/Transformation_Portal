#!/usr/bin/env python3
"""Negative-path coverage for ArtifactStore hash verification.

`tests/storage/test_cas_store_atomic.py` covers the success path and one
hash-mismatch case for ``add_file``. This file targets the remaining hash and
integrity error paths that decide whether the CAS can silently propagate
corrupted data:

- ``add_bytes`` write path with a forced hash mismatch (cas_store.py:414)
- ``verify_object`` returns False for tampered content
- ``verify_object`` raises CASError for missing objects
- ``materialize`` raises CASError for missing objects
- ``add_file`` raises CASError for a missing source file
- Exact CASError message format on hash mismatch (the line 353 raise) — clients
  parse this for forensics
- No leaked ``.tmp`` files after add_bytes hash-mismatch failure
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from transformation_portal.storage.cas_store import ArtifactStore, CASError

pytestmark = pytest.mark.unit


def test_add_bytes_raises_cas_error_on_hash_mismatch(tmp_path):
    """Force the hash check inside _atomic_write_bytes to fail and assert
    the public surface (add_bytes) raises CASError without leaving artifacts."""
    store = ArtifactStore(tmp_path / "cas")
    data = b"some content"
    expected_sha = hashlib.sha256(data).hexdigest()

    # First call to _sha256_file is the verification of the temp file inside
    # _atomic_write_bytes. Returning a wrong hash there triggers the
    # "Hash verification failed" raise at cas_store.py:414.
    def lying_sha(path, *args, **kwargs):
        return "0" * 64

    with patch.object(store, "_sha256_file", side_effect=lying_sha):
        with pytest.raises(CASError) as exc_info:
            store.add_bytes(data)

    # The public exception must wrap the inner failure with context, but the
    # original "Hash verification failed" message must surface for diagnostics.
    msg = str(exc_info.value)
    assert "Hash verification failed" in msg
    assert expected_sha in msg

    # No object should have been published, and no temp files should leak.
    assert not (tmp_path / "cas" / "objects" / expected_sha[:2] / expected_sha).exists()
    leftover = list((tmp_path / "cas" / "objects").rglob("*.tmp"))
    assert not leftover


def test_add_file_hash_mismatch_message_contains_both_hashes(tmp_path):
    """The CASError message must include both expected and actual hashes so
    operators can correlate corruption with upstream sources."""
    store = ArtifactStore(tmp_path / "cas")
    src = tmp_path / "in.bin"
    src.write_bytes(b"hello world")
    expected_sha = hashlib.sha256(b"hello world").hexdigest()

    original_sha = store._sha256_file
    calls = {"n": 0}

    def lying_sha(path, *args, **kwargs):
        calls["n"] += 1
        # First call computes the source hash (must be honest), second call
        # validates the temp file (must lie to trigger the mismatch path).
        if calls["n"] == 1:
            return original_sha(path, *args, **kwargs)
        return "f" * 64

    with patch.object(store, "_sha256_file", side_effect=lying_sha):
        with pytest.raises(CASError) as exc_info:
            store.add_file(src)

    msg = str(exc_info.value)
    assert "Hash verification failed" in msg
    assert expected_sha in msg
    assert "f" * 64 in msg


def test_verify_object_returns_true_for_intact_object(tmp_path):
    store = ArtifactStore(tmp_path / "cas")
    obj = store.add_bytes(b"intact bytes")
    assert store.verify_object(obj.sha256) is True


def test_verify_object_returns_false_for_tampered_object(tmp_path):
    store = ArtifactStore(tmp_path / "cas")
    obj = store.add_bytes(b"original bytes")

    # Tamper with the on-disk object — simulate silent disk corruption.
    Path(obj.path).write_bytes(b"tampered!!")

    assert store.verify_object(obj.sha256) is False


def test_verify_object_raises_for_missing_object(tmp_path):
    store = ArtifactStore(tmp_path / "cas")
    bogus = "0" * 64
    with pytest.raises(CASError, match="CAS object missing"):
        store.verify_object(bogus)


def test_materialize_raises_for_missing_object(tmp_path):
    store = ArtifactStore(tmp_path / "cas")
    bogus = "0" * 64
    dest = tmp_path / "out.bin"
    with pytest.raises(CASError, match="CAS object missing"):
        store.materialize(bogus, dest)


def test_add_file_raises_for_missing_source(tmp_path):
    store = ArtifactStore(tmp_path / "cas")
    missing = tmp_path / "does_not_exist.bin"
    with pytest.raises(CASError, match="Source file does not exist"):
        store.add_file(missing)


def test_verify_object_is_case_insensitive_on_input_hash(tmp_path):
    """SHA-256 hashes are commonly normalized to lowercase, but callers
    sometimes pass uppercase. The store must accept both."""
    store = ArtifactStore(tmp_path / "cas")
    obj = store.add_bytes(b"case test")
    assert store.verify_object(obj.sha256.upper()) is True
    assert store.verify_object(obj.sha256.lower()) is True
