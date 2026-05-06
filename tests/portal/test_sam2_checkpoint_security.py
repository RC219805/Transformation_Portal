"""Unit tests for portal SAM2 checkpoint security helpers."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from transformation_portal.portal import sam2_checkpoint_security
from transformation_portal.portal.sam2_checkpoint_security import (
    Sam2CheckpointValidationError,
    _ManagedSam2BoundedChecksumCache,
    _ManagedSam2ChecksumCacheEntry,
    _resolve_managed_sam2_checkpoint_validation,
    _Sam2CacheKey,
)

pytestmark = [pytest.mark.unit, pytest.mark.security]


def _realpath(path: Path) -> Path:
    return Path(os.path.realpath(path))


def _cache() -> _ManagedSam2BoundedChecksumCache:
    return _ManagedSam2BoundedChecksumCache()


def _lock() -> threading.Lock:
    return threading.Lock()


def test_sam2_checkpoint_security_import_does_not_import_app() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from transformation_portal.portal import sam2_checkpoint_security; print('app' in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"


def test_direct_helper_import_resolves_repo_controlled_missing_checkpoint(tmp_path: Path) -> None:
    repo_root = _realpath(tmp_path / "repo")
    trusted_root = repo_root / "models" / "sam2"
    allowed_roots = [repo_root]
    checkpoint_path = "./models/sam2/sam2.1_hiera_large.pt"

    validation = _resolve_managed_sam2_checkpoint_validation(
        checkpoint_path,
        allowed_input_roots=allowed_roots,
        trusted_roots=[trusted_root],
        trusted_sha256=set(),
        checksum_max_bytes=1024,
        checksum_cache=_cache(),
        checksum_cache_lock=_lock(),
        repo_root=repo_root,
    )

    assert validation.reason is None
    assert validation.normalized_path == str(trusted_root / "sam2.1_hiera_large.pt")


def test_sam2_checkpoint_security_error_reasons_stay_distinct(tmp_path: Path) -> None:
    allowed_root = _realpath(tmp_path / "allowed")
    outside_root = _realpath(tmp_path / "outside")
    allowed_root.mkdir()
    outside_root.mkdir()
    outside_checkpoint = outside_root / "sam2-outside.pt"
    outside_checkpoint.write_bytes(b"outside")
    cache = _cache()
    lock = _lock()

    invalid = sam2_checkpoint_security._resolve_managed_sam2_checkpoint_validation(
        "bad\x00path",
        allowed_input_roots=[allowed_root],
        trusted_roots=[],
        trusted_sha256=set(),
        checksum_max_bytes=1024,
        checksum_cache=cache,
        checksum_cache_lock=lock,
    )
    outside = sam2_checkpoint_security._resolve_managed_sam2_checkpoint_validation(
        str(outside_checkpoint),
        allowed_input_roots=[allowed_root],
        trusted_roots=[],
        trusted_sha256=set(),
        checksum_max_bytes=1024,
        checksum_cache=cache,
        checksum_cache_lock=lock,
    )

    assert invalid.reason == "invalid_path_value"
    assert outside.reason == "path_outside_allowed_roots"


def test_sam2_checkpoint_security_external_checkpoint_requires_trusted_digest(tmp_path: Path) -> None:
    input_root = _realpath(tmp_path)
    checkpoint_path = input_root / "sam2-governed.pt"
    checkpoint_bytes = b"trusted checkpoint bytes"
    checkpoint_path.write_bytes(checkpoint_bytes)
    digest = hashlib.sha256(checkpoint_bytes).hexdigest()
    cache = _cache()
    lock = _lock()

    trusted = sam2_checkpoint_security._resolve_managed_sam2_checkpoint_validation(
        str(checkpoint_path),
        allowed_input_roots=[input_root],
        trusted_roots=[],
        trusted_sha256={digest},
        checksum_max_bytes=1024,
        checksum_cache=cache,
        checksum_cache_lock=lock,
    )

    assert trusted.reason is None
    assert trusted.normalized_path == str(checkpoint_path)

    sam2_checkpoint_security._clear_managed_sam2_checksum_cache(cache, lock)
    untrusted = sam2_checkpoint_security._resolve_managed_sam2_checkpoint_validation(
        str(checkpoint_path),
        allowed_input_roots=[input_root],
        trusted_roots=[],
        trusted_sha256=set(),
        checksum_max_bytes=1024,
        checksum_cache=cache,
        checksum_cache_lock=lock,
    )

    assert untrusted.normalized_path is None
    assert untrusted.reason == "untrusted_checkpoint_path"


def test_sam2_checkpoint_security_oversized_checkpoint_fails_before_hashing(tmp_path: Path) -> None:
    input_root = _realpath(tmp_path)
    checkpoint_path = input_root / "sam2-oversized.pt"
    checkpoint_path.write_bytes(b"oversized")

    validation = sam2_checkpoint_security._resolve_managed_sam2_checkpoint_validation(
        str(checkpoint_path),
        allowed_input_roots=[input_root],
        trusted_roots=[],
        trusted_sha256=set(),
        checksum_max_bytes=1,
        checksum_cache=_cache(),
        checksum_cache_lock=_lock(),
        hash_file_sha256=lambda path: pytest.fail(f"unexpected hash for {path}"),
    )

    assert validation.normalized_path is None
    assert validation.reason == "checkpoint_file_too_large"


def test_sam2_checkpoint_security_validate_raises_reasoned_error(tmp_path: Path) -> None:
    input_root = _realpath(tmp_path)
    checkpoint_path = input_root / "sam2-untrusted.pt"
    checkpoint_path.write_bytes(b"untrusted")

    with pytest.raises(Sam2CheckpointValidationError) as exc_info:
        sam2_checkpoint_security._validate_managed_sam2_checkpoint_path(
            str(checkpoint_path),
            allowed_input_roots=[input_root],
            trusted_roots=[],
            trusted_sha256=set(),
            checksum_max_bytes=1024,
            checksum_cache=_cache(),
            checksum_cache_lock=_lock(),
        )

    assert exc_info.value.reason == "untrusted_checkpoint_path"
    assert str(exc_info.value) == "SAM2 checkpoint path is not trusted"


def test_sam2_checkpoint_security_checksum_cache_key_shape_tracks_file_identity(tmp_path: Path) -> None:
    checkpoint_path = _realpath(tmp_path / "sam2-keyed.pt")
    checkpoint_path.write_bytes(b"checkpoint")

    cache_key = sam2_checkpoint_security._managed_sam2_checksum_cache_key(checkpoint_path)

    assert isinstance(cache_key, _Sam2CacheKey)
    assert cache_key.path == str(checkpoint_path)
    assert cache_key.size == len(b"checkpoint")
    assert isinstance(cache_key.mtime_ns, int)
    assert isinstance(cache_key.dev, int)
    assert isinstance(cache_key.ino, int)
    assert isinstance(cache_key.ctime_ns, int)


def test_sam2_checkpoint_security_checksum_cache_reuses_hash_results_and_clear_resets_state(tmp_path: Path) -> None:
    checkpoint_path = _realpath(tmp_path / "sam2-cached.pt")
    checkpoint_bytes = b"cached checkpoint bytes"
    checkpoint_path.write_bytes(checkpoint_bytes)
    digest = hashlib.sha256(checkpoint_bytes).hexdigest()
    cache = _cache()
    lock = _lock()
    hash_calls: list[Path] = []

    def _counting_hash(path: Path) -> str:
        hash_calls.append(path)
        return digest

    first = sam2_checkpoint_security._cached_managed_sam2_checksum_result(
        checkpoint_path,
        trusted_sha256={digest},
        checksum_max_bytes=1024,
        checksum_cache=cache,
        checksum_cache_lock=lock,
        hash_file_sha256=_counting_hash,
    )
    second = sam2_checkpoint_security._cached_managed_sam2_checksum_result(
        checkpoint_path,
        trusted_sha256={digest},
        checksum_max_bytes=1024,
        checksum_cache=cache,
        checksum_cache_lock=lock,
        hash_file_sha256=_counting_hash,
    )

    assert first == _ManagedSam2ChecksumCacheEntry(digest=digest, reason=None)
    assert second == first
    assert hash_calls == [checkpoint_path]
    assert len(cache) == 1

    sam2_checkpoint_security._clear_managed_sam2_checksum_cache(cache, lock)

    assert len(cache) == 0


def test_sam2_checkpoint_security_bounded_checksum_cache_preserves_fifo_eviction() -> None:
    cache = _ManagedSam2BoundedChecksumCache(max_entries=2)
    entry = _ManagedSam2ChecksumCacheEntry(digest="abc", reason=None)
    key1 = _Sam2CacheKey("/path/a.pt", 100, 1000, 1, 1001, 2000)
    key2 = _Sam2CacheKey("/path/b.pt", 200, 1000, 1, 1002, 2000)
    key3 = _Sam2CacheKey("/path/c.pt", 300, 1000, 1, 1003, 2000)

    cache[key1] = entry
    cache[key2] = entry
    cache[key3] = entry

    assert key1 not in cache
    assert key2 in cache
    assert key3 in cache
    assert len(cache) == 2

    cache[key2] = _ManagedSam2ChecksumCacheEntry(digest="updated", reason=None)
    assert len(cache) == 2
    assert cache[key2].digest == "updated"

    cache.clear()

    assert len(cache) == 0
    assert len(cache._insertion_order) == 0
