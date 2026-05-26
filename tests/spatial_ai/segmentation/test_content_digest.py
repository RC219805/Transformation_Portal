"""Tests for the shared segmentation content-digest layer (N-3).

Covers:

* File SHA-256 memoization (cache hit ⇒ O(stat), cache miss on mtime
  change, perf regression assertion).
* ``SAM2CheckpointIntegrityError`` fail-closed path remains observable
  after the memoization refactor.
* Per-instance ``ArrayDigestCache`` reuse + override threading from
  ``SegmentationInput.content_digest``.
* Bit-identical digest formula versus the legacy in-module
  implementations so segmentation cache keys do not silently change.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from transformation_portal.spatial_ai.segmentation._content_digest import (
    ArrayDigestCache,
    compute_array_sha256,
    compute_file_sha256,
    compute_file_sha256_for_path,
)

pytestmark = pytest.mark.unit


def _write_file(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    return path


def _legacy_array_digest(arr: np.ndarray) -> str:
    """Reference implementation of the legacy in-module formula.

    Both ``_cache._stable_array_hash`` and ``sam2_backend._stable_image_hash``
    previously used this exact sequence (shape repr + dtype repr + raw
    uint8 buffer view). This local copy serves as a regression oracle.
    """
    arr = arr if arr.flags.c_contiguous else np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


@pytest.fixture(autouse=True)
def _isolate_lru_cache():
    """Each test sees a fresh LRU so cache state doesn't leak across tests."""
    compute_file_sha256.cache_clear()
    yield
    compute_file_sha256.cache_clear()


class TestFileDigestMemoization:
    def test_repeat_hash_of_same_file_is_dramatically_faster(self, tmp_path: Path) -> None:
        # 8 MiB payload — large enough that the first hash is measurably
        # I/O+CPU bound (single-digit ms minimum on typical CI runners),
        # and dwarfs the O(stat) cost of a cache hit.
        target = _write_file(tmp_path / "checkpoint.bin", os.urandom(8 * 1024 * 1024))

        t0 = time.perf_counter()
        digest_first = compute_file_sha256_for_path(target)
        t1 = time.perf_counter()
        digest_second = compute_file_sha256_for_path(target)
        t2 = time.perf_counter()

        first_elapsed = t1 - t0
        second_elapsed = t2 - t1

        assert digest_first == digest_second
        # The second call should be at least 20× faster (in practice 100×+
        # because it's a dict lookup) — a conservative floor leaves
        # headroom for slow runners while still catching a regression
        # that bypasses the cache.
        assert second_elapsed * 20 < first_elapsed, (
            f"expected ≥20× speedup from cache hit; " f"first={first_elapsed:.4f}s second={second_elapsed:.4f}s"
        )

    def test_mtime_change_invalidates_cache(self, tmp_path: Path) -> None:
        target = _write_file(tmp_path / "checkpoint.bin", b"alpha-content")
        digest_before = compute_file_sha256_for_path(target)

        # Overwrite with different content and bump mtime far enough that
        # even coarse filesystems register the change.
        target.write_bytes(b"beta-content")
        new_mtime_ns = target.stat().st_mtime_ns + 10_000_000_000  # +10s
        os.utime(target, ns=(new_mtime_ns, new_mtime_ns))

        digest_after = compute_file_sha256_for_path(target)

        assert digest_before != digest_after, "fresh content with new mtime must invalidate the cache"
        assert digest_after == hashlib.sha256(b"beta-content").hexdigest()

    def test_explicit_size_mtime_keys_route_to_lru(self, tmp_path: Path) -> None:
        target = _write_file(tmp_path / "checkpoint.bin", b"x" * 1024)
        stat = target.stat()

        # First call populates the LRU.
        first = compute_file_sha256(str(target), stat.st_size, stat.st_mtime_ns)
        info_before = compute_file_sha256.cache_info()

        # Same (path, size, mtime_ns) tuple → hit (no new misses).
        second = compute_file_sha256(str(target), stat.st_size, stat.st_mtime_ns)
        info_after = compute_file_sha256.cache_info()

        assert first == second
        assert info_after.hits == info_before.hits + 1
        assert info_after.misses == info_before.misses


class TestCheckpointIntegrityStillFailsClosed:
    """SAM2 ``_validate_checkpoint_sha256`` correctness is preserved after N-3."""

    def test_mismatch_raises_typed_error(self, tmp_path: Path) -> None:
        # Import lazily so this test runs in the core lane without
        # pulling SAM2 deps.
        from transformation_portal.spatial_ai.segmentation.sam2_backend import (
            SAM2Backend,
            SAM2CheckpointIntegrityError,
        )

        target = _write_file(tmp_path / "fake_checkpoint.pt", b"not-a-real-checkpoint")
        expected = "0" * 64  # cannot match the real SHA-256 of the bytes

        with pytest.raises(SAM2CheckpointIntegrityError, match="SHA-256 mismatch"):
            SAM2Backend._validate_checkpoint_sha256(target, expected)

    def test_match_does_not_raise(self, tmp_path: Path) -> None:
        from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

        payload = b"trusted-bytes"
        target = _write_file(tmp_path / "trusted_checkpoint.pt", payload)
        expected = hashlib.sha256(payload).hexdigest()

        # No exception means the cached digest matched.
        SAM2Backend._validate_checkpoint_sha256(target, expected)

    def test_mismatch_still_raises_when_cache_is_warm(self, tmp_path: Path) -> None:
        """A populated LRU entry must not mask a subsequent mismatch on a
        different expected digest — proves memoization doesn't weaken
        the integrity check."""
        from transformation_portal.spatial_ai.segmentation.sam2_backend import (
            SAM2Backend,
            SAM2CheckpointIntegrityError,
        )

        payload = b"trusted-bytes"
        target = _write_file(tmp_path / "trusted_checkpoint.pt", payload)
        true_digest = hashlib.sha256(payload).hexdigest()

        SAM2Backend._validate_checkpoint_sha256(target, true_digest)  # warms cache
        with pytest.raises(SAM2CheckpointIntegrityError, match="SHA-256 mismatch"):
            SAM2Backend._validate_checkpoint_sha256(target, "0" * 64)


class TestArrayDigestCache:
    def test_same_array_hits_cache(self) -> None:
        cache = ArrayDigestCache()
        arr = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)

        # Mock the canonical helper so we can count cache misses.
        with patch(
            "transformation_portal.spatial_ai.segmentation._content_digest.compute_array_sha256",
            side_effect=compute_array_sha256,
        ) as spy:
            first = cache.get_or_compute(arr)
            second = cache.get_or_compute(arr)

        assert first == second
        assert spy.call_count == 1, "second call must hit the cache"

    def test_none_returns_sentinel(self) -> None:
        cache = ArrayDigestCache()
        assert cache.get_or_compute(None) == "none"

    def test_override_short_circuits(self) -> None:
        cache = ArrayDigestCache()
        arr = np.zeros((4, 4, 3), dtype=np.float32)
        override = "deadbeef" * 8

        with patch(
            "transformation_portal.spatial_ai.segmentation._content_digest.compute_array_sha256",
        ) as spy:
            result = cache.get_or_compute(arr, override=override)

        assert result == override
        assert spy.call_count == 0, "override must not trigger a fresh hash"

        # Subsequent calls without override pick up the cached override.
        cached = cache.get_or_compute(arr)
        assert cached == override

    def test_clear_drops_entries(self) -> None:
        cache = ArrayDigestCache()
        arr = np.zeros((4, 4, 3), dtype=np.float32)
        cache.get_or_compute(arr)
        cache.clear()

        with patch(
            "transformation_portal.spatial_ai.segmentation._content_digest.compute_array_sha256",
            side_effect=compute_array_sha256,
        ) as spy:
            cache.get_or_compute(arr)

        assert spy.call_count == 1, "post-clear lookup must miss the cache"


class TestLegacyFormulaPreserved:
    """Bit-identical output guards segmentation cache keys from drift."""

    def test_compute_array_sha256_matches_legacy_formula(self) -> None:
        rng = np.random.default_rng(seed=42)
        arr = rng.random((128, 192, 3), dtype=np.float32)

        assert compute_array_sha256(arr) == _legacy_array_digest(arr)

    def test_non_contiguous_array_is_normalized_before_hashing(self) -> None:
        base = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
        view = base[::2]  # non-contiguous

        assert not view.flags.c_contiguous
        assert compute_array_sha256(view) == _legacy_array_digest(view)


class TestSegmentationInputThreading:
    """``SegmentationInput.content_digest`` plumbs precomputed digests."""

    def test_field_defaults_to_none(self) -> None:
        from transformation_portal.spatial_ai.segmentation.contracts import SegmentationInput

        seg_input = SegmentationInput(
            image=np.zeros((4, 4, 3), dtype=np.float32),
            gamma=1.0,
            mode="auto",
        )
        assert seg_input.content_digest is None

    def test_field_accepts_explicit_digest(self) -> None:
        from transformation_portal.spatial_ai.segmentation.contracts import SegmentationInput

        digest = "a" * 64
        seg_input = SegmentationInput(
            image=np.zeros((4, 4, 3), dtype=np.float32),
            gamma=1.0,
            mode="auto",
            content_digest=digest,
        )
        assert seg_input.content_digest == digest
