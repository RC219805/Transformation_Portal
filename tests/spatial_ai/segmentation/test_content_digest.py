"""Tests for the shared segmentation content-digest layer (N-3).

Covers:

* File SHA-256 memoization: deterministic ``cache_info()`` hit/miss
  assertions (no wall-clock dependency) + stat-key invalidation that
  isolates the mtime field from the size field.
* The dedicated ``compute_file_sha256_uncached`` integrity path always
  re-streams the bytes, so ``_validate_checkpoint_sha256`` correctness
  survives a same-size, mtime-restored rewrite that would fool the
  stat-keyed LRU.
* ``SAM2CheckpointIntegrityError`` remains observable.
* Per-instance ``ArrayDigestCache`` reuse + override threading from
  ``SegmentationInput.content_digest`` + bounded LRU eviction.
* ``SegmentationInput`` rejects malformed ``content_digest`` values at
  the contract boundary.
* Bit-identical digest formula versus the legacy in-module
  implementations so segmentation cache keys do not silently change.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from transformation_portal.spatial_ai.segmentation._content_digest import (
    ArrayDigestCache,
    compute_array_sha256,
    compute_file_sha256,
    compute_file_sha256_for_path,
    compute_file_sha256_uncached,
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
    # pylint: disable=no-value-for-parameter  # `lru_cache` adds cache_clear/cache_info; pylint can't infer this.
    compute_file_sha256.cache_clear()
    yield
    compute_file_sha256.cache_clear()  # pylint: disable=no-value-for-parameter


class TestFileDigestMemoization:
    """Deterministic cache-hit assertions on the informational LRU path."""

    def test_repeat_lookup_hits_lru(self, tmp_path: Path) -> None:
        target = _write_file(tmp_path / "checkpoint.bin", b"deterministic-payload")

        digest_first = compute_file_sha256_for_path(target)
        info_after_first = compute_file_sha256.cache_info()  # pylint: disable=no-value-for-parameter

        digest_second = compute_file_sha256_for_path(target)
        info_after_second = compute_file_sha256.cache_info()  # pylint: disable=no-value-for-parameter

        assert digest_first == digest_second
        # Second call must be served from the LRU — exactly one extra hit,
        # zero new misses. This is the deterministic equivalent of the
        # previous wall-clock-based regression check.
        assert info_after_second.hits == info_after_first.hits + 1
        assert info_after_second.misses == info_after_first.misses

    def test_mtime_only_change_invalidates_cache(self, tmp_path: Path) -> None:
        """Use same-size payloads so the assertion exercises the mtime
        component of the cache key, not the size component."""
        same_size_a = b"alpha-content"
        same_size_b = b"omega-content"
        assert len(same_size_a) == len(same_size_b)

        target = _write_file(tmp_path / "checkpoint.bin", same_size_a)
        original_stat = target.stat()
        digest_before = compute_file_sha256_for_path(target)

        # Rewrite with same-size, different content; advance mtime far
        # enough that even coarse filesystems register the change.
        target.write_bytes(same_size_b)
        bumped_mtime_ns = original_stat.st_mtime_ns + 10_000_000_000  # +10s
        os.utime(target, ns=(bumped_mtime_ns, bumped_mtime_ns))

        new_stat = target.stat()
        # Sanity: size is unchanged, so any cache hit would have to come
        # from a non-mtime key component.
        assert new_stat.st_size == original_stat.st_size

        digest_after = compute_file_sha256_for_path(target)

        assert digest_before != digest_after, "mtime change with same size must invalidate the cache"
        assert digest_after == hashlib.sha256(same_size_b).hexdigest()

    def test_full_identity_tuple_keys_the_lru(self, tmp_path: Path) -> None:
        target = _write_file(tmp_path / "checkpoint.bin", b"x" * 1024)
        stat = target.stat()

        # pylint: disable=no-value-for-parameter
        first = compute_file_sha256(
            str(target),
            int(stat.st_dev),
            int(stat.st_ino),
            int(stat.st_size),
            int(stat.st_mtime_ns),
            int(stat.st_ctime_ns),
        )
        info_before = compute_file_sha256.cache_info()
        second = compute_file_sha256(
            str(target),
            int(stat.st_dev),
            int(stat.st_ino),
            int(stat.st_size),
            int(stat.st_mtime_ns),
            int(stat.st_ctime_ns),
        )
        info_after = compute_file_sha256.cache_info()
        # pylint: enable=no-value-for-parameter

        assert first == second
        assert info_after.hits == info_before.hits + 1
        assert info_after.misses == info_before.misses


class TestUncachedIntegrityPath:
    """``compute_file_sha256_uncached`` ignores the LRU and always reads bytes."""

    def test_uncached_bypasses_lru(self, tmp_path: Path) -> None:
        target = _write_file(tmp_path / "checkpoint.bin", b"trusted-bytes")

        # Warm the informational LRU.
        warmed = compute_file_sha256_for_path(target)
        info_before = compute_file_sha256.cache_info()  # pylint: disable=no-value-for-parameter

        # The uncached helper must not register any new LRU hit or miss
        # — proof that integrity validation does not consult the cache.
        fresh = compute_file_sha256_uncached(target)
        info_after = compute_file_sha256.cache_info()  # pylint: disable=no-value-for-parameter

        assert warmed == fresh
        assert info_after.hits == info_before.hits
        assert info_after.misses == info_before.misses

    def test_uncached_detects_same_size_mtime_restored_overwrite(self, tmp_path: Path) -> None:
        """The exact attack the reviewers flagged: same size, mtime restored
        to the original value. Stat-keyed memoization would return a stale
        digest; ``compute_file_sha256_uncached`` reads fresh bytes."""
        original = b"trusted-bytes"
        replacement = b"tampered-data"
        assert len(original) == len(replacement)

        target = _write_file(tmp_path / "checkpoint.bin", original)
        original_stat = target.stat()
        baseline_digest = compute_file_sha256_uncached(target)
        assert baseline_digest == hashlib.sha256(original).hexdigest()

        target.write_bytes(replacement)
        # Restore mtime to the original value — the attack vector the
        # reviewers identified for the stat-keyed cache.
        os.utime(target, ns=(original_stat.st_mtime_ns, original_stat.st_mtime_ns))

        tampered_digest = compute_file_sha256_uncached(target)
        assert tampered_digest != baseline_digest
        assert tampered_digest == hashlib.sha256(replacement).hexdigest()


class TestCheckpointIntegrityStillFailsClosed:
    """SAM2 ``_validate_checkpoint_sha256`` correctness is preserved after N-3."""

    def test_mismatch_raises_typed_error(self, tmp_path: Path) -> None:
        from transformation_portal.spatial_ai.segmentation.sam2_backend import (
            SAM2Backend,
            SAM2CheckpointIntegrityError,
        )

        target = _write_file(tmp_path / "fake_checkpoint.pt", b"not-a-real-checkpoint")
        expected = "0" * 64

        with pytest.raises(SAM2CheckpointIntegrityError, match="SHA-256 mismatch"):
            SAM2Backend._validate_checkpoint_sha256(target, expected)

    def test_match_does_not_raise(self, tmp_path: Path) -> None:
        from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

        payload = b"trusted-bytes"
        target = _write_file(tmp_path / "trusted_checkpoint.pt", payload)
        expected = hashlib.sha256(payload).hexdigest()

        SAM2Backend._validate_checkpoint_sha256(target, expected)  # no exception

    def test_repeat_validation_rereads_bytes(self, tmp_path: Path) -> None:
        """Two consecutive validations must both stream the file — proves
        the integrity path is not memoized regardless of stat tuple."""
        from transformation_portal.spatial_ai.segmentation import sam2_backend as _backend

        payload = b"trusted-bytes"
        target = _write_file(tmp_path / "trusted_checkpoint.pt", payload)
        expected = hashlib.sha256(payload).hexdigest()

        with patch(
            "transformation_portal.spatial_ai.segmentation._content_digest._stream_sha256",
            wraps=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(),
        ) as spy:
            _backend.SAM2Backend._validate_checkpoint_sha256(target, expected)
            _backend.SAM2Backend._validate_checkpoint_sha256(target, expected)

        assert spy.call_count == 2, "integrity validation must re-stream bytes on every call"


class TestArrayDigestCache:
    def test_same_array_hits_cache(self) -> None:
        cache = ArrayDigestCache()
        arr = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)

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

    def test_bounded_lru_eviction(self) -> None:
        """A long-running backend must not grow this cache unbounded."""
        cache = ArrayDigestCache(maxsize=3)
        arrays = [np.full((2, 2, 3), value, dtype=np.float32) for value in range(5)]

        for arr in arrays:
            cache.get_or_compute(arr)

        # Only the last 3 ids should remain.
        assert len(cache) == 3

    def test_rejects_non_positive_maxsize(self) -> None:
        with pytest.raises(ValueError):
            ArrayDigestCache(maxsize=0)


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

    def test_field_normalises_to_lowercase(self) -> None:
        from transformation_portal.spatial_ai.segmentation.contracts import SegmentationInput

        digest = "A" * 64
        seg_input = SegmentationInput(
            image=np.zeros((4, 4, 3), dtype=np.float32),
            gamma=1.0,
            mode="auto",
            content_digest=digest,
        )
        assert seg_input.content_digest == "a" * 64

    @pytest.mark.parametrize(
        "bad_digest",
        [
            "short",  # too short
            "z" * 64,  # non-hex chars
            "a" * 63,  # one char short
            "a" * 65,  # one char long
        ],
    )
    def test_field_rejects_malformed_digest(self, bad_digest: str) -> None:
        from transformation_portal.spatial_ai.segmentation.contracts import SegmentationInput

        with pytest.raises(ValueError, match="content_digest must be a 64-character SHA-256 hex string"):
            SegmentationInput(
                image=np.zeros((4, 4, 3), dtype=np.float32),
                gamma=1.0,
                mode="auto",
                content_digest=bad_digest,
            )
