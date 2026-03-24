"""Thread-safety and concurrency tests for DepthCache.

These tests verify:
1. Concurrent same-key writes don't produce failures
2. Concurrent different-key writes are isolated
3. Concurrent reads during writes are safe
4. Stats collection is thread-safe
5. Cache eviction under concurrent load is safe
6. Clear operation is thread-safe
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import numpy as np
import pytest

from transformation_portal.lux_depth_v3.depth_cache import DepthCache

pytestmark = pytest.mark.unit


class TestDepthCacheConcurrencySameKey:
    """Tests for concurrent operations on the same cache key."""

    def test_concurrent_same_key_writes_no_failures(self, tmp_path, caplog) -> None:
        """Concurrent same-key writes should not produce internal store-failure warnings."""
        cache = DepthCache(tmp_path, max_size_gb=1.0)

        def store_depth(value: int) -> None:
            depth = np.full((100, 100), value, dtype=np.float32)
            cache.store("same_image", "same_config", depth)

        with caplog.at_level(logging.WARNING, logger="transformation_portal.lux_depth_v3.depth_cache"):
            threads = [threading.Thread(target=store_depth, args=(i,)) for i in range(10)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        assert not any("Failed to cache depth" in record.message for record in caplog.records)

        cached = cache.get("same_image", "same_config")
        assert cached is not None

    def test_concurrent_same_key_writes_produces_valid_result(self, tmp_path) -> None:
        """Concurrent same-key writes should produce a valid cached result."""
        cache = DepthCache(tmp_path, max_size_gb=1.0)
        results: List[bool] = []

        def store_and_verify(value: int) -> None:
            depth = np.full((50, 50), float(value), dtype=np.float32)
            cache.store("key1", "config1", depth)
            # Immediately try to read back
            cached = cache.get("key1", "config1")
            results.append(cached is not None and cached.shape == (50, 50))

        threads = [threading.Thread(target=store_and_verify, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All stores should succeed
        assert all(results)
        # Final cached value should be valid
        final = cache.get("key1", "config1")
        assert final is not None
        assert final.dtype == np.float32


class TestDepthCacheConcurrencyDifferentKeys:
    """Tests for concurrent operations on different cache keys."""

    def test_concurrent_different_key_writes_isolated(self, tmp_path) -> None:
        """Concurrent writes to different keys should be isolated."""
        cache = DepthCache(tmp_path, max_size_gb=1.0)
        num_keys = 20

        def store_unique_key(idx: int) -> None:
            depth = np.full((30, 30), float(idx), dtype=np.float32)
            cache.store(f"image_{idx}", "config", depth)

        threads = [threading.Thread(target=store_unique_key, args=(i,)) for i in range(num_keys)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All keys should be stored correctly
        for i in range(num_keys):
            cached = cache.get(f"image_{i}", "config")
            assert cached is not None, f"Key image_{i} not found"
            assert cached.shape == (30, 30)
            # Values should match what was stored
            assert np.allclose(cached, float(i))

    def test_concurrent_mixed_operations(self, tmp_path) -> None:
        """Concurrent mixed read/write operations should be safe."""
        cache = DepthCache(tmp_path, max_size_gb=1.0)

        # Pre-populate some entries
        for i in range(5):
            depth = np.full((25, 25), float(i), dtype=np.float32)
            cache.store(f"pre_{i}", "config", depth)

        errors: List[Exception] = []

        def read_operation(idx: int) -> None:
            try:
                cached = cache.get(f"pre_{idx % 5}", "config")
                # Should be either the original value or None if evicted
                if cached is not None:
                    assert cached.shape == (25, 25)
            except Exception as e:
                errors.append(e)

        def write_operation(idx: int) -> None:
            try:
                depth = np.full((25, 25), float(idx + 100), dtype=np.float32)
                cache.store(f"new_{idx}", "config", depth)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(10):
            threads.append(threading.Thread(target=read_operation, args=(i,)))
            threads.append(threading.Thread(target=write_operation, args=(i,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent operations: {errors}"


class TestDepthCacheConcurrencyStats:
    """Tests for thread-safe statistics collection."""

    def test_stats_during_concurrent_writes(self, tmp_path) -> None:
        """Stats collection should be thread-safe during concurrent writes."""
        cache = DepthCache(tmp_path, max_size_gb=1.0)
        stats_results: List[dict] = []
        errors: List[Exception] = []

        def store_entry(idx: int) -> None:
            try:
                depth = np.full((20, 20), float(idx), dtype=np.float32)
                cache.store(f"key_{idx}", "config", depth)
            except Exception as e:
                errors.append(e)

        def collect_stats() -> None:
            try:
                for _ in range(5):
                    stats = cache.stats()
                    stats_results.append(stats)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(15):
            threads.append(threading.Thread(target=store_entry, args=(i,)))
        threads.append(threading.Thread(target=collect_stats))
        threads.append(threading.Thread(target=collect_stats))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent stats: {errors}"
        # All stats should have valid structure
        for stats in stats_results:
            assert "entry_count" in stats
            assert "size_gb" in stats
            assert "max_size_gb" in stats
            assert stats["entry_count"] >= 0
            assert stats["size_gb"] >= 0.0


class TestDepthCacheConcurrencyClear:
    """Tests for thread-safe cache clearing."""

    def test_clear_during_concurrent_writes(self, tmp_path) -> None:
        """Clear operation should not raise exceptions during concurrent writes.

        Note: When clear() runs concurrently with store operations, some stores may
        fail gracefully because their temporary files get deleted. This is expected
        behavior - the test verifies no uncaught exceptions propagate, not that all
        operations succeed.
        """
        cache = DepthCache(tmp_path, max_size_gb=1.0)
        exceptions_raised: List[Exception] = []

        def store_entries(idx: int) -> None:
            try:
                for j in range(3):
                    depth = np.full((15, 15), float(idx * 10 + j), dtype=np.float32)
                    cache.store(f"key_{idx}_{j}", "config", depth)
            except Exception as e:
                # Store operations may fail during concurrent clear - this is OK
                # as long as exceptions are caught and logged, not propagated
                exceptions_raised.append(e)

        def clear_cache() -> None:
            try:
                cache.clear()
            except Exception as e:
                exceptions_raised.append(e)

        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=store_entries, args=(i,)))
        threads.append(threading.Thread(target=clear_cache))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No uncaught exceptions should propagate from the threads
        # (Warnings are acceptable and expected for race conditions)
        assert len(exceptions_raised) == 0, f"Uncaught exceptions during concurrent clear: {exceptions_raised}"


class TestDepthCacheConcurrencyThreadPool:
    """Tests using ThreadPoolExecutor for high-concurrency scenarios."""

    def test_high_concurrency_with_thread_pool(self, tmp_path) -> None:
        """High-concurrency operations using ThreadPoolExecutor should be safe.

        With a 1GB cache and 50 small operations (10x10 float32 = 400 bytes each),
        no eviction should occur, so all operations should succeed.
        """
        cache = DepthCache(tmp_path, max_size_gb=1.0)

        def operation(idx: int) -> bool:
            depth = np.full((10, 10), float(idx), dtype=np.float32)
            cache.store(f"pool_key_{idx}", "config", depth)
            cached = cache.get(f"pool_key_{idx}", "config")
            return cached is not None

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(operation, i) for i in range(50)]
            results = [f.result() for f in as_completed(futures)]

        # All operations should succeed - total size is only ~20KB, well under 1GB limit
        success_count = sum(1 for r in results if r)
        assert success_count == 50, f"Only {success_count}/50 operations succeeded"

    def test_concurrent_approximate_size_tracking(self, tmp_path) -> None:
        """Approximate size tracking should remain consistent under concurrent load."""
        cache = DepthCache(tmp_path, max_size_gb=10.0)

        def store_entry(idx: int) -> None:
            depth = np.full((50, 50), float(idx), dtype=np.float32)
            cache.store(f"size_key_{idx}", "config", depth)

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(store_entry, range(40)))

        # Stats should reflect stored entries
        stats = cache.stats()
        assert stats["entry_count"] == 40

        # Expected size calculation:
        # 40 entries * (50*50*4 bytes per entry) = 400,000 bytes ≈ 0.000373 GB
        expected_size_gb = 40 * (50 * 50 * 4) / (1024**3)

        # Use relative tolerance (10%) rather than absolute, as it's more robust
        # for varying cache sizes and accounts for filesystem overhead
        assert stats["size_gb"] == pytest.approx(expected_size_gb, rel=0.1)
