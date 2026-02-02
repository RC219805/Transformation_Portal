"""Tests for Phase 2 performance optimizations.

Validates:
1. Content-addressable depth cache (store/retrieve/eviction)
2. Parallel batch processing (correctness and race conditions)
3. Configuration flags for parallelization
4. Cache hit rate tracking
5. Graceful fallback to sequential processing
"""
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pytest
import time

from transformation_portal.lux_depth_v3.config import EnhanceConfig
from transformation_portal.lux_depth_v3.depth_cache import DepthCache
from transformation_portal.lux_depth_v3.input_manager import ImageInput


class TestDepthCache:
    """Test content-addressable depth cache."""

    def test_cache_store_and_retrieve(self, tmp_path):
        """Verify cache can store and retrieve depth maps."""
        cache = DepthCache(tmp_path, max_size_gb=1.0)

        # Create test depth map
        depth = np.random.rand(100, 100).astype(np.float32)
        image_hash = "test_image_sha256"
        config_hash = "test_config_fp"

        # Store in cache
        cache.store(image_hash, config_hash, depth)

        # Retrieve from cache
        retrieved = cache.get(image_hash, config_hash)

        assert retrieved is not None
        assert np.allclose(retrieved, depth)

    def test_cache_miss_returns_none(self, tmp_path):
        """Verify cache miss returns None."""
        cache = DepthCache(tmp_path)

        result = cache.get("nonexistent_image", "nonexistent_config")

        assert result is None

    def test_cache_invalidation_on_config_change(self, tmp_path):
        """Verify different config produces cache miss."""
        cache = DepthCache(tmp_path)

        depth = np.random.rand(100, 100).astype(np.float32)
        image_hash = "same_image"
        config_hash_1 = "config_v1"
        config_hash_2 = "config_v2"

        # Store with config v1
        cache.store(image_hash, config_hash_1, depth)

        # Query with config v2 should miss
        result = cache.get(image_hash, config_hash_2)

        assert result is None

    def test_cache_eviction_on_size_limit(self, tmp_path):
        """Verify LRU eviction when cache exceeds size limit."""
        # Set very small size limit
        cache = DepthCache(tmp_path, max_size_gb=0.001)  # 1MB

        # Store multiple large depth maps to trigger eviction
        for i in range(10):
            depth = np.random.rand(500, 500).astype(np.float32)  # ~1MB each
            cache.store(f"image_{i}", "config", depth)

        # Cache should have evicted some entries
        stats = cache.stats()
        assert stats['entry_count'] < 10

    def test_cache_stats(self, tmp_path):
        """Verify cache statistics are accurate."""
        cache = DepthCache(tmp_path)

        # Store some depth maps
        for i in range(5):
            depth = np.random.rand(100, 100).astype(np.float32)
            cache.store(f"image_{i}", "config", depth)

        stats = cache.stats()

        assert stats['entry_count'] == 5
        assert stats['size_gb'] > 0
        assert stats['max_size_gb'] == 10.0

    def test_cache_clear(self, tmp_path):
        """Verify cache can be cleared."""
        cache = DepthCache(tmp_path)

        # Store depth maps
        for i in range(3):
            depth = np.random.rand(100, 100).astype(np.float32)
            cache.store(f"image_{i}", "config", depth)

        # Clear cache
        cache.clear()

        # Verify all entries removed
        stats = cache.stats()
        assert stats['entry_count'] == 0

    def test_cache_atomic_writes(self, tmp_path):
        """Verify cache writes are atomic (no partial files)."""
        cache = DepthCache(tmp_path)

        depth = np.random.rand(1000, 1000).astype(np.float32)

        # Store depth
        cache.store("test_image", "test_config", depth)

        # Verify no .tmp files left behind
        tmp_files = list(cache.cache_dir.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_cache_corrupted_file_handling(self, tmp_path):
        """Verify cache handles corrupted files gracefully."""
        cache = DepthCache(tmp_path)

        # Create corrupted cache file
        corrupted_path = cache.cache_dir / "corrupted_hash.npy"
        corrupted_path.write_text("not a valid numpy file")

        # Should return None for corrupted file
        result = cache.get("corrupted", "hash")

        assert result is None


class TestParallelProcessing:
    """Test parallel batch processing."""

    @pytest.fixture
    def mock_orchestrator(self, tmp_path):
        """Create mock orchestrator with parallel processing enabled."""
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator
        from transformation_portal.lux_depth_v3.config import EnhanceConfig, ModelVariant

        config = EnhanceConfig(
            model_variant=ModelVariant.METRIC_LARGE,
            enable_parallel_processing=True,
            max_parallel_workers=2,
            enable_v2=False,  # Disable V2 for simpler testing
        )

        with patch('transformation_portal.lux_depth_v3.orchestrator.DA3InferenceEngine'):
            orch = EnhanceOrchestrator(config, tmp_path, verify_outputs=False)
            return orch

    def test_parallel_preprocessing(self, mock_orchestrator, tmp_path):
        """Verify parallel preprocessing generates correct metadata."""
        # Create test images
        test_images = []
        for i in range(5):
            img_path = tmp_path / f"test_{i}.jpg"
            img_path.touch()
            test_images.append(ImageInput(img_path))

        # Run parallel preprocessing
        results = mock_orchestrator._parallel_preprocess_batch(test_images, tmp_path)

        # Verify all images processed
        assert len(results) == 5

        # Verify each result has required fields
        for result in results:
            assert 'status' in result
            assert 'image_input' in result
            assert 'output_key' in result

    def test_parallel_batch_processing_fallback(self, mock_orchestrator, tmp_path):
        """Verify fallback to sequential for small batches."""
        # Create small batch (< 4 images)
        test_images = [ImageInput(tmp_path / f"test_{i}.jpg") for i in range(2)]
        for img in test_images:
            img.path.touch()

        with patch.object(mock_orchestrator, 'enhance_image', return_value={'status': 'ok'}):
            results = mock_orchestrator.enhance_batch_parallel(test_images, tmp_path)

        # Should use sequential processing
        assert len(results) == 2

    def test_parallel_processing_error_handling(self, mock_orchestrator, tmp_path):
        """Verify parallel processing handles errors gracefully."""
        # Create test images
        test_images = [ImageInput(tmp_path / f"test_{i}.jpg") for i in range(5)]
        for img in test_images:
            img.path.touch()

        # Mock enhance_image to fail for some images
        def mock_enhance(img, input_root):
            if "test_2" in str(img.path):
                raise RuntimeError("Simulated failure")
            return {'status': 'ok', 'image': str(img.path)}

        with patch.object(mock_orchestrator, 'enhance_image', side_effect=mock_enhance):
            results = mock_orchestrator.enhance_batch_parallel(test_images, tmp_path)

        # Verify some succeeded, some failed
        assert len(results) == 5
        error_count = sum(1 for r in results if r.get('status') == 'error')
        assert error_count == 1


class TestPhase2Config:
    """Test Phase 2 configuration flags."""

    def test_parallel_processing_default_enabled(self):
        """Verify parallel processing is enabled by default."""
        config = EnhanceConfig()

        assert config.enable_parallel_processing is True

    def test_parallel_workers_auto_detect(self):
        """Verify worker count auto-detection."""
        config = EnhanceConfig()

        assert config.max_parallel_workers is None  # Auto-detect

    def test_depth_cache_default_disabled(self):
        """Verify depth cache is opt-in (disabled by default)."""
        config = EnhanceConfig()

        assert config.enable_depth_cache is False

    def test_depth_cache_size_limit_configurable(self):
        """Verify cache size limit is configurable."""
        config = EnhanceConfig(
            enable_depth_cache=True,
            depth_cache_max_size_gb=5.0
        )

        assert config.depth_cache_max_size_gb == 5.0

    def test_parallel_processing_can_be_disabled(self):
        """Verify parallel processing can be disabled."""
        config = EnhanceConfig(enable_parallel_processing=False)

        assert config.enable_parallel_processing is False


class TestCacheIntegration:
    """Test depth cache integration with orchestrator."""

    @pytest.fixture
    def orchestrator_with_cache(self, tmp_path):
        """Create orchestrator with depth cache enabled."""
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator
        from transformation_portal.lux_depth_v3.config import EnhanceConfig, ModelVariant

        config = EnhanceConfig(
            model_variant=ModelVariant.METRIC_LARGE,
            enable_depth_cache=True,
            depth_cache_max_size_gb=1.0,
            enable_v2=False,
        )

        with patch('transformation_portal.lux_depth_v3.orchestrator.DA3InferenceEngine'):
            orch = EnhanceOrchestrator(config, tmp_path, verify_outputs=False)
            return orch

    def test_cache_initialization(self, orchestrator_with_cache):
        """Verify cache is initialized when enabled."""
        assert orchestrator_with_cache.depth_cache is not None
        assert orchestrator_with_cache.depth_cache.max_size_gb == 1.0

    def test_cache_disabled_when_flag_off(self, tmp_path):
        """Verify cache is None when disabled."""
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator
        from transformation_portal.lux_depth_v3.config import EnhanceConfig, ModelVariant

        config = EnhanceConfig(
            model_variant=ModelVariant.METRIC_LARGE,
            enable_depth_cache=False,
            enable_v2=False,
        )

        with patch('transformation_portal.lux_depth_v3.orchestrator.DA3InferenceEngine'):
            orch = EnhanceOrchestrator(config, tmp_path)

        assert orch.depth_cache is None


class TestBackwardCompatibility:
    """Test backward compatibility with Phase 1."""

    def test_sequential_processing_still_works(self, tmp_path):
        """Verify sequential processing (Phase 1) still functional."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig

        # Disable parallel processing
        config = EnhanceConfig(
            enable_parallel_processing=False,
            enable_manifest_cache=True,  # Phase 1 feature
            chunked_hashing=True,  # Phase 1 feature
        )

        assert config.enable_parallel_processing is False
        assert config.enable_manifest_cache is True
        assert config.chunked_hashing is True

    def test_phase1_optimizations_preserved(self):
        """Verify Phase 1 optimizations work with Phase 2."""
        config = EnhanceConfig(
            enable_manifest_cache=True,
            chunked_hashing=True,
            enable_parallel_processing=True,
            enable_depth_cache=True,
        )

        # All optimizations can coexist
        assert config.enable_manifest_cache is True
        assert config.chunked_hashing is True
        assert config.enable_parallel_processing is True
        assert config.enable_depth_cache is True


class TestPerformanceMetrics:
    """Test performance tracking for Phase 2."""

    def test_cache_stats_tracking(self, tmp_path):
        """Verify cache stats provide useful metrics."""
        cache = DepthCache(tmp_path)

        # Add some entries
        for i in range(3):
            depth = np.random.rand(100, 100).astype(np.float32)
            cache.store(f"img_{i}", "config", depth)

        stats = cache.stats()

        assert 'entry_count' in stats
        assert 'size_gb' in stats
        assert 'max_size_gb' in stats
        assert 'cache_dir' in stats
        assert stats['entry_count'] == 3

    def test_parallel_worker_count_calculation(self, tmp_path):
        """Verify worker count is calculated correctly."""
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator
        from transformation_portal.lux_depth_v3.config import EnhanceConfig, ModelVariant
        from multiprocessing import cpu_count

        config = EnhanceConfig(
            model_variant=ModelVariant.METRIC_LARGE,
            enable_parallel_processing=True,
            max_parallel_workers=None,  # Auto-detect
            enable_v2=False,
        )

        with patch('transformation_portal.lux_depth_v3.orchestrator.DA3InferenceEngine'):
            orch = EnhanceOrchestrator(config, tmp_path)

        # Should be cpu_count - 1, minimum 1
        expected_workers = max(1, cpu_count() - 1)
        assert orch.max_workers == expected_workers


class TestThreadSafety:
    """Validate thread safety for concurrent operations (Fix #3)."""

    def test_manifest_cache_concurrent_reads(self, tmp_path):
        """Test LRU cache handles concurrent reads safely."""
        import threading
        import os
        from transformation_portal.lux_depth_v3.manifest import CombinedManifest, InputMetadata
        from transformation_portal.lux_depth_v3.orchestrator import _load_manifest_cached

        # Create a manifest
        manifest_path = tmp_path / "test_manifest.json"
        manifest = CombinedManifest(
            input=InputMetadata(
                image_path="test.jpg",
                image_sha256="abc123",
                image_size_bytes=1000,
                image_dimensions=[100, 100]
            )
        )
        manifest.write(manifest_path)

        # Get mtime for cache
        mtime = os.path.getmtime(str(manifest_path))

        # Concurrent read test
        results = []
        exceptions = []

        def read_manifest():
            try:
                m = _load_manifest_cached(str(manifest_path), mtime)
                results.append(m.input.image_sha256 if m and m.input else None)
            except Exception as e:
                exceptions.append(e)

        # Spawn 10 threads reading same manifest simultaneously
        threads = [threading.Thread(target=read_manifest) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify: no exceptions, all return same data, cache working
        assert len(exceptions) == 0, f"Exceptions during concurrent reads: {exceptions}"
        assert len(results) == 10
        assert all(r == "abc123" for r in results), "Inconsistent data returned"

    def test_manifest_cache_concurrent_writes(self, tmp_path):
        """Test LRU cache handles concurrent writes to different manifests safely."""
        import threading
        import os
        from transformation_portal.lux_depth_v3.manifest import CombinedManifest, InputMetadata
        from transformation_portal.lux_depth_v3.orchestrator import _load_manifest_cached

        exceptions = []

        def write_and_read_manifest(idx):
            try:
                manifest_path = tmp_path / f"manifest_{idx}.json"
                manifest = CombinedManifest(
                    input=InputMetadata(
                        image_path=f"test_{idx}.jpg",
                        image_sha256=f"hash_{idx}",
                        image_size_bytes=1000,
                        image_dimensions=[100, 100]
                    )
                )
                manifest.write(manifest_path)

                # Get mtime for cache
                mtime = os.path.getmtime(str(manifest_path))

                # Read back via cache
                loaded = _load_manifest_cached(str(manifest_path), mtime)
                assert loaded.input.image_sha256 == f"hash_{idx}"
            except Exception as e:
                exceptions.append(e)

        # Spawn 10 threads writing different manifests
        threads = [threading.Thread(target=write_and_read_manifest, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify: no exceptions, cache handled concurrent access
        assert len(exceptions) == 0, f"Exceptions during concurrent writes: {exceptions}"

    def test_depth_cache_concurrent_store(self, tmp_path):
        """Test depth cache handles concurrent stores safely."""
        import threading
        from transformation_portal.lux_depth_v3.depth_cache import DepthCache

        cache = DepthCache(tmp_path, max_size_gb=1.0)
        depth = np.random.rand(100, 100).astype(np.float32)
        exceptions = []

        def store_depth(idx):
            try:
                # Different images but same config
                cache.store(f"image_{idx}", "config_abc", depth)
            except Exception as e:
                exceptions.append(e)

        # Spawn 10 threads storing different depths
        threads = [threading.Thread(target=store_depth, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify: no exceptions, all entries stored
        assert len(exceptions) == 0, f"Exceptions during concurrent stores: {exceptions}"
        stats = cache.stats()
        assert stats['entry_count'] == 10

    def test_depth_cache_concurrent_same_key(self, tmp_path):
        """Test depth cache handles concurrent writes to same key (last write wins)."""
        import threading
        from transformation_portal.lux_depth_v3.depth_cache import DepthCache

        cache = DepthCache(tmp_path, max_size_gb=1.0)
        exceptions = []

        def store_depth(value):
            try:
                depth = np.full((100, 100), value, dtype=np.float32)
                cache.store("same_image", "same_config", depth)
            except Exception as e:
                exceptions.append(e)

        # Spawn 10 threads writing same key with different values
        threads = [threading.Thread(target=store_depth, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify: no exceptions, one cache entry exists
        assert len(exceptions) == 0, f"Exceptions during concurrent same-key stores: {exceptions}"
        stats = cache.stats()
        assert stats['entry_count'] == 1

        # Verify final value is from one of the writers (last write wins)
        result = cache.get("same_image", "same_config")
        assert result is not None

    def test_depth_cache_read_while_evict(self, tmp_path):
        """Test depth cache handles read during eviction gracefully."""
        import threading
        from transformation_portal.lux_depth_v3.depth_cache import DepthCache

        # Small cache to trigger eviction
        cache = DepthCache(tmp_path, max_size_gb=0.002)  # 2MB
        exceptions = []
        reads_succeeded = []

        # Fill cache
        for i in range(5):
            depth = np.random.rand(500, 500).astype(np.float32)  # ~1MB each
            cache.store(f"image_{i}", "config", depth)

        def add_entry():
            """Add new entry to trigger eviction."""
            try:
                depth = np.random.rand(500, 500).astype(np.float32)
                cache.store("new_image", "config", depth)
            except Exception as e:
                exceptions.append(e)

        def read_entry():
            """Try to read while eviction might be happening."""
            try:
                result = cache.get("image_0", "config")
                reads_succeeded.append(result is not None)
            except Exception as e:
                exceptions.append(e)

        # Spawn threads: one to trigger eviction, one to read
        t1 = threading.Thread(target=add_entry)
        t2 = threading.Thread(target=read_entry)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Verify: no exceptions (graceful handling)
        assert len(exceptions) == 0, f"Exceptions during concurrent read/evict: {exceptions}"

    def test_parallel_batch_no_race_conditions(self, tmp_path):
        """Test parallel batch processing is thread-safe."""
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator
        from transformation_portal.lux_depth_v3.config import EnhanceConfig, ModelVariant
        from transformation_portal.lux_depth_v3.input_manager import ImageInput
        from PIL import Image

        # Create test images
        test_images = []
        for i in range(10):
            img_path = tmp_path / f"test_{i}.jpg"
            img = Image.new('RGB', (100, 100), color=(i*20, i*20, i*20))
            img.save(img_path, quality=95)
            test_images.append(ImageInput(img_path))

        config = EnhanceConfig(
            model_variant=ModelVariant.METRIC_SMALL,
            enable_parallel_processing=True,
            enable_v2=False,
            max_parallel_workers=4
        )

        with patch('transformation_portal.lux_depth_v3.orchestrator.DA3InferenceEngine') as mock_engine:
            # Mock depth inference with correct shape for postprocessing
            mock_instance = MagicMock()

            def mock_predict(img):
                mock_result = MagicMock()
                # Return 2D depth array (not 1D)
                mock_result.depth = np.random.rand(100, 100).astype(np.float32)
                mock_result.original_image = img
                mock_result.metadata = {}
                return mock_result

            mock_instance.predict = mock_predict
            mock_engine.return_value = mock_instance

            orch = EnhanceOrchestrator(config, tmp_path / "output")

            # Process batch in parallel
            results = orch.enhance_batch_parallel(test_images, input_root=tmp_path)

        # Verify: all images processed, no corruption
        assert len(results) == 10
        assert all(r.get('status') in ['ok', 'skipped'] for r in results)

    def test_atomic_writes_prevent_corruption(self, tmp_path):
        """Test atomic write pattern prevents corruption during concurrent access."""
        import threading
        from transformation_portal.lux_depth_v3.io_atomic import atomic_write_bytes

        target_file = tmp_path / "concurrent_test.txt"
        exceptions = []

        def write_atomically(content):
            try:
                # Simulate slow write
                time.sleep(0.01)
                atomic_write_bytes(target_file, content.encode('utf-8'))
            except Exception as e:
                exceptions.append(e)

        # Spawn 5 threads writing different content
        threads = [
            threading.Thread(target=write_atomically, args=(f"Content_{i}\n" * 100,))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify: no exceptions, final file is valid (one complete write)
        assert len(exceptions) == 0, f"Exceptions during concurrent writes: {exceptions}"
        assert target_file.exists()

        # File should contain complete content from one writer (not corrupted)
        content = target_file.read_text()
        assert content.count('\n') % 100 == 0, "File corruption detected"

    def test_lru_cache_eviction_thread_safe(self, tmp_path):
        """Test LRU cache eviction doesn't break concurrent access."""
        import threading
        from functools import lru_cache

        # Simulate manifest cache behavior
        @lru_cache(maxsize=5)
        def cached_func(key):
            return f"value_{key}"

        exceptions = []
        results = []

        def access_cache(key):
            try:
                result = cached_func(key)
                results.append(result)
            except Exception as e:
                exceptions.append(e)

        # Spawn threads accessing cache with keys that will trigger eviction
        threads = []
        for i in range(20):  # More than maxsize
            t = threading.Thread(target=access_cache, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Verify: no exceptions, all accesses succeeded
        assert len(exceptions) == 0, f"Exceptions during LRU eviction: {exceptions}"
        assert len(results) == 20

    def test_depth_cache_stats_accurate_under_concurrency(self, tmp_path):
        """Test cache stats remain accurate with concurrent access."""
        import threading
        from transformation_portal.lux_depth_v3.depth_cache import DepthCache

        cache = DepthCache(tmp_path, max_size_gb=1.0)
        exceptions = []

        def store_and_retrieve(idx):
            try:
                depth = np.random.rand(100, 100).astype(np.float32)
                cache.store(f"image_{idx}", "config", depth)

                # Retrieve to update access time
                cache.get(f"image_{idx}", "config")
            except Exception as e:
                exceptions.append(e)

        # Spawn 10 threads
        threads = [threading.Thread(target=store_and_retrieve, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify: no exceptions, stats accurate
        assert len(exceptions) == 0, f"Exceptions during concurrent ops: {exceptions}"
        stats = cache.stats()
        assert stats['entry_count'] == 10
        assert stats['size_gb'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
