"""Tests for Phase 2 performance optimizations.

Validates:
1. Content-addressable depth cache (store/retrieve/eviction)
2. Parallel batch processing (correctness and race conditions)
3. Configuration flags for parallelization
4. Cache hit rate tracking
5. Graceful fallback to sequential processing
"""

import copy
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tests.lux_depth_v3.test_depth_cache_identity_v3 import _identity as _cache_identity
from tests.lux_depth_v3.test_depth_cache_identity_v3 import _sha as _cache_sha
from transformation_portal.lux_depth_v3.config import EnhanceConfig
from transformation_portal.lux_depth_v3.depth_cache import DepthCache
from transformation_portal.lux_depth_v3.input_manager import ImageInput

pytestmark = pytest.mark.unit


class TestDepthCache:
    """Test content-addressable depth cache."""

    def test_cache_store_and_retrieve(self, tmp_path):
        """Verify cache can store and retrieve depth maps."""
        cache = DepthCache(tmp_path, max_size_gb=1.0)

        # Create test depth map
        depth = np.random.rand(100, 100).astype(np.float32)
        identity = _cache_identity()

        # Store in cache
        assert cache.store(identity, depth)

        # Retrieve from cache
        retrieved = cache.get(identity)

        assert retrieved is not None
        assert np.allclose(retrieved, depth)

    def test_cache_miss_returns_none(self, tmp_path):
        """Verify cache miss returns None."""
        cache = DepthCache(tmp_path)

        result = cache.get(_cache_identity(input_label="nonexistent"))

        assert result is None

    def test_cache_invalidation_on_config_change(self, tmp_path):
        """Verify different config produces cache miss."""
        cache = DepthCache(tmp_path)

        depth = np.random.rand(100, 100).astype(np.float32)
        identity_v1 = _cache_identity(config=_cache_sha("config-v1"))
        identity_v2 = _cache_identity(config=_cache_sha("config-v2"))

        # Store with config v1
        assert cache.store(identity_v1, depth)

        # Query with config v2 should miss
        result = cache.get(identity_v2)

        assert result is None

    def test_cache_eviction_on_size_limit(self, tmp_path):
        """Verify LRU eviction when cache exceeds size limit."""
        # Set very small size limit
        cache = DepthCache(tmp_path, max_size_gb=0.001)  # 1MB

        # Store multiple large depth maps to trigger eviction
        for i in range(10):
            depth = np.random.rand(500, 500).astype(np.float32)  # ~1MB each
            cache.store(_cache_identity(input_label=f"image-{i}"), depth)

        # Cache should have evicted some entries
        stats = cache.stats()
        assert stats["entry_count"] < 10

    def test_cache_stats(self, tmp_path):
        """Verify cache statistics are accurate."""
        cache = DepthCache(tmp_path)

        # Store some depth maps
        for i in range(5):
            depth = np.random.rand(100, 100).astype(np.float32)
            assert cache.store(_cache_identity(input_label=f"image-{i}"), depth)

        stats = cache.stats()

        assert stats["entry_count"] == 5
        assert stats["size_gb"] > 0
        assert stats["max_size_gb"] == 10.0

    def test_cache_clear(self, tmp_path):
        """Verify cache can be cleared."""
        cache = DepthCache(tmp_path)

        # Store depth maps
        for i in range(3):
            depth = np.random.rand(100, 100).astype(np.float32)
            assert cache.store(_cache_identity(input_label=f"image-{i}"), depth)

        # Clear cache
        cache.clear()

        # Verify all entries removed
        stats = cache.stats()
        assert stats["entry_count"] == 0

    def test_cache_atomic_writes(self, tmp_path):
        """Verify cache writes are atomic (no partial files)."""
        cache = DepthCache(tmp_path)

        depth = np.random.rand(1000, 1000).astype(np.float32)

        # Store depth
        assert cache.store(_cache_identity(), depth)

        # Verify no .tmp files left behind
        tmp_files = list(cache.cache_dir.rglob("*.tmp*"))
        assert len(tmp_files) == 0

    def test_cache_corrupted_file_handling(self, tmp_path):
        """Verify cache handles corrupted files gracefully."""
        cache = DepthCache(tmp_path)

        # Create corrupted cache file
        corrupted_path = cache.cache_dir / "corrupted_hash.npy"
        corrupted_path.write_text("not a valid numpy file")

        # Should return None for corrupted file
        result = cache.get(_cache_identity(input_label="corrupted"))

        assert result is None


class TestParallelProcessing:
    """Test parallel batch processing."""

    @pytest.fixture
    def mock_orchestrator(self, tmp_path):
        """Create mock orchestrator with parallel processing enabled."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig, ModelVariant
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        config = EnhanceConfig(
            model_variant=ModelVariant.METRIC_LARGE,
            enable_parallel_processing=True,
            max_parallel_workers=2,
            enable_v2=False,  # Disable V2 for simpler testing
        )

        with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry"):
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
            assert "status" in result
            assert "image_input" in result
            assert "output_key" in result

    def test_parallel_preprocessing_preserves_input_order(self, mock_orchestrator, tmp_path):
        """Parallel preprocessing should preserve caller input order."""
        test_images = []
        for i in range(5):
            img_path = tmp_path / f"ordered_{i}.jpg"
            img_path.touch()
            test_images.append(ImageInput(img_path))

        def _reverse_completion_order(futures):  # noqa: ANN001, ANN202
            return reversed(list(futures))

        with patch(
            "transformation_portal.lux_depth_v3.orchestrator.as_completed",
            side_effect=_reverse_completion_order,
        ):
            results = mock_orchestrator._parallel_preprocess_batch(test_images, tmp_path)

        assert [result["image_input"].path for result in results] == [img.path for img in test_images]

    def test_parallel_batch_processing_fallback(self, mock_orchestrator, tmp_path):
        """Verify fallback to sequential for small batches."""
        # Create small batch (< 4 images)
        test_images = [ImageInput(tmp_path / f"test_{i}.jpg") for i in range(2)]
        for img in test_images:
            img.path.touch()

        with patch.object(mock_orchestrator, "enhance_image", return_value={"status": "ok"}):
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
        def mock_enhance(img, input_root, _precomputed_paths=None):
            if "test_2" in str(img.path):
                raise RuntimeError("Simulated failure")
            return {"status": "ok", "image": str(img.path)}

        with patch.object(mock_orchestrator, "enhance_image", side_effect=mock_enhance):
            results = mock_orchestrator.enhance_batch_parallel(test_images, tmp_path)

        # Verify some succeeded, some failed
        assert len(results) == 5
        error_count = sum(1 for r in results if r.get("status") == "error")
        assert error_count == 1

    def test_parallel_batch_processing_preserves_result_order(self, mock_orchestrator, tmp_path):
        """Parallel batch results should stay aligned with input discovery order."""
        test_images = [ImageInput(tmp_path / f"batch_{i}.jpg") for i in range(5)]
        for img in test_images:
            img.path.touch()

        def _reverse_completion_order(futures):  # noqa: ANN001, ANN202
            return reversed(list(futures))

        def _mock_enhance(img, input_root, _precomputed_paths=None):  # noqa: ANN001, ARG001
            return {"status": "ok", "image": str(img.path)}

        with (
            patch(
                "transformation_portal.lux_depth_v3.orchestrator.as_completed",
                side_effect=_reverse_completion_order,
            ),
            patch.object(mock_orchestrator, "enhance_image", side_effect=_mock_enhance),
        ):
            results = mock_orchestrator.enhance_batch_parallel(test_images, tmp_path)

        assert [result["image"] for result in results] == [str(img.path) for img in test_images]


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
        config = EnhanceConfig(enable_depth_cache=True, depth_cache_max_size_gb=5.0)

        assert config.depth_cache_max_size_gb == 5.0

    def test_parallel_processing_can_be_disabled(self):
        """Verify parallel processing can be disabled."""
        config = EnhanceConfig(enable_parallel_processing=False)

        assert config.enable_parallel_processing is False


class TestCacheIntegration:
    """Test depth cache integration with orchestrator."""

    def test_cache_initialization_requires_prepared_execution(self, tmp_path):
        """Direct callers cannot enable a cache without plan authority."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig, ModelVariant
        from transformation_portal.lux_depth_v3.execution_plan_adapter import LuxExecutionPlanAuthorityError
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        config = EnhanceConfig(
            model_variant=ModelVariant.METRIC_LARGE,
            enable_depth_cache=True,
            depth_cache_max_size_gb=1.0,
            enable_v2=False,
        )
        original_config = copy.deepcopy(config)

        with (
            patch("transformation_portal.lux_depth_v3.orchestrator.apply_effective_da3_runtime_config") as da3_resolver,
            patch("transformation_portal.lux_depth_v3.orchestrator.apply_effective_raw_runtime_config") as raw_resolver,
            pytest.raises(LuxExecutionPlanAuthorityError, match="from_prepared"),
        ):
            EnhanceOrchestrator(config, tmp_path / "output", verify_outputs=False)

        da3_resolver.assert_not_called()
        raw_resolver.assert_not_called()
        assert config == original_config
        assert not (tmp_path / "output").exists()

    def test_prepared_cache_initializes_with_planned_size_limit(self, tmp_path):
        """Prepared execution carries the exact configured cache quota."""
        from transformation_portal.lux_depth_v3.execution_lifecycle import prepare_lux_execution
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        input_root = tmp_path / "inputs"
        input_root.mkdir()
        image = input_root / "scene.jpg"
        image.write_bytes(b"not-decoded-during-plan-preparation")
        prepared = prepare_lux_execution(
            EnhanceConfig(
                depth_backend="synthetic",
                allow_synthetic_fallback=True,
                enable_depth_cache=True,
                depth_cache_max_size_gb=1.25,
                enable_v2=False,
            ),
            input_root,
            [image],
        )

        orchestrator = EnhanceOrchestrator.from_prepared(prepared, tmp_path / "output", verify_outputs=False)

        assert orchestrator.depth_cache is not None
        assert orchestrator.depth_cache.max_size_gb == 1.25

    def test_cache_disabled_when_flag_off(self, tmp_path):
        """Verify cache is None when disabled."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig, ModelVariant
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        config = EnhanceConfig(
            model_variant=ModelVariant.METRIC_LARGE,
            enable_depth_cache=False,
            enable_v2=False,
        )

        with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry"):
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
            assert cache.store(_cache_identity(input_label=f"img-{i}"), depth)

        stats = cache.stats()

        assert "entry_count" in stats
        assert "size_gb" in stats
        assert "max_size_gb" in stats
        assert "cache_dir" in stats
        assert stats["entry_count"] == 3

    def test_parallel_worker_count_calculation(self, tmp_path):
        """Verify worker count is calculated correctly."""
        from multiprocessing import cpu_count

        from transformation_portal.lux_depth_v3.config import EnhanceConfig, ModelVariant
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        config = EnhanceConfig(
            model_variant=ModelVariant.METRIC_LARGE,
            enable_parallel_processing=True,
            max_parallel_workers=None,  # Auto-detect
            enable_v2=False,
        )

        with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry"):
            orch = EnhanceOrchestrator(config, tmp_path)

        # Should be cpu_count - 1, minimum 1
        expected_workers = max(1, cpu_count() - 1)
        assert orch.max_workers == expected_workers


class TestThreadSafety:
    """Validate thread safety for concurrent operations (Fix #3)."""

    def test_manifest_cache_concurrent_reads(self, tmp_path):
        """Test LRU cache handles concurrent reads safely."""
        import os
        import threading

        from transformation_portal.lux_depth_v3.manifest import CombinedManifest, InputMetadata
        from transformation_portal.lux_depth_v3.orchestrator import _load_manifest_cached

        # Create a manifest
        manifest_path = tmp_path / "test_manifest.json"
        manifest = CombinedManifest(
            input=InputMetadata(
                image_path="test.jpg", image_sha256="abc123", image_size_bytes=1000, image_dimensions=[100, 100]
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
        import os
        import threading

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
                        image_dimensions=[100, 100],
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
                cache.store(_cache_identity(input_label=f"image-{idx}"), depth)
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
        assert stats["entry_count"] == 10

    def test_depth_cache_concurrent_same_key(self, tmp_path):
        """Test depth cache rejects divergent writes to one identity safely."""
        import threading

        from transformation_portal.lux_depth_v3.depth_cache import DepthCache

        cache = DepthCache(tmp_path, max_size_gb=1.0)
        exceptions = []

        def store_depth(value):
            try:
                depth = np.full((100, 100), value, dtype=np.float32)
                cache.store(_cache_identity(input_label="same-image"), depth)
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
        assert stats["entry_count"] == 1

        # The first complete publication wins; every hit remains verified.
        result = cache.get(_cache_identity(input_label="same-image"))
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
            cache.store(_cache_identity(input_label=f"image-{i}"), depth)

        def add_entry():
            """Add new entry to trigger eviction."""
            try:
                depth = np.random.rand(500, 500).astype(np.float32)
                cache.store(_cache_identity(input_label="new-image"), depth)
            except Exception as e:
                exceptions.append(e)

        def read_entry():
            """Try to read while eviction might be happening."""
            try:
                result = cache.get(_cache_identity(input_label="image-0"))
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
        from PIL import Image

        from transformation_portal.lux_depth_v3.config import EnhanceConfig, ModelVariant
        from transformation_portal.lux_depth_v3.input_manager import ImageInput
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        # Create test images
        test_images = []
        for i in range(10):
            img_path = tmp_path / f"test_{i}.jpg"
            img = Image.new("RGB", (100, 100), color=(i * 20, i * 20, i * 20))
            img.save(img_path, quality=95)
            test_images.append(ImageInput(img_path))

        config = EnhanceConfig(
            model_variant=ModelVariant.METRIC_SMALL, enable_parallel_processing=True, enable_v2=False, max_parallel_workers=4
        )

        with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry"):
            orch = EnhanceOrchestrator(config, tmp_path / "output")

            # Mock the depth backend
            mock_backend = MagicMock(spec=["name", "compute"])
            mock_backend.name = "mock"

            def mock_compute(img):
                # Convert PIL Image to numpy if needed
                import numpy as np
                from PIL import Image

                if isinstance(img, Image.Image):
                    img_array = np.array(img)
                else:
                    img_array = img

                mock_result = MagicMock(spec=["depth", "depth_map", "original_image", "metadata"])
                # Return 2D depth array (not 1D)
                mock_result.depth = np.random.rand(100, 100).astype(np.float32)
                mock_result.depth_map = mock_result.depth
                mock_result.original_image = img_array
                mock_result.metadata = {}
                return mock_result

            mock_backend.compute = mock_compute
            orch.depth_backend = mock_backend

            # Process batch in parallel
            results = orch.enhance_batch_parallel(test_images, input_root=tmp_path)

        # Verify: all images processed, no corruption
        assert len(results) == 10
        assert all(r.get("status") in ["ok", "skipped"] for r in results)

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
                atomic_write_bytes(target_file, content.encode("utf-8"))
            except Exception as e:
                exceptions.append(e)

        # Spawn 5 threads writing different content
        threads = [threading.Thread(target=write_atomically, args=(f"Content_{i}\n" * 100,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify: no exceptions, final file is valid (one complete write)
        assert len(exceptions) == 0, f"Exceptions during concurrent writes: {exceptions}"
        assert target_file.exists()

        # File should contain complete content from one writer (not corrupted)
        content = target_file.read_text()
        assert content.count("\n") % 100 == 0, "File corruption detected"

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
                identity = _cache_identity(input_label=f"image-{idx}")
                cache.store(identity, depth)

                # Retrieve to update access time
                cache.get(identity)
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
        assert stats["entry_count"] == 10
        assert stats["size_gb"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
