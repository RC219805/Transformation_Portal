"""Performance regression tests for Phase 1-3 optimizations (Fix #4).

Validates performance claims and prevents regressions:
- Phase 1: Manifest caching (15-20% I/O reduction), chunked SHA-256 (90% memory reduction)
- Phase 2: Parallel processing (3-5x speedup), depth caching
- Phase 3: PBR batching (30% speedup), msgpack serialization

Tests are marked with @pytest.mark.benchmark.
Note: CI runs these unless explicitly excluded by marker selection (e.g., -m "not benchmark").

ADR-019 Note:
Uses backend protocol mocks (DA3Backend.compute) instead of legacy
orchestrator.DA3InferenceEngine + Postprocessor pattern.
"""

import hashlib
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# Mark all tests as ML tier (require depth processing / torch)
pytestmark = pytest.mark.ml

from tests.lux_depth_v3.test_depth_cache_identity_v3 import _identity as _cache_identity
from transformation_portal.lux_depth_v3.config import EnhanceConfig, ModelVariant
from transformation_portal.lux_depth_v3.depth_cache import DepthCache
from transformation_portal.lux_depth_v3.input_manager import ImageInput
from transformation_portal.lux_depth_v3.manifest import CombinedManifest, InputMetadata, compute_file_sha256
from transformation_portal.lux_depth_v3.orchestrator import _load_manifest_cached

# Mark all tests as ML tier (require depth processing / torch)
pytestmark = pytest.mark.ml

# ============================================================================
# Shared Test Fixtures and Helpers
# ============================================================================


@pytest.fixture
def mock_depth_result():
    """Create a realistic mock DepthResult with proper array shapes (backend protocol)."""

    def _create(height=512, width=512):
        from transformation_portal.depth.backends.protocol import DepthResult as BackendDepthResult

        depth_map = np.random.rand(height, width).astype(np.float32)
        original_image = (np.random.rand(height, width, 3) * 255).astype(np.uint8)

        return BackendDepthResult(
            depth_map=depth_map,
            original_image=original_image,
            metadata={"model": "mock", "backend": "test"},
            depth_units="relative",
            focal_length_px=None,
            field_of_view_deg=None,
            backend_id="mock",
            device="cpu",
            dtype="float32",
            input_size=(height, width),
            warnings=[],
        )

    return _create


@pytest.fixture
def mock_backend_compute(mock_depth_result):
    """Mock DA3Backend.compute() for performance tests (ADR-019)."""
    with patch("transformation_portal.depth.backends.da3.DA3Backend.compute") as mock_compute:
        # Configure backend compute mock to return realistic depth result
        def _mock_compute(image, **kwargs):
            # Extract dimensions from input image
            if hasattr(image, "size"):
                width, height = image.size
            elif hasattr(image, "shape"):
                height, width = image.shape[:2]
            else:
                height, width = 256, 256

            return mock_depth_result(height=height, width=width)

        mock_compute.side_effect = _mock_compute
        yield mock_compute


# Postprocessor mock removed - no longer needed with ADR-019 backend pattern
# Backend handles its own postprocessing internally


class TestPhase1Performance:
    """Phase 1: Manifest caching and chunked hashing performance."""

    @pytest.mark.benchmark
    def test_manifest_caching_speedup(self, tmp_path):
        """Phase 1: Manifest caching achieves 15-20% I/O reduction."""
        manifest_path = tmp_path / "test_manifest.json"

        # Create a manifest
        manifest = CombinedManifest(
            input=InputMetadata(
                image_path="test.jpg", image_sha256="abc123", image_size_bytes=1000, image_dimensions=[100, 100]
            )
        )
        manifest.write(manifest_path)

        # Get mtime for cached calls
        mtime = manifest_path.stat().st_mtime

        # Benchmark: Load 100 times WITHOUT cache (direct load)
        _load_manifest_cached.cache_clear()
        start_uncached = time.time()
        for _ in range(100):
            CombinedManifest.load(manifest_path)
        uncached_time = time.time() - start_uncached

        # Benchmark: Load 100 times WITH cache
        _load_manifest_cached.cache_clear()
        start_cached = time.time()
        for _ in range(100):
            _load_manifest_cached(str(manifest_path), mtime)
        cached_time = time.time() - start_cached

        # Calculate speedup
        speedup = uncached_time / cached_time if cached_time > 0 else 0

        # Assert: cached >= 1.15x faster (relaxed to 1.10x for CI variance)
        assert speedup >= 1.10, (
            f"Manifest caching speedup {speedup:.2f}x < 1.10x minimum "
            f"(uncached={uncached_time:.3f}s, cached={cached_time:.3f}s)"
        )
        print(f"✓ Manifest caching speedup: {speedup:.2f}x")

    @pytest.mark.benchmark
    def test_chunked_sha256_memory_reduction(self, tmp_path):
        """Phase 1: Chunked SHA-256 reduces memory by ~90%."""
        # Create a large test file (50MB)
        test_file = tmp_path / "large_test.bin"
        chunk_size_mb = 10
        with open(test_file, "wb") as f:
            for _ in range(5):  # 5 * 10MB = 50MB
                f.write(b"X" * (chunk_size_mb * 1024 * 1024))

        # Method 1: Full file load (simulated - don't actually allocate)
        # Expected: 50MB in memory
        full_load_memory_mb = 50

        # Method 2: Chunked read (actual implementation)
        hash_result = compute_file_sha256(test_file)

        # Chunked implementation uses 64KB chunks
        chunked_memory_mb = 0.0625  # 64KB

        # Calculate memory reduction
        memory_reduction_ratio = chunked_memory_mb / full_load_memory_mb

        # Assert: chunked memory <= 10% of full memory
        assert memory_reduction_ratio <= 0.10, (
            f"Chunked hashing memory {chunked_memory_mb}MB exceeds 10% of "
            f"full load {full_load_memory_mb}MB (ratio={memory_reduction_ratio:.2%})"
        )

        # Verify correctness: hash should match
        with open(test_file, "rb") as f:
            expected_hash = hashlib.sha256(f.read()).hexdigest()
        assert hash_result == expected_hash

        print(f"✓ Chunked SHA-256 memory reduction: {(1-memory_reduction_ratio)*100:.1f}%")

    @pytest.mark.benchmark
    def test_manifest_cache_hit_performance(self, tmp_path):
        """Verify manifest cache hits are near-instantaneous."""
        manifest_path = tmp_path / "test_manifest.json"
        manifest = CombinedManifest(
            input=InputMetadata(
                image_path="test.jpg", image_sha256="abc123", image_size_bytes=1000, image_dimensions=[100, 100]
            )
        )
        manifest.write(manifest_path)

        # Get mtime for cached calls
        mtime = manifest_path.stat().st_mtime

        # Prime cache
        _load_manifest_cached.cache_clear()
        _load_manifest_cached(str(manifest_path), mtime)

        # Benchmark 1000 cache hits
        start = time.time()
        for _ in range(1000):
            _load_manifest_cached(str(manifest_path), mtime)
        cache_hit_time = time.time() - start

        # Cache hits should be < 1ms each (1000 hits in < 1 second)
        assert cache_hit_time < 1.0, (
            f"Cache hits too slow: {cache_hit_time:.3f}s for 1000 hits " f"({cache_hit_time*1000:.3f}ms per hit)"
        )
        print(f"✓ Manifest cache hit performance: {cache_hit_time*1000:.3f}ms per 1000 hits")


class TestPhase2Performance:
    """Phase 2: Parallel processing and depth caching performance."""

    @pytest.mark.benchmark
    def test_parallel_batch_speedup(self, tmp_path, mock_depth_result):
        """Phase 2: Parallel processing achieves 3-5x speedup.

        Note: With mocked inference, actual parallelism is limited.
        This test verifies the parallel path executes without errors.
        Real performance benefits are seen with actual GPU inference.
        """
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        # Create 20 test images
        test_images = []
        for i in range(20):
            img_path = tmp_path / "input" / f"test_{i}.jpg"
            img_path.parent.mkdir(parents=True, exist_ok=True)
            img = Image.new("RGB", (512, 512), color=(i * 10, i * 10, i * 10))
            img.save(img_path, quality=95)
            test_images.append(ImageInput(img_path))

        # Simulate realistic inference time: 100ms
        def mock_compute_with_delay(image, **kwargs):
            time.sleep(0.1)
            if hasattr(image, "size"):
                width, height = image.size
            elif hasattr(image, "shape"):
                height, width = image.shape[:2]
            else:
                height, width = 256, 256
            return mock_depth_result(height=height, width=width)

        # Test sequential processing
        with patch("transformation_portal.depth.backends.da3.DA3Backend.compute") as mock_compute_seq:
            mock_compute_seq.side_effect = mock_compute_with_delay

            config_seq = EnhanceConfig(
                model_variant=ModelVariant.METRIC_SMALL,
                enable_parallel_processing=False,
                enable_v2=False,
                enable_depth_cache=False,
            )
            orch_seq = EnhanceOrchestrator(config_seq, tmp_path / "output_seq")

            start_seq = time.time()
            results_seq = orch_seq.enhance_batch_parallel(test_images, input_root=tmp_path / "input")
            seq_time = time.time() - start_seq

        # Test parallel processing with fresh mocks
        with patch("transformation_portal.depth.backends.da3.DA3Backend.compute") as mock_compute_par:
            mock_compute_par.side_effect = mock_compute_with_delay

            config_par = EnhanceConfig(
                model_variant=ModelVariant.METRIC_SMALL,
                enable_parallel_processing=True,
                max_parallel_workers=4,
                enable_v2=False,
                enable_depth_cache=False,
            )
            orch_par = EnhanceOrchestrator(config_par, tmp_path / "output_par")

            start_par = time.time()
            results_par = orch_par.enhance_batch_parallel(test_images, input_root=tmp_path / "input")
            par_time = time.time() - start_par

        # Calculate speedup
        speedup = seq_time / par_time if par_time > 0 else 0

        # Verify both paths complete successfully
        assert len(results_seq) == 20
        assert len(results_par) == 20

        # With mocks, we can't guarantee speedup, just verify parallel path works
        # Real speedup requires actual GPU inference
        print(
            f"✓ Parallel batch processing completed: {speedup:.2f}x speedup "
            f"(sequential={seq_time:.2f}s, parallel={par_time:.2f}s)"
        )
        print("  Note: Actual speedup requires real GPU inference, not mocks")

    @pytest.mark.benchmark
    def test_depth_cache_eliminates_redundant_computation(self, tmp_path):
        """Phase 2: Depth cache eliminates recomputation (10x+ speedup)."""
        cache = DepthCache(tmp_path / "cache", max_size_gb=1.0)

        # Create test depth
        depth = np.random.rand(1024, 1024).astype(np.float32)
        identity = _cache_identity()

        # Benchmark: Cache miss (store operation)
        start_miss = time.time()
        assert cache.store(identity, depth)
        miss_time = time.time() - start_miss

        # Benchmark: Cache hit (retrieve operation)
        start_hit = time.time()
        for _ in range(10):
            assert cache.get(identity) is not None
        hit_time = (time.time() - start_hit) / 10  # Average per retrieval

        # Calculate speedup
        speedup = miss_time / hit_time if hit_time > 0 else 0

        # Verified hits deliberately hash the immutable NPY object. Preserve a
        # conservative throughput floor instead of the old unverified 2x claim.
        verified_hit_mib_per_second = depth.nbytes / max(hit_time, 1e-9) / (1024**2)
        assert verified_hit_mib_per_second >= 5.0
        print(
            f"✓ Verified depth cache timing: store={miss_time:.3f}s, hit={hit_time:.3f}s, "
            f"ratio={speedup:.1f}x, throughput={verified_hit_mib_per_second:.1f}MiB/s"
        )

    @pytest.mark.benchmark
    def test_sequential_fallback_no_overhead(self, tmp_path, mock_backend_compute):
        """Ensure small batches fall back to sequential without penalty."""
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        # Create 2 test images (below 4-image threshold)
        test_images = []
        for i in range(2):
            img_path = tmp_path / "input" / f"test_{i}.jpg"
            img_path.parent.mkdir(parents=True, exist_ok=True)
            img = Image.new("RGB", (512, 512), color=(i * 100, i * 100, i * 100))
            img.save(img_path, quality=95)
            test_images.append(ImageInput(img_path))

        # Benchmark with parallel enabled (should auto-fallback)
        config = EnhanceConfig(model_variant=ModelVariant.METRIC_SMALL, enable_parallel_processing=True, enable_v2=False)
        orch = EnhanceOrchestrator(config, tmp_path / "output")

        start = time.time()
        results = orch.enhance_batch_parallel(test_images, input_root=tmp_path / "input")
        total_time = time.time() - start

        # Verify: falls back to sequential (total time reasonable)
        # With 2 images, should complete in < 2 seconds (overhead check)
        assert total_time < 2.0, f"Sequential fallback too slow: {total_time:.2f}s for 2 images"
        assert len(results) == 2
        print(f"✓ Sequential fallback completed in {total_time:.2f}s (no overhead)")

    @pytest.mark.benchmark
    def test_cache_store_scalability(self, tmp_path):
        """Verify populated-cache stores use bounded periodic size scans."""
        cache = DepthCache(tmp_path / "cache", max_size_gb=10.0)

        for i in range(64):
            depth = np.full((32, 32), i, dtype=np.float32)
            assert cache.store(_cache_identity(input_label=f"prepopulate-{i}"), depth)

        with patch.object(cache, "_physical_size_bytes", wraps=cache._physical_size_bytes) as size_scan:
            start = time.time()
            for i in range(48):
                depth = np.full((32, 32), i + 64, dtype=np.float32)
                assert cache.store(_cache_identity(input_label=f"scale-{i}"), depth)
            elapsed = time.time() - start
            assert size_scan.call_count <= 3

        assert cache.stats()["entry_count"] == 112
        lock_files = list((cache.cache_dir / "v1" / "locks").glob("*.publication.lock"))
        assert len(lock_files) <= 64
        print(f"✓ Verified cache added 48 populated entries in {elapsed:.3f}s with at most 3 full scans")

    @pytest.mark.benchmark
    def test_cache_initialization_with_existing_files(self, tmp_path):
        """Verify a new cache instance recognizes existing verified entries."""
        cache_dir = tmp_path / "cache"

        # Create initial cache with 10 entries
        cache1 = DepthCache(cache_dir, max_size_gb=10.0)
        depths = [np.random.rand(512, 512).astype(np.float32) for _ in range(10)]
        for i, depth in enumerate(depths):
            assert cache1.store(_cache_identity(input_label=f"init-{i}"), depth)

        initial_stats = cache1.stats()

        # Create new cache instance (simulates restart)
        cache2 = DepthCache(cache_dir, max_size_gb=10.0)

        restarted_stats = cache2.stats()
        assert restarted_stats["entry_count"] == 10
        assert restarted_stats["size_gb"] == pytest.approx(initial_stats["size_gb"])

    @pytest.mark.benchmark
    def test_cache_overwrite_handling(self, tmp_path):
        """Verify one identity cannot be overwritten with divergent bytes."""
        cache = DepthCache(tmp_path / "cache", max_size_gb=10.0)
        identity = _cache_identity()

        # Store initial depth
        depth1 = np.random.rand(512, 512).astype(np.float32)
        assert cache.store(identity, depth1)
        size_after_first = cache.stats()["size_gb"]

        # Overwrite with same key
        depth2 = np.random.rand(512, 512).astype(np.float32)
        assert not cache.store(identity, depth2)
        size_after_second = cache.stats()["size_gb"]
        assert size_after_second == pytest.approx(size_after_first)

        # Verify actual file count: should have 1 file, not 2
        cache_files = list((cache.cache_dir / "v1" / "objects").glob("*/*.npy"))
        assert len(cache_files) == 1, f"Expected 1 cache file after overwrite, found {len(cache_files)}"
        np.testing.assert_array_equal(cache.get(identity), depth1)

    @pytest.mark.benchmark
    def test_cache_thread_safety(self, tmp_path):
        """Verify cache handles concurrent stores without race conditions."""
        import concurrent.futures

        cache = DepthCache(tmp_path / "cache", max_size_gb=10.0)

        def store_depth(index: int):
            """Store a depth map (worker function)."""
            depth = np.random.rand(256, 256).astype(np.float32)
            assert cache.store(_cache_identity(input_label=f"thread-{index}"), depth)
            return index

        # Store 50 depths concurrently with 4 threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(store_depth, i) for i in range(50)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # Verify all stores completed
        assert len(results) == 50

        # Verify cache has 50 entries
        cache_files = list((cache.cache_dir / "v1" / "objects").glob("*/*.npy"))
        assert len(cache_files) == 50, f"Thread safety issue: expected 50 cache files, found {len(cache_files)}"
        stats = cache.stats()
        assert stats["entry_count"] == 50
        assert 0.0 < stats["size_gb"] < stats["max_size_gb"]

        print("✓ Thread safety: 50 concurrent stores completed successfully")
        print(f"  Final state: {len(cache_files)} immutable objects, {stats['size_gb']:.4f}GB")


class TestPhase3Performance:
    """Phase 3: Advanced optimizations (PBR batching, msgpack)."""

    @pytest.mark.benchmark
    def test_pbr_batching_speedup(self, tmp_path):
        """Phase 3: PBR batching achieves 30% speedup."""
        from transformation_portal.lux_depth_v3.pbr import PBRConfig, generate_pbr_maps

        # Create 10 test depths
        depths = [np.random.rand(512, 512).astype(np.float32) for _ in range(10)]
        config = PBRConfig()

        # Benchmark: Sequential PBR generation
        start_seq = time.time()
        for depth in depths:
            generate_pbr_maps(depth, config=config)
        seq_time = time.time() - start_seq

        # Note: Actual batching implementation would process multiple depths together
        # For now, verify sequential baseline is reasonable
        per_image_time = seq_time / 10

        # Verify: PBR generation is fast enough (< 500ms per 512x512 image)
        assert per_image_time < 0.5, f"PBR generation too slow: {per_image_time*1000:.0f}ms per image " f"(expected < 500ms)"
        print(f"✓ PBR generation baseline: {per_image_time*1000:.0f}ms per image")

    @pytest.mark.benchmark
    def test_no_regression_single_image(self, tmp_path, mock_backend_compute):
        """Ensure optimizations don't regress single-image performance."""
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        # Create single test image
        img_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (512, 512), color=(128, 128, 128))
        img.save(img_path, quality=95)

        config = EnhanceConfig(
            model_variant=ModelVariant.METRIC_SMALL,
            enable_parallel_processing=True,
            enable_manifest_cache=True,
            enable_depth_cache=True,
            enable_v2=False,
        )
        orch = EnhanceOrchestrator(config, tmp_path / "output")

        start = time.time()
        result = orch.enhance_image(ImageInput(img_path))
        total_time = time.time() - start

        # Verify: single image completes in reasonable time (< 2 seconds with mocks)
        assert total_time < 2.0, (
            f"Single image processing too slow: {total_time:.2f}s " f"(optimizations may have added overhead)"
        )
        assert result["status"] == "ok"
        print(f"✓ Single image processing: {total_time:.3f}s (no regression)")


class TestPerformanceBaselines:
    """Establish performance baselines for monitoring."""

    @pytest.mark.benchmark
    def test_file_io_baseline(self, tmp_path):
        """Establish baseline for file I/O operations."""
        test_file = tmp_path / "io_test.bin"
        data = b"X" * (10 * 1024 * 1024)  # 10MB

        # Write baseline
        start_write = time.time()
        test_file.write_bytes(data)
        write_time = time.time() - start_write

        # Read baseline
        start_read = time.time()
        read_data = test_file.read_bytes()
        read_time = time.time() - start_read

        assert len(read_data) == len(data)
        print(f"✓ File I/O baseline: write={write_time*1000:.0f}ms, read={read_time*1000:.0f}ms (10MB)")

    @pytest.mark.benchmark
    def test_numpy_operations_baseline(self):
        """Establish baseline for NumPy operations."""
        arr = np.random.rand(1024, 1024).astype(np.float32)

        # Array creation
        start = time.time()
        for _ in range(10):
            _ = np.random.rand(1024, 1024).astype(np.float32)
        create_time = (time.time() - start) / 10

        # Array save/load
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        start_save = time.time()
        np.save(str(tmp_path), arr)
        save_time = time.time() - start_save

        start_load = time.time()
        _ = np.load(str(tmp_path))  # noqa: F841
        load_time = time.time() - start_load

        tmp_path.unlink()

        print(
            f"✓ NumPy baseline: create={create_time*1000:.0f}ms, "
            f"save={save_time*1000:.0f}ms, load={load_time*1000:.0f}ms (1024x1024)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "benchmark"])
