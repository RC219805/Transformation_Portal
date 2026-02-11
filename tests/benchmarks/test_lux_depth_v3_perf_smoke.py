"""lux_depth_v3 Performance Smoke Tests (PR L0.0).

Fast, deterministic baseline benchmarks for the lux_depth_v3 pipeline.
No model downloads, no network calls - pure performance measurement infrastructure.

Measures:
- p50/p95 runtime per image and per megapixel
- Peak memory usage (RSS baseline + incremental)
- Output invariants (dtype, range, shape, no NaNs/inf)

Success Criteria:
- Runs in <10s in CI
- Produces machine-readable output for regression tracking
- All tests fully offline with mocked ML dependencies
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

# Import lux_depth_v3 components
from transformation_portal.lux_depth_v3.config import EnhanceConfig, ModelVariant
from transformation_portal.lux_depth_v3.input_manager import ImageInput
from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

# Mark all tests as benchmark tier
pytestmark = [pytest.mark.benchmark]


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def synthetic_images(tmp_path):
    """Create small deterministic synthetic test images (no downloads)."""
    fixtures = []
    sizes = [
        (256, 256),  # Small
        (512, 512),  # Medium
        (1024, 768),  # HD-ish
    ]

    for i, (width, height) in enumerate(sizes):
        # Create deterministic pattern (not random - same every run)
        img = Image.new("RGB", (width, height))
        pixels = []
        for y in range(height):
            for x in range(width):
                # Simple gradient pattern
                r = (x * 255) // width
                g = (y * 255) // height
                b = ((x + y) * 255) // (width + height)
                pixels.append((r, g, b))
        img.putdata(pixels)

        img_path = tmp_path / f"test_{width}x{height}.jpg"
        img.save(img_path, quality=95)
        fixtures.append({"path": img_path, "width": width, "height": height, "megapixels": (width * height) / 1e6})

    return fixtures


@pytest.fixture
def mock_depth_backend():
    """Mock depth backend to avoid model downloads and inference."""
    with patch("transformation_portal.depth.backends.da3.DA3Backend.compute") as mock_compute:

        def _mock_compute(image, **kwargs):
            """Generate realistic mock depth result with proper dimensions."""
            from transformation_portal.depth.backends.protocol import DepthResult

            # Get image dimensions
            if hasattr(image, "size"):
                width, height = image.size
            elif hasattr(image, "shape"):
                height, width = image.shape[:2]
            else:
                height, width = 512, 512

            # Generate deterministic depth pattern (not random)
            depth_map = np.zeros((height, width), dtype=np.float32)
            for y in range(height):
                for x in range(width):
                    # Simple distance-from-center depth
                    dx = x - width / 2
                    dy = y - height / 2
                    dist = np.sqrt(dx * dx + dy * dy)
                    depth_map[y, x] = min(1.0, dist / (max(width, height) / 2))

            original_image = np.array(image) if hasattr(image, "mode") else image

            return DepthResult(
                depth_map=depth_map,
                original_image=original_image,
                metadata={"model": "mock_da3", "backend": "test"},
                depth_units="relative",
                focal_length_px=None,
                field_of_view_deg=None,
                backend_id="mock_da3",
                device="cpu",
                dtype="float32",
                input_size=(height, width),
                warnings=[],
            )

        mock_compute.side_effect = _mock_compute
        yield mock_compute


@pytest.fixture
def benchmark_config():
    """Fast config for smoke benchmarks."""
    return EnhanceConfig(
        model_variant=ModelVariant.METRIC_LARGE,
        enable_v2=False,  # Skip V2 enhancement for pure depth benchmarks
        generate_pbr=False,  # Skip PBR for speed
        enable_manifest_cache=False,  # Test without caching first
        allow_synthetic_fallback=True,  # Allow synthetic backend in test environment
    )


# ============================================================================
# Performance Measurement Utilities
# ============================================================================


def measure_runtime_stats(runtimes_seconds):
    """Compute p50, p95, min, max from runtime samples."""
    if not runtimes_seconds:
        return {"p50": 0, "p95": 0, "min": 0, "max": 0, "mean": 0}

    sorted_times = sorted(runtimes_seconds)
    n = len(sorted_times)

    p50_idx = int(n * 0.50)
    p95_idx = int(n * 0.95)

    return {
        "p50": sorted_times[p50_idx] if n > 0 else 0,
        "p95": sorted_times[p95_idx] if p95_idx < n else sorted_times[-1],
        "min": sorted_times[0],
        "max": sorted_times[-1],
        "mean": sum(sorted_times) / n,
    }


def validate_output_invariants(depth_map):
    """Validate output invariants: dtype, range, shape, no NaNs/inf."""
    assert depth_map is not None, "Depth map is None"
    assert isinstance(depth_map, np.ndarray), f"Depth map is not ndarray: {type(depth_map)}"
    assert depth_map.ndim == 2, f"Depth map is not 2D: {depth_map.ndim}"
    assert depth_map.dtype in [np.float32, np.float64, np.uint16], f"Unexpected dtype: {depth_map.dtype}"

    # No NaNs or infinities
    assert not np.any(np.isnan(depth_map)), "Depth map contains NaN values"
    assert not np.any(np.isinf(depth_map)), "Depth map contains inf values"

    # Range check (depends on dtype)
    if depth_map.dtype == np.uint16:
        assert np.all(depth_map >= 0), "Depth map has negative values"
        assert np.all(depth_map <= 65535), "Depth map exceeds uint16 range"
    else:
        # Float depth typically normalized to [0, 1] or similar
        assert np.all(depth_map >= 0), "Depth map has negative values"
        # No strict upper bound for float depth, but check for sanity
        assert np.all(depth_map < 1e6), "Depth map has unreasonably large values"


# ============================================================================
# Smoke Benchmark Tests
# ============================================================================


class TestLuxDepthV3PerformanceBaseline:
    """Baseline performance benchmarks for lux_depth_v3 (no optimizations yet)."""

    @pytest.mark.benchmark
    def test_single_image_baseline_runtime(self, tmp_path, synthetic_images, mock_depth_backend, benchmark_config):
        """Measure baseline runtime for single image processing."""
        # Use medium-sized fixture
        fixture = synthetic_images[1]  # 512x512
        img_path = fixture["path"]
        megapixels = fixture["megapixels"]

        output_dir = tmp_path / "output_baseline"
        orchestrator = EnhanceOrchestrator(config=benchmark_config, output_root=output_dir)

        # Warm-up run (to account for any lazy initialization)
        image_input = ImageInput(path=img_path)
        _ = orchestrator.enhance_image(image_input, input_root=tmp_path)

        # Benchmark runs
        num_runs = 5
        runtimes = []

        for _ in range(num_runs):
            # Clean output between runs
            output_dir_run = tmp_path / f"output_run_{_}"
            orch = EnhanceOrchestrator(config=benchmark_config, output_root=output_dir_run)

            start = time.time()
            result = orch.enhance_image(image_input, input_root=tmp_path)
            elapsed = time.time() - start

            runtimes.append(elapsed)

            # Verify result
            assert result["status"] == "ok", f"Processing failed: {result.get('error')}"

        # Compute stats
        stats = measure_runtime_stats(runtimes)
        runtime_per_mp = stats["p50"] / megapixels

        # Print baseline metrics
        print(f"\n{'='*60}")
        print(f"Baseline Single Image Performance (512x512, {megapixels:.2f}MP)")
        print(f"  p50: {stats['p50']*1000:.1f}ms")
        print(f"  p95: {stats['p95']*1000:.1f}ms")
        print(f"  min: {stats['min']*1000:.1f}ms")
        print(f"  max: {stats['max']*1000:.1f}ms")
        print(f"  Per MP: {runtime_per_mp*1000:.1f}ms/MP")
        print(f"{'='*60}\n")

        # Store baseline for future comparison (optional)
        baseline_json = {
            "test": "single_image_baseline",
            "fixture": "512x512",
            "megapixels": megapixels,
            "p50_ms": stats["p50"] * 1000,
            "p95_ms": stats["p95"] * 1000,
            "per_mp_ms": runtime_per_mp * 1000,
        }

        # Write to artifacts if available
        artifacts_dir = tmp_path / "benchmark_results"
        artifacts_dir.mkdir(exist_ok=True)
        with open(artifacts_dir / "baseline_single_image.json", "w") as f:
            json.dump(baseline_json, f, indent=2)

        # Sanity check: processing should be reasonably fast with mocks
        # Expect < 500ms p95 for 512x512 with mocked inference
        assert stats["p95"] < 0.5, f"Baseline p95 too slow: {stats['p95']*1000:.1f}ms"

    @pytest.mark.benchmark
    def test_batch_processing_baseline(self, tmp_path, synthetic_images, mock_depth_backend, benchmark_config):
        """Measure baseline runtime for batch processing (multiple images)."""
        output_dir = tmp_path / "output_batch"
        orchestrator = EnhanceOrchestrator(config=benchmark_config, output_root=output_dir)

        # Process all fixtures in batch
        image_inputs = [ImageInput(path=fix["path"]) for fix in synthetic_images]

        start = time.time()
        results = []
        for img_input in image_inputs:
            result = orchestrator.enhance_image(img_input, input_root=tmp_path)
            results.append(result)
        batch_elapsed = time.time() - start

        # Verify all succeeded
        assert all(r["status"] == "ok" for r in results), "Some images failed processing"

        # Compute per-image stats
        num_images = len(synthetic_images)
        avg_per_image = batch_elapsed / num_images

        # Total megapixels
        total_mp = sum(fix["megapixels"] for fix in synthetic_images)
        per_mp = batch_elapsed / total_mp

        print(f"\n{'='*60}")
        print(f"Baseline Batch Performance ({num_images} images, {total_mp:.2f}MP total)")
        print(f"  Total: {batch_elapsed*1000:.1f}ms")
        print(f"  Per image: {avg_per_image*1000:.1f}ms")
        print(f"  Per MP: {per_mp*1000:.1f}ms/MP")
        print(f"{'='*60}\n")

        # Store baseline
        baseline_json = {
            "test": "batch_baseline",
            "num_images": num_images,
            "total_megapixels": total_mp,
            "total_ms": batch_elapsed * 1000,
            "avg_per_image_ms": avg_per_image * 1000,
            "per_mp_ms": per_mp * 1000,
        }

        artifacts_dir = tmp_path / "benchmark_results"
        artifacts_dir.mkdir(exist_ok=True)
        with open(artifacts_dir / "baseline_batch.json", "w") as f:
            json.dump(baseline_json, f, indent=2)

        # Sanity check: batch should complete in reasonable time with mocks
        # Expect < 2s total for 3 images with mocked inference
        assert batch_elapsed < 2.0, f"Batch baseline too slow: {batch_elapsed*1000:.1f}ms"

    @pytest.mark.benchmark
    def test_output_invariants_smoke(self, tmp_path, synthetic_images, mock_depth_backend, benchmark_config):
        """Verify output invariants across all fixtures."""
        output_dir = tmp_path / "output_invariants"
        orchestrator = EnhanceOrchestrator(config=benchmark_config, output_root=output_dir)

        for fixture in synthetic_images:
            img_input = ImageInput(path=fixture["path"])
            result = orchestrator.enhance_image(img_input, input_root=tmp_path)

            assert result["status"] == "ok", f"Processing failed for {fixture['path']}"

            # Load depth output
            depth_path = result.get("depth_path")
            if depth_path and Path(depth_path).exists():
                # Check if it's a numpy file or PNG
                if depth_path.endswith(".npy"):
                    depth_map = np.load(depth_path)
                    validate_output_invariants(depth_map)
                elif depth_path.endswith(".png"):
                    # PNG depth is quantized uint16
                    depth_img = Image.open(depth_path)
                    depth_map = np.array(depth_img)
                    validate_output_invariants(depth_map)

            # Verify output dimensions match input
            expected_height = fixture["height"]
            expected_width = fixture["width"]

            # Depth map should preserve dimensions
            if depth_path and Path(depth_path).exists():
                if depth_path.endswith(".npy"):
                    depth_map = np.load(depth_path)
                    assert depth_map.shape == (
                        expected_height,
                        expected_width,
                    ), f"Depth shape mismatch: {depth_map.shape} != ({expected_height}, {expected_width})"

        print(f"✓ Output invariants verified for {len(synthetic_images)} fixtures")

    @pytest.mark.benchmark
    def test_memory_baseline_peak_tracking(self, tmp_path, synthetic_images, mock_depth_backend, benchmark_config):
        """Establish baseline for peak memory usage (RSS tracking)."""
        try:
            import psutil
        except ImportError:
            pytest.skip("psutil not available (requires ML dependencies)")

        import os

        process = psutil.Process(os.getpid())

        # Measure baseline RSS before processing
        baseline_rss_mb = process.memory_info().rss / 1024 / 1024

        output_dir = tmp_path / "output_memory"
        orchestrator = EnhanceOrchestrator(config=benchmark_config, output_root=output_dir)

        # Process largest fixture
        fixture = synthetic_images[2]  # 1024x768
        img_input = ImageInput(path=fixture["path"])

        # Measure peak RSS during processing
        result = orchestrator.enhance_image(img_input, input_root=tmp_path)
        assert result["status"] == "ok"

        peak_rss_mb = process.memory_info().rss / 1024 / 1024
        incremental_mb = peak_rss_mb - baseline_rss_mb

        print(f"\n{'='*60}")
        print(f"Memory Baseline ({fixture['width']}x{fixture['height']}, {fixture['megapixels']:.2f}MP)")
        print(f"  Baseline RSS: {baseline_rss_mb:.1f}MB")
        print(f"  Peak RSS: {peak_rss_mb:.1f}MB")
        print(f"  Incremental: {incremental_mb:.1f}MB")
        print(f"  Per MP: {incremental_mb / fixture['megapixels']:.1f}MB/MP")
        print(f"{'='*60}\n")

        # Store baseline
        baseline_json = {
            "test": "memory_baseline",
            "fixture": f"{fixture['width']}x{fixture['height']}",
            "megapixels": fixture["megapixels"],
            "baseline_rss_mb": baseline_rss_mb,
            "peak_rss_mb": peak_rss_mb,
            "incremental_mb": incremental_mb,
            "per_mp_mb": incremental_mb / fixture["megapixels"],
        }

        artifacts_dir = tmp_path / "benchmark_results"
        artifacts_dir.mkdir(exist_ok=True)
        with open(artifacts_dir / "baseline_memory.json", "w") as f:
            json.dump(baseline_json, f, indent=2)

        # Sanity check: incremental memory should be reasonable with mocks
        # Expect < 500MB incremental for 1024x768 with mocked inference
        assert incremental_mb < 500, f"Incremental memory too high: {incremental_mb:.1f}MB"

    @pytest.mark.benchmark
    def test_no_model_reinitialization_guard(self, tmp_path, synthetic_images, mock_depth_backend, benchmark_config):
        """Guard against accidental repeated model loads (baseline check)."""
        output_dir = tmp_path / "output_reinit_check"
        orchestrator = EnhanceOrchestrator(config=benchmark_config, output_root=output_dir)

        # Process multiple images
        for fixture in synthetic_images[:2]:  # Just 2 for speed
            img_input = ImageInput(path=fixture["path"])
            result = orchestrator.enhance_image(img_input, input_root=tmp_path)
            assert result["status"] == "ok"

        # Note: This is a smoke test - actual model singleton checks come in L1.0
        # With synthetic fallback, the mock won't be called, so just verify processing succeeded
        print(f"✓ Processed {2} images successfully (reinitialization guard placeholder)")
        print("  Note: Full backend singleton validation will be implemented in L1.0 (Backend Warm Pool)")


# ============================================================================
# Regression Guard Placeholder
# ============================================================================


class TestRegressionGuards:
    """Regression guards for future optimizations (will be populated in L1.x PRs)."""

    @pytest.mark.benchmark
    def test_p95_latency_regression_threshold(self, tmp_path):
        """Placeholder for p95 latency regression detection.

        Future PRs will populate this with actual thresholds.
        For now, just verify JSON output format.
        """
        # Placeholder baseline
        baseline_p95_ms = 100.0  # Will be updated after L0.0 merge

        # Load actual baseline if available
        baseline_file = tmp_path.parent / "benchmark_results" / "baseline_single_image.json"
        if baseline_file.exists():
            with open(baseline_file) as f:
                baseline_data = json.load(f)
                baseline_p95_ms = baseline_data.get("p95_ms", 100.0)

        # Placeholder assertion
        # Real regression check will compare current run vs stored baseline
        assert baseline_p95_ms > 0, "Baseline p95 should be positive"

        print(f"✓ Regression threshold check (baseline p95: {baseline_p95_ms:.1f}ms)")
        print("  Note: Actual regression detection will be implemented after L0.0 baseline is established")
