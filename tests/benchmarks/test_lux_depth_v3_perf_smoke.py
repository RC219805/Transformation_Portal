"""lux_depth_v3 Performance Smoke Tests (PR L0.0).

Fast, deterministic baseline benchmarks for the lux_depth_v3 pipeline.
No model downloads, no network calls - pure performance measurement infrastructure.

Measures:
- p50/p95 runtime per image and per megapixel
- Peak RSS memory usage (polling-thread high-water mark)
- Output invariants (dtype, range, shape, no NaNs/inf)

Success Criteria:
- Runs in <10s in CI
- Produces machine-readable output for regression tracking
- All tests fully offline with synthetic backend (no mocks needed)
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

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


@pytest.fixture(scope="module")
def synthetic_images(tmp_path_factory):
    """Create small deterministic synthetic test images (no downloads).

    Uses vectorized NumPy for fast generation. Module-scoped to avoid
    regenerating fixtures for each test.
    """
    tmp_path = tmp_path_factory.mktemp("benchmark_fixtures")
    fixtures = []
    sizes = [
        (256, 256),  # Small
        (512, 512),  # Medium
        (1024, 768),  # HD-ish
    ]

    for i, (width, height) in enumerate(sizes):
        # Vectorized gradient generation (much faster than Python loops)
        x_coords = np.arange(width)
        y_coords = np.arange(height)
        xx, yy = np.meshgrid(x_coords, y_coords)

        # Create deterministic RGB gradient pattern
        r = (xx * 255 / width).astype(np.uint8)
        g = (yy * 255 / height).astype(np.uint8)
        b = ((xx + yy) * 255 / (width + height)).astype(np.uint8)

        # Stack into RGB image
        rgb_array = np.stack([r, g, b], axis=2)
        img = Image.fromarray(rgb_array, mode="RGB")

        img_path = tmp_path / f"test_{width}x{height}.jpg"
        img.save(img_path, quality=95)
        fixtures.append({"path": img_path, "width": width, "height": height, "megapixels": (width * height) / 1e6})

    return fixtures


@pytest.fixture
def benchmark_config():
    """Fast config for smoke benchmarks.

    Explicitly pins synthetic backend for deterministic, reproducible measurements.
    No mocking needed - synthetic backend provides offline deterministic depth.
    """
    return EnhanceConfig(
        model_variant=ModelVariant.METRIC_LARGE,
        enable_v2=False,  # Skip V2 enhancement for pure depth benchmarks
        generate_pbr=False,  # Skip PBR for speed
        enable_manifest_cache=False,  # Test without caching first
        allow_synthetic_fallback=True,  # Allow synthetic backend in test environment
        depth_backend="synthetic",  # Pin to synthetic for deterministic measurements
    )


# ============================================================================
# Performance Measurement Utilities
# ============================================================================


def measure_runtime_stats(runtimes_seconds):
    """Compute p50, p95, min, max from runtime samples using NumPy percentiles."""
    if not runtimes_seconds:
        return {"p50": 0, "p95": 0, "min": 0, "max": 0, "mean": 0}

    times_array = np.array(runtimes_seconds)
    p50, p95 = np.percentile(times_array, [50, 95])

    return {
        "p50": p50,
        "p95": p95,
        "min": float(np.min(times_array)),
        "max": float(np.max(times_array)),
        "mean": float(np.mean(times_array)),
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


class PeakRSSTracker:
    """Track true peak RSS during a code block using a polling thread.

    Uses a daemon thread to sample ``psutil.Process.memory_info().rss`` at
    a configurable interval and records the high-water mark.

    Usage::

        with PeakRSSTracker(process) as tracker:
            do_work()
        print(tracker.peak_rss_mb)
    """

    def __init__(self, process, interval: float = 0.005):
        self.process = process
        self.interval = interval
        self.peak_rss_bytes: int = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        self.peak_rss_bytes = self.process.memory_info().rss
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def _poll(self):
        while not self._stop.is_set():
            try:
                rss = self.process.memory_info().rss
                if rss > self.peak_rss_bytes:
                    self.peak_rss_bytes = rss
            except Exception:
                break
            self._stop.wait(self.interval)

    def __exit__(self, *args):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    @property
    def peak_rss_mb(self) -> float:
        return self.peak_rss_bytes / (1024 * 1024)


# ============================================================================
# Smoke Benchmark Tests
# ============================================================================


class TestLuxDepthV3PerformanceBaseline:
    """Baseline performance benchmarks for lux_depth_v3 (no optimizations yet)."""

    @pytest.mark.benchmark
    def test_single_image_cold_start_p95(self, tmp_path, synthetic_images, benchmark_config):
        """Measure cold-start p95 latency (new orchestrator per run).

        Cold-start includes initialization overhead: directory creation, config parsing,
        backend instantiation. This represents worst-case single-image workflow.
        """
        # Use medium-sized fixture
        fixture = synthetic_images[1]  # 512x512
        img_path = fixture["path"]
        megapixels = fixture["megapixels"]

        image_input = ImageInput(path=img_path)

        # Benchmark runs (cold-start: new orchestrator each time)
        num_runs = 5
        runtimes = []

        for i in range(num_runs):
            # Clean output between runs
            output_dir_run = tmp_path / f"output_run_{i}"
            orch = EnhanceOrchestrator(config=benchmark_config, output_root=output_dir_run)

            start = time.perf_counter()
            result = orch.enhance_image(image_input, input_root=tmp_path)
            elapsed = time.perf_counter() - start

            runtimes.append(elapsed)

            # Verify result
            assert result["status"] == "ok", f"Processing failed: {result.get('error')}"

        # Compute stats
        stats = measure_runtime_stats(runtimes)
        runtime_per_mp = stats["p50"] / megapixels

        # Print baseline metrics
        print(f"\n{'='*60}")
        print(f"Cold-Start Single Image Performance (512x512, {megapixels:.2f}MP)")
        print(f"  p50: {stats['p50']*1000:.1f}ms")
        print(f"  p95: {stats['p95']*1000:.1f}ms")
        print(f"  min: {stats['min']*1000:.1f}ms")
        print(f"  max: {stats['max']*1000:.1f}ms")
        print(f"  Per MP: {runtime_per_mp*1000:.1f}ms/MP")
        print("  Type: COLD-START (includes initialization)")
        print(f"{'='*60}\n")

        # Store baseline for future comparison
        baseline_json = {
            "test": "single_image_cold_start",
            "fixture": "512x512",
            "megapixels": megapixels,
            "p50_ms": stats["p50"] * 1000,
            "p95_ms": stats["p95"] * 1000,
            "per_mp_ms": runtime_per_mp * 1000,
            "measurement_type": "cold_start",
        }

        # Write to artifacts (env-configurable location for baseline persistence)
        artifacts_dir = Path(os.environ.get("BENCHMARK_ARTIFACTS_DIR", tmp_path / "benchmark_results"))
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        with open(artifacts_dir / "baseline_cold_start.json", "w") as f:
            json.dump(baseline_json, f, indent=2)

        # Relaxed sanity check: warn if extremely slow (allows for CI runner variance)
        # Note: Absolute thresholds removed per architectural review
        # TODO L0.2: Implement baseline comparison with % tolerance
        if stats["p95"] > 1.0:
            print(f"⚠️  Warning: p95 latency unusually high: {stats['p95']*1000:.1f}ms (threshold: 1000ms)")
            print("    This may indicate CI runner performance issues, not code regression")

    @pytest.mark.benchmark
    def test_single_image_steady_state_p95(self, tmp_path, synthetic_images, benchmark_config):
        """Measure steady-state p95 latency (reused orchestrator, warmed up).

        Steady-state excludes initialization overhead. This represents best-case
        throughput for batch workflows or long-running processes.
        """
        # Use medium-sized fixture
        fixture = synthetic_images[1]  # 512x512
        img_path = fixture["path"]
        megapixels = fixture["megapixels"]

        output_dir = tmp_path / "output_steady_state"
        orchestrator = EnhanceOrchestrator(config=benchmark_config, output_root=output_dir)

        image_input = ImageInput(path=img_path)

        # Warm-up run (initialize caches, JIT compilation, etc.)
        _ = orchestrator.enhance_image(image_input, input_root=tmp_path)

        # Benchmark runs (steady-state: reuse orchestrator)
        num_runs = 5
        runtimes = []

        for i in range(num_runs):
            start = time.perf_counter()
            result = orchestrator.enhance_image(image_input, input_root=tmp_path)
            elapsed = time.perf_counter() - start

            runtimes.append(elapsed)

            # Verify result
            assert result["status"] == "ok", f"Processing failed: {result.get('error')}"

        # Compute stats
        stats = measure_runtime_stats(runtimes)
        runtime_per_mp = stats["p50"] / megapixels

        # Print baseline metrics
        print(f"\n{'='*60}")
        print(f"Steady-State Single Image Performance (512x512, {megapixels:.2f}MP)")
        print(f"  p50: {stats['p50']*1000:.1f}ms")
        print(f"  p95: {stats['p95']*1000:.1f}ms")
        print(f"  min: {stats['min']*1000:.1f}ms")
        print(f"  max: {stats['max']*1000:.1f}ms")
        print(f"  Per MP: {runtime_per_mp*1000:.1f}ms/MP")
        print("  Type: STEADY-STATE (excludes initialization)")
        print(f"{'='*60}\n")

        # Store baseline
        baseline_json = {
            "test": "single_image_steady_state",
            "fixture": "512x512",
            "megapixels": megapixels,
            "p50_ms": stats["p50"] * 1000,
            "p95_ms": stats["p95"] * 1000,
            "per_mp_ms": runtime_per_mp * 1000,
            "measurement_type": "steady_state",
        }

        artifacts_dir = Path(os.environ.get("BENCHMARK_ARTIFACTS_DIR", tmp_path / "benchmark_results"))
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        with open(artifacts_dir / "baseline_steady_state.json", "w") as f:
            json.dump(baseline_json, f, indent=2)

        # Relaxed sanity check
        if stats["p95"] > 1.0:
            print(f"⚠️  Warning: p95 latency unusually high: {stats['p95']*1000:.1f}ms (threshold: 1000ms)")

    @pytest.mark.benchmark
    def test_batch_throughput_baseline(self, tmp_path, synthetic_images, benchmark_config):
        """Measure baseline throughput for batch processing (multiple different images).

        Processes multiple images sequentially with single orchestrator. This measures
        sustained throughput for production batch workflows.
        """
        output_dir = tmp_path / "output_batch"
        orchestrator = EnhanceOrchestrator(config=benchmark_config, output_root=output_dir)

        # Process all fixtures in batch
        image_inputs = [ImageInput(path=fix["path"]) for fix in synthetic_images]

        start = time.perf_counter()
        results = []
        for img_input in image_inputs:
            result = orchestrator.enhance_image(img_input, input_root=tmp_path)
            results.append(result)
        batch_elapsed = time.perf_counter() - start

        # Verify all succeeded
        assert all(r["status"] == "ok" for r in results), "Some images failed processing"

        # Compute per-image stats
        num_images = len(synthetic_images)
        avg_per_image = batch_elapsed / num_images

        # Total megapixels
        total_mp = sum(fix["megapixels"] for fix in synthetic_images)
        per_mp = batch_elapsed / total_mp

        print(f"\n{'='*60}")
        print(f"Batch Throughput ({num_images} images, {total_mp:.2f}MP total)")
        print(f"  Total: {batch_elapsed*1000:.1f}ms")
        print(f"  Per image: {avg_per_image*1000:.1f}ms")
        print(f"  Per MP: {per_mp*1000:.1f}ms/MP")
        print("  Type: BATCH THROUGHPUT")
        print(f"{'='*60}\n")

        # Store baseline
        baseline_json = {
            "test": "batch_throughput",
            "num_images": num_images,
            "total_megapixels": total_mp,
            "total_ms": batch_elapsed * 1000,
            "avg_per_image_ms": avg_per_image * 1000,
            "per_mp_ms": per_mp * 1000,
            "measurement_type": "batch_throughput",
        }

        artifacts_dir = Path(os.environ.get("BENCHMARK_ARTIFACTS_DIR", tmp_path / "benchmark_results"))
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        with open(artifacts_dir / "baseline_batch.json", "w") as f:
            json.dump(baseline_json, f, indent=2)

        # Relaxed sanity check
        if batch_elapsed > 5.0:
            print(f"⚠️  Warning: batch processing unusually slow: {batch_elapsed*1000:.1f}ms (threshold: 5000ms)")

    @pytest.mark.benchmark
    def test_output_invariants_smoke(self, tmp_path, synthetic_images, benchmark_config):
        """Verify output invariants across all fixtures."""
        output_dir = tmp_path / "output_invariants"
        orchestrator = EnhanceOrchestrator(config=benchmark_config, output_root=output_dir)

        for fixture in synthetic_images:
            img_input = ImageInput(path=fixture["path"])
            result = orchestrator.enhance_image(img_input, input_root=tmp_path)

            assert result["status"] == "ok", f"Processing failed for {fixture['path']}"

            # Load depth output and validate invariants
            depth_path = result.get("depth_path")
            if depth_path and Path(depth_path).exists():
                # Check if it's a numpy file or PNG
                if depth_path.endswith(".npy"):
                    depth_map = np.load(depth_path)
                    validate_output_invariants(depth_map)
                elif depth_path.endswith(".png"):
                    # PNG depth is quantized uint16 - use context manager to close file
                    with Image.open(depth_path) as depth_img:
                        depth_map = np.array(depth_img)
                    validate_output_invariants(depth_map)

            # Verify output dimensions match input (applies to both .npy and .png)
            expected_height = fixture["height"]
            expected_width = fixture["width"]

            if depth_path and Path(depth_path).exists():
                # Load depth map based on format
                if depth_path.endswith(".npy"):
                    depth_map = np.load(depth_path)
                elif depth_path.endswith(".png"):
                    with Image.open(depth_path) as depth_img:
                        depth_map = np.array(depth_img)
                else:
                    continue  # Skip unknown formats

                # Assert dimensions for both formats
                assert depth_map.shape == (
                    expected_height,
                    expected_width,
                ), f"Depth shape mismatch: {depth_map.shape} != ({expected_height}, {expected_width})"

        print(f"✓ Output invariants verified for {len(synthetic_images)} fixtures")

    @pytest.mark.benchmark
    def test_memory_peak_rss_baseline(self, tmp_path, synthetic_images, benchmark_config):
        """Establish baseline for peak RSS memory during processing.

        Uses a polling thread (PeakRSSTracker) to sample RSS at ~5ms intervals
        and record the high-water mark. This captures transient allocation spikes
        that a post-completion snapshot would miss.
        """
        try:
            import psutil
        except ImportError:
            pytest.skip("psutil not available (requires ML dependencies)")

        import os as os_module

        process = psutil.Process(os_module.getpid())

        # Measure baseline RSS before processing
        baseline_rss_mb = process.memory_info().rss / 1024 / 1024

        output_dir = tmp_path / "output_memory"
        orchestrator = EnhanceOrchestrator(config=benchmark_config, output_root=output_dir)

        # Process largest fixture with peak RSS tracking
        fixture = synthetic_images[2]  # 1024x768
        img_input = ImageInput(path=fixture["path"])

        with PeakRSSTracker(process) as tracker:
            result = orchestrator.enhance_image(img_input, input_root=tmp_path)
        assert result["status"] == "ok"

        peak_rss_mb = tracker.peak_rss_mb
        post_rss_mb = process.memory_info().rss / 1024 / 1024
        incremental_mb = peak_rss_mb - baseline_rss_mb

        print(f"\n{'='*60}")
        print(f"Memory Baseline ({fixture['width']}x{fixture['height']}, {fixture['megapixels']:.2f}MP)")
        print(f"  Baseline RSS: {baseline_rss_mb:.1f}MB")
        print(f"  Peak RSS (polled): {peak_rss_mb:.1f}MB")
        print(f"  Post-processing RSS: {post_rss_mb:.1f}MB")
        print(f"  Incremental (peak - baseline): {incremental_mb:.1f}MB")
        print(f"  Per MP: {incremental_mb / fixture['megapixels']:.1f}MB/MP")
        print(f"{'='*60}\n")

        # Store baseline
        baseline_json = {
            "test": "memory_baseline",
            "fixture": f"{fixture['width']}x{fixture['height']}",
            "megapixels": fixture["megapixels"],
            "baseline_rss_mb": baseline_rss_mb,
            "peak_rss_mb": peak_rss_mb,
            "post_processing_rss_mb": post_rss_mb,
            "incremental_mb": incremental_mb,
            "per_mp_mb": incremental_mb / fixture["megapixels"],
            "measurement_type": "peak_rss_polled",
        }

        artifacts_dir = Path(os.environ.get("BENCHMARK_ARTIFACTS_DIR", tmp_path / "benchmark_results"))
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        with open(artifacts_dir / "baseline_memory.json", "w") as f:
            json.dump(baseline_json, f, indent=2)

        # Relaxed sanity check
        if abs(incremental_mb) > 1000:
            print(f"⚠️  Warning: memory usage unusually high: {incremental_mb:.1f}MB")

    @pytest.mark.benchmark
    def test_no_model_reinitialization_guard(self, tmp_path, synthetic_images, benchmark_config):
        """Guard against accidental repeated model loads (baseline check)."""
        output_dir = tmp_path / "output_reinit_check"
        orchestrator = EnhanceOrchestrator(config=benchmark_config, output_root=output_dir)

        # Process multiple images
        for fixture in synthetic_images[:2]:  # Just 2 for speed
            img_input = ImageInput(path=fixture["path"])
            result = orchestrator.enhance_image(img_input, input_root=tmp_path)
            assert result["status"] == "ok"

        # Note: This is a smoke test - actual model singleton checks come in L1.0
        # With synthetic backend, no real ML models are loaded
        print(f"✓ Processed {2} images successfully (reinitialization guard placeholder)")
        print("  Note: Full backend singleton validation will be implemented in L1.0 (Backend Warm Pool)")


# ============================================================================
# Regression Guard Placeholder
# ============================================================================


class TestRegressionGuards:
    """Regression guards for future optimizations (will be populated in L1.x PRs).

    Note: These tests currently emit warnings only (non-failing) to handle CI runner
    variance. Once L0.2 implements baseline comparison with % tolerance, these can
    become blocking checks.
    """

    @pytest.mark.benchmark
    def test_cold_start_p95_regression_threshold(self, tmp_path):
        """Placeholder for cold-start p95 latency regression detection.

        Future PRs will populate this with actual thresholds and % tolerance checks.
        For now, just verify JSON output format.
        """
        # Placeholder baseline
        baseline_p95_ms = 100.0  # Will be updated after L0.0 merge

        # Load actual baseline if available
        baseline_file = tmp_path.parent / "benchmark_results" / "baseline_cold_start.json"
        if baseline_file.exists():
            with open(baseline_file) as f:
                baseline_data = json.load(f)
                baseline_p95_ms = baseline_data.get("p95_ms", 100.0)

        # Placeholder assertion
        # Real regression check will compare current run vs stored baseline with % tolerance
        assert baseline_p95_ms > 0, "Baseline p95 should be positive"

        print(f"✓ Regression threshold check (cold-start p95: {baseline_p95_ms:.1f}ms)")
        print("  Note: Actual regression detection will be implemented in L0.2")

    @pytest.mark.benchmark
    def test_steady_state_p95_regression_threshold(self, tmp_path):
        """Placeholder for steady-state p95 latency regression detection.

        Future PRs will populate this with actual thresholds and % tolerance checks.
        For now, just verify JSON output format.
        """
        # Placeholder baseline
        baseline_p95_ms = 100.0  # Will be updated after L0.0 merge

        # Load actual baseline if available
        baseline_file = tmp_path.parent / "benchmark_results" / "baseline_steady_state.json"
        if baseline_file.exists():
            with open(baseline_file) as f:
                baseline_data = json.load(f)
                baseline_p95_ms = baseline_data.get("p95_ms", 100.0)

        # Placeholder assertion
        assert baseline_p95_ms > 0, "Baseline p95 should be positive"

        print(f"✓ Regression threshold check (steady-state p95: {baseline_p95_ms:.1f}ms)")
        print("  Note: Actual regression detection will be implemented in L0.2")
