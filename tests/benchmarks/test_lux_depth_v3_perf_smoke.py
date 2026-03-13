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
import subprocess
import sys
import threading
import time
import textwrap
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Import lux_depth_v3 components
from transformation_portal.lux_depth_v3._benchmark_contract import (
    assert_regression_within_tolerance,
    load_benchmark_metrics,
    write_benchmark_metrics,
)
from transformation_portal.lux_depth_v3.config import EnhanceConfig, ModelVariant
from transformation_portal.lux_depth_v3.input_manager import ImageInput
from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

# Mark all tests as benchmark tier
pytestmark = [pytest.mark.benchmark]
BASELINES_DIR = Path(__file__).with_name("baselines")
REPO_ROOT = Path(__file__).resolve().parents[2]


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


def committed_baseline_path(filename: str) -> Path:
    """Return path to a committed benchmark baseline fixture."""
    return BASELINES_DIR / filename


def write_runtime_baseline(tmp_path: Path, filename: str, payload: dict) -> None:
    """Persist runtime metrics for debugging and local comparison."""
    artifacts_dir = Path(
        os.environ.get(
            "BENCHMARK_ARTIFACTS_DIR",
            tmp_path / "benchmark_results",
        )
    )
    write_benchmark_metrics(artifacts_dir / filename, payload)


def measure_memory_baseline_subprocess(fixture_path: Path, input_root: Path, output_dir: Path) -> dict:
    """Measure processing RSS in a fresh subprocess to avoid in-worker memory drift."""
    script = textwrap.dedent("""
        import json
        import threading
        from pathlib import Path

        import psutil

        from transformation_portal.lux_depth_v3.config import EnhanceConfig, ModelVariant
        from transformation_portal.lux_depth_v3.input_manager import ImageInput
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator


        class PeakRSSTracker:
            def __init__(self, process, interval: float = 0.005):
                self.process = process
                self.interval = interval
                self.peak_rss_bytes = 0
                self.samples = 0
                self._stop = threading.Event()
                self._ready = threading.Event()
                self._thread = None

            def __enter__(self):
                self.peak_rss_bytes = self.process.memory_info().rss
                self.samples = 0
                self._stop.clear()
                self._ready.clear()
                self._thread = threading.Thread(target=self._poll, daemon=True)
                self._thread.start()
                timeout = max(0.05, self.interval * 10)
                if not self._ready.wait(timeout=timeout):
                    self._stop.set()
                    if self._thread is not None:
                        self._thread.join(timeout=1.0)
                    raise RuntimeError(
                        f"PeakRSSTracker first-sample barrier timed out after {timeout}s"
                    )
                return self

            def __exit__(self, exc_type, exc, tb):
                self._stop.set()
                if self._thread is not None:
                    self._thread.join(timeout=1.0)
                self._sample()
                return False

            def _poll(self):
                self._sample()
                self._ready.set()
                while not self._stop.wait(self.interval):
                    self._sample()

            def _sample(self):
                try:
                    rss = self.process.memory_info().rss
                except (ProcessLookupError, PermissionError):
                    return
                self.samples += 1
                if rss > self.peak_rss_bytes:
                    self.peak_rss_bytes = rss

            @property
            def peak_rss_mb(self):
                return self.peak_rss_bytes / 1024 / 1024


        fixture_path = Path(__import__("sys").argv[1])
        input_root = Path(__import__("sys").argv[2])
        output_dir = Path(__import__("sys").argv[3])

        process = psutil.Process()
        config = EnhanceConfig(
            model_variant=ModelVariant.METRIC_LARGE,
            enable_v2=False,
            generate_pbr=False,
            enable_manifest_cache=False,
            allow_synthetic_fallback=True,
            depth_backend="synthetic",
        )
        orchestrator = EnhanceOrchestrator(config=config, output_root=output_dir)
        baseline_rss_mb = process.memory_info().rss / 1024 / 1024

        with PeakRSSTracker(process) as tracker:
            result = orchestrator.enhance_image(ImageInput(path=fixture_path), input_root=input_root)

        if result["status"] != "ok":
            raise RuntimeError(f"Unexpected benchmark status: {result}")

        peak_rss_mb = tracker.peak_rss_mb
        post_rss_mb = process.memory_info().rss / 1024 / 1024
        print(
            json.dumps(
                {
                    "baseline_rss_mb": baseline_rss_mb,
                    "peak_rss_mb": peak_rss_mb,
                    "post_processing_rss_mb": post_rss_mb,
                    "incremental_mb": peak_rss_mb - baseline_rss_mb,
                    "sampling_interval_s": tracker.interval,
                    "sample_count": tracker.samples,
                },
                sort_keys=True,
            )
        )
        """)
    pythonpath_entries = [str(REPO_ROOT / "src")]
    existing_pythonpath = os.environ.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(pythonpath_entries)}
    completed = subprocess.run(
        [sys.executable, "-c", script, str(fixture_path), str(input_root), str(output_dir)],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


class PeakRSSTracker:
    """Track true peak RSS during a code block using a polling thread.

    Uses a daemon thread to sample ``psutil.Process.memory_info().rss`` at
    a configurable interval and records the high-water mark.  A first-sample
    barrier ensures at least one poll completes before ``__enter__`` returns,
    preventing early allocation spikes from being missed.

    Usage::

        with PeakRSSTracker(process) as tracker:
            do_work()
        print(tracker.peak_rss_mb, tracker.samples)
    """

    def __init__(self, process, interval: float = 0.005):
        self.process = process
        self.interval = interval
        self.peak_rss_bytes: int = 0
        self.samples: int = 0
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        self.peak_rss_bytes = self.process.memory_info().rss
        self.samples = 0
        self._stop.clear()
        self._ready.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        timeout = max(0.05, self.interval * 10)
        if not self._ready.wait(timeout=timeout):
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=1.0)
            raise RuntimeError(
                f"PeakRSSTracker first-sample barrier timed out after {timeout}s. "
                f"System may be under extreme load; measurement would be invalid."
            )
        return self

    def _poll(self):
        # Immediate first sample before signalling readiness
        self._sample()
        self._ready.set()

        while not self._stop.wait(self.interval):
            self._sample()

    def _sample(self):
        try:
            rss = self.process.memory_info().rss
        except (ProcessLookupError, PermissionError):
            return
        self.samples += 1
        if rss > self.peak_rss_bytes:
            self.peak_rss_bytes = rss

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
        write_runtime_baseline(tmp_path, "baseline_cold_start.json", baseline_json)
        committed_baseline = load_benchmark_metrics(
            committed_baseline_path("baseline_cold_start.json"),
        )
        assert_regression_within_tolerance(
            label="Cold-start p95",
            measured_value=baseline_json["p95_ms"],
            baseline_value=float(committed_baseline["p95_ms"]),
            tolerance_fraction=0.25,
            unit="ms",
        )

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

        write_runtime_baseline(tmp_path, "baseline_steady_state.json", baseline_json)
        committed_baseline = load_benchmark_metrics(
            committed_baseline_path("baseline_steady_state.json"),
        )
        assert_regression_within_tolerance(
            label="Steady-state p95",
            measured_value=baseline_json["p95_ms"],
            baseline_value=float(committed_baseline["p95_ms"]),
            tolerance_fraction=0.20,
            unit="ms",
        )

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

        write_runtime_baseline(tmp_path, "baseline_batch.json", baseline_json)
        committed_baseline = load_benchmark_metrics(
            committed_baseline_path("baseline_batch.json"),
        )
        assert_regression_within_tolerance(
            label="Batch total runtime",
            measured_value=baseline_json["total_ms"],
            baseline_value=float(committed_baseline["total_ms"]),
            tolerance_fraction=0.20,
            unit="ms",
        )

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
    def test_memory_peak_rss_baseline(self, tmp_path, synthetic_images):
        """Establish baseline for peak RSS memory during image processing.

        Measures processing-only peak RSS: baseline is taken AFTER orchestrator
        construction so that ``incremental_mb`` reflects only the memory used by
        ``enhance_image()``, not one-time initialization overhead.

        Runs in a fresh subprocess so RSS is not polluted by previous tests in
        the same pytest worker.
        """
        try:
            import psutil
        except ImportError:
            pytest.skip("psutil not available (requires ML dependencies)")

        # Process largest fixture with peak RSS tracking
        fixture = synthetic_images[2]  # 1024x768
        output_dir = tmp_path / "output_memory"
        measurement = measure_memory_baseline_subprocess(
            fixture_path=fixture["path"],
            input_root=tmp_path,
            output_dir=output_dir,
        )
        baseline_rss_mb = float(measurement["baseline_rss_mb"])
        peak_rss_mb = float(measurement["peak_rss_mb"])
        post_rss_mb = float(measurement["post_processing_rss_mb"])
        incremental_mb = float(measurement["incremental_mb"])
        sample_count = int(measurement["sample_count"])
        sampling_interval_s = float(measurement["sampling_interval_s"])

        print(f"\n{'='*60}")
        print(f"Memory Baseline ({fixture['width']}x{fixture['height']}, {fixture['megapixels']:.2f}MP)")
        print(f"  Baseline RSS (post-init): {baseline_rss_mb:.1f}MB")
        print(f"  Peak RSS (polled): {peak_rss_mb:.1f}MB")
        print(f"  Post-processing RSS: {post_rss_mb:.1f}MB")
        print(f"  Incremental (peak - baseline): {incremental_mb:.1f}MB")
        print(f"  Per MP: {incremental_mb / fixture['megapixels']:.1f}MB/MP")
        print(f"  Samples: {sample_count}")
        print("  Semantic: processing-only (excludes orchestrator init)")
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
            "measurement_semantic": "processing_only",
            "sampling_interval_s": sampling_interval_s,
            "sample_count": sample_count,
        }

        write_runtime_baseline(tmp_path, "baseline_memory.json", baseline_json)
        committed_baseline = load_benchmark_metrics(
            committed_baseline_path("baseline_memory.json"),
        )
        assert_regression_within_tolerance(
            label="Incremental peak RSS",
            measured_value=baseline_json["incremental_mb"],
            baseline_value=float(committed_baseline["incremental_mb"]),
            tolerance_fraction=0.20,
            unit="MB",
        )

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
    """Regression guard fixture sanity for committed benchmark baselines."""

    @pytest.mark.benchmark
    def test_cold_start_p95_regression_threshold(self, tmp_path):
        """Committed cold-start baseline must exist and remain well-formed."""
        baseline_data = load_benchmark_metrics(
            committed_baseline_path("baseline_cold_start.json"),
        )
        assert baseline_data["p95_ms"] > 0
        assert baseline_data["measurement_type"] == "cold_start"

    @pytest.mark.benchmark
    def test_steady_state_p95_regression_threshold(self, tmp_path):
        """Committed steady-state baseline must exist and remain well-formed."""
        baseline_data = load_benchmark_metrics(
            committed_baseline_path("baseline_steady_state.json"),
        )
        assert baseline_data["p95_ms"] > 0
        assert baseline_data["measurement_type"] == "steady_state"
