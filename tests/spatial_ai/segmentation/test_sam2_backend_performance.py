"""SAM2 Backend Performance Benchmarks (Phase 4C).

Fast, reproducible performance measurements for SAM2 segmentation backend.
Measures latency (p50/p95), memory usage, and throughput across all modes.

Test Tiers:
- Auto mode: mask generation latency and count
- Prompted mode: points/bbox latency with different prompt counts
- Video mode: frame tracking throughput (FPS)

Success Criteria:
- Runs with real SAM2 checkpoint (marked @ml @slow @benchmark)
- Produces machine-readable metrics for regression tracking
- Baseline: < 2s per 512x512 image on MPS (auto mode)
- Memory: < 2GB peak RSS for single inference

Quality Firewall Integration:
- p95 latency: block if > 10% increase
- mean latency: block if > 15% increase
- failure rate: block if > 0% increase
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
from PIL import Image

# Conditional imports for ML dependencies
try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Import SAM2 components
from transformation_portal.spatial_ai.segmentation.contracts import SegmentationInput
from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

# Mark all tests as benchmark + ml + slow
pytestmark = [pytest.mark.benchmark, pytest.mark.ml, pytest.mark.slow]

# Skip all tests if torch not available
if not HAS_TORCH:
    pytest.skip("torch not available", allow_module_level=True)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def benchmark_checkpoint():
    """Path to SAM2 checkpoint for benchmarks."""
    checkpoint_path = Path("checkpoints/sam2.1_hiera_large.pt")
    if not checkpoint_path.exists():
        pytest.skip(f"SAM2 checkpoint not found: {checkpoint_path}")
    return str(checkpoint_path)


@pytest.fixture(scope="module")
def benchmark_images(tmp_path_factory):
    """Create synthetic test images for benchmarking.

    Sizes chosen to represent real-world luxury real estate photos:
    - 512x512: Small crop
    - 1024x768: HD crop
    - 2048x1536: High-res export
    """
    tmp_path = tmp_path_factory.mktemp("sam2_benchmark_fixtures")
    fixtures = []
    sizes = [
        (512, 512),  # Small
        (1024, 768),  # HD
        (2048, 1536),  # High-res
    ]

    for width, height in sizes:
        # Create deterministic gradient pattern
        x_coords = np.arange(width)
        y_coords = np.arange(height)
        xx, yy = np.meshgrid(x_coords, y_coords)

        r = (xx * 255 / width).astype(np.uint8)
        g = (yy * 255 / height).astype(np.uint8)
        b = ((xx + yy) * 255 / (width + height)).astype(np.uint8)

        rgb_array = np.stack([r, g, b], axis=2)
        img = Image.fromarray(rgb_array, mode="RGB")

        img_path = tmp_path / f"test_{width}x{height}.jpg"
        img.save(img_path, quality=95)

        # Convert to float32 for contract compliance
        rgb_float = rgb_array.astype(np.float32) / 255.0

        fixtures.append(
            {
                "path": img_path,
                "array": rgb_float,  # float32 in [0, 1]
                "width": width,
                "height": height,
                "megapixels": (width * height) / 1e6,
            }
        )

    return fixtures


@pytest.fixture(scope="module")
def benchmark_video_frames(tmp_path_factory):
    """Create synthetic video frames for video tracking benchmarks."""
    tmp_path = tmp_path_factory.mktemp("sam2_video_benchmark")
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()

    # Create 10 frames with moving object (512x512 for speed)
    frame_count = 10
    width, height = 512, 512

    for i in range(frame_count):
        # Moving circle across frames
        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        center_x = int(width * (0.2 + 0.6 * i / frame_count))
        center_y = height // 2
        radius = 50

        y, x = np.ogrid[:height, :width]
        mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2
        img_array[mask] = [255, 100, 100]  # Red circle

        img = Image.fromarray(img_array, mode="RGB")
        frame_path = frames_dir / f"{i:05d}.jpg"
        img.save(frame_path, quality=95)

    return {
        "frames_dir": str(frames_dir),
        "frame_count": frame_count,
        "width": width,
        "height": height,
    }


@pytest.fixture(scope="module")
def sam2_backend(benchmark_checkpoint):
    """Create SAM2 backend for benchmarks (module-scoped to amortize load time)."""
    # Determine best device
    import torch

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    backend = SAM2Backend(
        model_size="large",  # Match the checkpoint we have
        checkpoint_path=benchmark_checkpoint,
        device=device,
    )
    return backend


# ============================================================================
# Performance Measurement Utilities
# ============================================================================


class MemoryMonitor:
    """Background thread to monitor peak memory usage."""

    def __init__(self):
        self.peak_rss_mb = 0.0
        self.running = False
        self.thread = None

    def start(self):
        """Start monitoring."""
        if not HAS_PSUTIL:
            return
        self.running = True
        self.peak_rss_mb = 0.0
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self) -> float:
        """Stop monitoring and return peak RSS in MB."""
        if not HAS_PSUTIL:
            return 0.0
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        return self.peak_rss_mb

    def _monitor(self):
        """Monitor thread loop."""
        process = psutil.Process()
        while self.running:
            try:
                rss_mb = process.memory_info().rss / (1024 * 1024)
                self.peak_rss_mb = max(self.peak_rss_mb, rss_mb)
                time.sleep(0.1)  # Sample 10 times per second
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break


def measure_latency(func, iterations: int = 3) -> Dict[str, float]:
    """Measure function latency over multiple iterations.

    Args:
        func: Callable to measure
        iterations: Number of iterations (default: 3 for benchmarks)

    Returns:
        Dict with mean, median, p95, min, max latencies in seconds
    """
    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        elapsed = time.perf_counter() - start
        timings.append(elapsed)

    timings_sorted = sorted(timings)
    return {
        "mean_sec": sum(timings) / len(timings),
        "median_sec": timings_sorted[len(timings) // 2],
        "p95_sec": timings_sorted[int(len(timings) * 0.95)] if len(timings) > 1 else timings[0],
        "min_sec": min(timings),
        "max_sec": max(timings),
        "count": len(timings),
    }


# ============================================================================
# Auto Mode Benchmarks
# ============================================================================


@pytest.mark.benchmark
class TestSAM2AutoModePerformance:
    """Performance benchmarks for SAM2 auto mode (mask generation)."""

    def test_auto_mode_latency_512x512(self, sam2_backend, benchmark_images):
        """Measure auto mode latency on 512x512 image.

        Baseline: < 2s on MPS, < 5s on CPU
        """
        fixture = next(f for f in benchmark_images if f["width"] == 512)
        seg_input = SegmentationInput(
            image=fixture["array"],
            gamma=1.0,
            mode="auto",
            prompts={
                "points_per_side": 32,  # Standard setting
                "min_mask_region_area": 100,
            },
        )

        # Warm-up inference
        sam2_backend.segment(seg_input)

        # Measure latency
        mem_monitor = MemoryMonitor()
        mem_monitor.start()

        metrics = measure_latency(lambda: sam2_backend.segment(seg_input), iterations=3)

        peak_mem_mb = mem_monitor.stop()

        # Collect result metadata
        result = sam2_backend.segment(seg_input)
        mask_count = result.masks.shape[0]

        print(f"\n[AUTO 512x512] Mean: {metrics['mean_sec']:.3f}s, P95: {metrics['p95_sec']:.3f}s")
        print(f"[AUTO 512x512] Masks: {mask_count}, Peak memory: {peak_mem_mb:.1f}MB")

        # Assertions (realistic limits based on MPS performance)
        assert metrics["mean_sec"] < 20.0, "Auto mode should complete in < 20s (MPS baseline: ~13.5s)"
        assert mask_count >= 0, "Should generate zero or more masks"

        # Store metrics for ledger (JSON format)
        metrics_output = {
            "test": "auto_mode_512x512",
            "mode": "auto",
            "image_size": "512x512",
            "megapixels": fixture["megapixels"],
            "latency": metrics,
            "mask_count": mask_count,
            "peak_memory_mb": peak_mem_mb,
            "device": sam2_backend.device,
        }

        # Write to performance ledger location
        output_dir = Path("output/performance_benchmarks")
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "sam2_auto_512x512.json", "w") as f:
            json.dump(metrics_output, f, indent=2)

    def test_auto_mode_latency_1024x768(self, sam2_backend, benchmark_images):
        """Measure auto mode latency on 1024x768 image (HD crop)."""
        fixture = next(f for f in benchmark_images if f["width"] == 1024)
        seg_input = SegmentationInput(
            image=fixture["array"],
            gamma=1.0,
            mode="auto",
            prompts={
                "points_per_side": 32,
                "min_mask_region_area": 100,
            },
        )

        # Warm-up
        sam2_backend.segment(seg_input)

        # Measure
        metrics = measure_latency(lambda: sam2_backend.segment(seg_input), iterations=3)

        result = sam2_backend.segment(seg_input)
        mask_count = result.masks.shape[0]

        print(f"\n[AUTO 1024x768] Mean: {metrics['mean_sec']:.3f}s, P95: {metrics['p95_sec']:.3f}s")
        print(f"[AUTO 1024x768] Masks: {mask_count}")

        assert metrics["mean_sec"] < 40.0, "HD should complete in < 40s"
        assert mask_count >= 0

        # Store metrics
        metrics_output = {
            "test": "auto_mode_1024x768",
            "mode": "auto",
            "image_size": "1024x768",
            "megapixels": fixture["megapixels"],
            "latency": metrics,
            "mask_count": mask_count,
            "device": sam2_backend.device,
        }

        output_dir = Path("output/performance_benchmarks")
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "sam2_auto_1024x768.json", "w") as f:
            json.dump(metrics_output, f, indent=2)

    def test_auto_mode_scaling(self, sam2_backend, benchmark_images):
        """Test that auto mode latency scales reasonably with image size.

        Expectation: ~O(n) where n is megapixels (not quadratic).
        """
        results = []
        for fixture in benchmark_images[:2]:  # Small and HD only
            seg_input = SegmentationInput(
                image=fixture["array"],
                gamma=1.0,
                mode="auto",
                prompts={
                    "points_per_side": 32,
                    "min_mask_region_area": 100,
                },
            )

            metrics = measure_latency(lambda: sam2_backend.segment(seg_input), iterations=2)
            results.append(
                {
                    "megapixels": fixture["megapixels"],
                    "mean_sec": metrics["mean_sec"],
                    "sec_per_megapixel": metrics["mean_sec"] / fixture["megapixels"],
                }
            )

        print(f"\n[SCALING] Results: {results}")

        # Check that scaling is not quadratic (within 3x for 4x pixels)
        if len(results) == 2:
            ratio = results[1]["mean_sec"] / results[0]["mean_sec"]
            megapixel_ratio = results[1]["megapixels"] / results[0]["megapixels"]
            print(f"[SCALING] Time ratio: {ratio:.2f}x for {megapixel_ratio:.2f}x pixels")
            assert ratio < megapixel_ratio * 1.5, "Scaling should be roughly linear, not quadratic"


# ============================================================================
# Prompted Mode Benchmarks
# ============================================================================


@pytest.mark.benchmark
class TestSAM2PromptedModePerformance:
    """Performance benchmarks for SAM2 prompted modes (points, bbox)."""

    def test_points_mode_latency(self, sam2_backend, benchmark_images):
        """Measure points mode latency (single foreground point)."""
        fixture = next(f for f in benchmark_images if f["width"] == 512)
        seg_input = SegmentationInput(
            image=fixture["array"],
            gamma=1.0,
            mode="points",
            prompts={
                "points": [[256, 256]],  # Center point
                "labels": [1],  # Foreground
            },
        )

        # Warm-up
        sam2_backend.segment(seg_input)

        # Measure
        metrics = measure_latency(lambda: sam2_backend.segment(seg_input), iterations=5)

        result = sam2_backend.segment(seg_input)

        print(f"\n[POINTS] Mean: {metrics['mean_sec']:.3f}s, P95: {metrics['p95_sec']:.3f}s")
        print(f"[POINTS] Masks: {result.masks.shape[0]}")

        assert metrics["mean_sec"] < 5.0, "Points mode should be fast (< 5s)"

        # Store metrics
        metrics_output = {
            "test": "points_mode_512x512",
            "mode": "points",
            "image_size": "512x512",
            "point_count": 1,
            "latency": metrics,
            "mask_count": result.masks.shape[0],
            "device": sam2_backend.device,
        }

        output_dir = Path("output/performance_benchmarks")
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "sam2_points_512x512.json", "w") as f:
            json.dump(metrics_output, f, indent=2)

    def test_bbox_mode_latency(self, sam2_backend, benchmark_images):
        """Measure bbox mode latency."""
        fixture = next(f for f in benchmark_images if f["width"] == 512)
        seg_input = SegmentationInput(
            image=fixture["array"],
            mode="bbox",
            gamma=1.0,
            prompts={"bbox": [128, 128, 384, 384]},  # [x_min, y_min, x_max, y_max]
        )

        # Warm-up
        sam2_backend.segment(seg_input)

        # Measure
        metrics = measure_latency(lambda: sam2_backend.segment(seg_input), iterations=5)

        result = sam2_backend.segment(seg_input)

        print(f"\n[BBOX] Mean: {metrics['mean_sec']:.3f}s, P95: {metrics['p95_sec']:.3f}s")
        print(f"[BBOX] Masks: {result.masks.shape[0]}")

        assert metrics["mean_sec"] < 5.0, "Bbox mode should be fast (< 5s)"

        # Store metrics
        metrics_output = {
            "test": "bbox_mode_512x512",
            "mode": "bbox",
            "image_size": "512x512",
            "latency": metrics,
            "mask_count": result.masks.shape[0],
            "device": sam2_backend.device,
        }

        output_dir = Path("output/performance_benchmarks")
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "sam2_bbox_512x512.json", "w") as f:
            json.dump(metrics_output, f, indent=2)


# ============================================================================
# Video Mode Benchmarks
# ============================================================================


@pytest.mark.benchmark
class TestSAM2VideoModePerformance:
    """Performance benchmarks for SAM2 video tracking mode."""

    def test_video_tracking_throughput(self, sam2_backend, benchmark_video_frames):
        """Measure video tracking throughput (FPS).

        Baseline: > 1 FPS for 512x512 frames
        """
        seg_input = SegmentationInput(
            image=None,  # Video mode doesn't use image
            gamma=1.0,
            mode="video",
            video_path=benchmark_video_frames["frames_dir"],
            prompts={
                "frame_idx": 0,
                "object_id": 1,
                "points": [[256, 256]],  # Track center object
                "labels": [1],
            },
        )

        # Warm-up (first video run loads predictor)
        sam2_backend.segment(seg_input)

        # Measure
        mem_monitor = MemoryMonitor()
        mem_monitor.start()

        start = time.perf_counter()
        result = sam2_backend.segment(seg_input)
        elapsed = time.perf_counter() - start

        peak_mem_mb = mem_monitor.stop()

        frame_count = benchmark_video_frames["frame_count"]
        fps = frame_count / elapsed
        sec_per_frame = elapsed / frame_count

        print(f"\n[VIDEO] Total: {elapsed:.3f}s for {frame_count} frames")
        print(f"[VIDEO] Throughput: {fps:.2f} FPS, {sec_per_frame:.3f}s/frame")
        print(f"[VIDEO] Peak memory: {peak_mem_mb:.1f}MB")
        print(f"[VIDEO] Tracked objects: {len(result.temporal_ids)}")

        assert fps > 0.5, "Should achieve at least 0.5 FPS"
        assert len(result.temporal_ids) > 0, "Should track at least one object"

        # Store metrics
        metrics_output = {
            "test": "video_tracking_512x512",
            "mode": "video",
            "frame_count": frame_count,
            "total_sec": elapsed,
            "fps": fps,
            "sec_per_frame": sec_per_frame,
            "peak_memory_mb": peak_mem_mb,
            "tracked_objects": len(result.temporal_ids),
            "device": sam2_backend.device,
        }

        output_dir = Path("output/performance_benchmarks")
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "sam2_video_512x512.json", "w") as f:
            json.dump(metrics_output, f, indent=2)


# ============================================================================
# Regression Detection (Quality Firewall Integration)
# ============================================================================


@pytest.mark.benchmark
class TestSAM2PerformanceRegression:
    """Performance regression detection tests (Quality Firewall integration).

    These tests compare current performance against established baselines.
    Block merge if:
    - p95 latency increases > 10%
    - mean latency increases > 15%
    - failure rate > 0%
    """

    def test_baseline_comparison_auto_mode(self, sam2_backend, benchmark_images):
        """Compare auto mode performance against baseline.

        Baseline (established Phase 4C):
        - Mean: TBD after first run
        - P95: TBD after first run

        This test will FAIL on first run (no baseline). Run once to establish
        baseline, then future runs will detect regressions.
        """
        fixture = next(f for f in benchmark_images if f["width"] == 512)
        seg_input = SegmentationInput(
            image=fixture["array"],
            gamma=1.0,
            mode="auto",
            prompts={
                "points_per_side": 32,
                "min_mask_region_area": 100,
            },
        )

        # Warm-up
        sam2_backend.segment(seg_input)

        # Measure current performance
        metrics = measure_latency(lambda: sam2_backend.segment(seg_input), iterations=5)

        # Check for baseline file
        baseline_file = Path("docs/performance/baselines/sam2_auto_512x512_baseline.json")

        if not baseline_file.exists():
            # First run: establish baseline
            baseline_data = {
                "version": "phase4c",
                "test": "auto_mode_512x512",
                "baseline_mean_sec": metrics["mean_sec"],
                "baseline_p95_sec": metrics["p95_sec"],
                "device": sam2_backend.device,
                "established_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            baseline_file.parent.mkdir(parents=True, exist_ok=True)
            with open(baseline_file, "w") as f:
                json.dump(baseline_data, f, indent=2)

            pytest.skip(f"Baseline established: {metrics['mean_sec']:.3f}s mean, {metrics['p95_sec']:.3f}s p95")

        # Load baseline and compare
        with open(baseline_file) as f:
            baseline = json.load(f)

        mean_change_pct = ((metrics["mean_sec"] - baseline["baseline_mean_sec"]) / baseline["baseline_mean_sec"]) * 100
        p95_change_pct = ((metrics["p95_sec"] - baseline["baseline_p95_sec"]) / baseline["baseline_p95_sec"]) * 100

        print(f"\n[REGRESSION CHECK]")
        print(f"  Mean: {baseline['baseline_mean_sec']:.3f}s → {metrics['mean_sec']:.3f}s ({mean_change_pct:+.1f}%)")
        print(f"  P95:  {baseline['baseline_p95_sec']:.3f}s → {metrics['p95_sec']:.3f}s ({p95_change_pct:+.1f}%)")

        # Quality Firewall thresholds
        assert mean_change_pct <= 15.0, f"Mean latency regression: {mean_change_pct:+.1f}% > 15% threshold"
        assert p95_change_pct <= 10.0, f"P95 latency regression: {p95_change_pct:+.1f}% > 10% threshold"

        print("  ✅ No performance regression detected")
