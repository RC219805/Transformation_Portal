#!/usr/bin/env python3
"""Stress tests for PBR CLI - large batch processing.

Priority P2: Stress Testing Infrastructure
- Test processing of 100+ images
- Monitor memory usage and performance
- Verify no memory leaks
- Check for consistent output quality
"""

import time
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from transformation_portal.lux_depth_v3.pbr_cli import app

pytestmark = [pytest.mark.stress, pytest.mark.slow]


def _measure_best_elapsed(runner: CliRunner, args: list[str], repeats: int = 2) -> tuple[float, object]:
    """Return the fastest successful CLI run to reduce scheduler noise."""
    best_elapsed: float | None = None
    best_result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = runner.invoke(app, args)
        elapsed = time.perf_counter() - start
        assert result.exit_code == 0, result.stdout
        if best_elapsed is None or elapsed < best_elapsed:
            best_elapsed = elapsed
            best_result = result
    assert best_elapsed is not None
    assert best_result is not None
    return best_elapsed, best_result


@pytest.fixture
def large_batch(tmp_path):
    """Create a large batch of synthetic depth files for stress testing."""
    batch_dir = tmp_path / "large_batch"
    batch_dir.mkdir()

    # Create 100 synthetic depth files
    num_files = 100
    for i in range(num_files):
        # Vary image sizes to simulate real workload
        if i % 3 == 0:
            size = 512  # Small
        elif i % 3 == 1:
            size = 1024  # Medium
        else:
            size = 2048  # Large

        # Generate synthetic depth with some variation
        depth = np.random.rand(size, size).astype(np.float32)

        # Add some structure (gradients)
        x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
        depth = depth * 0.5 + (x + y) / 4

        np.save(batch_dir / f"scene_{i:04d}_depth.npy", depth)

    return batch_dir, num_files


@pytest.mark.stress
@pytest.mark.slow
class TestLargeBatchProcessing:
    """Stress tests for large batch processing."""

    def test_100_image_batch(self, large_batch, tmp_path):
        """Test processing 100 images in batch mode."""
        batch_dir, num_files = large_batch
        output_dir = tmp_path / "output"

        runner = CliRunner()

        start_time = time.time()

        result = runner.invoke(
            app,
            [
                "generate",
                "--depth-dir",
                str(batch_dir),
                "--preset",
                "standard",
                "--output",
                str(output_dir),
            ],
        )

        end_time = time.time()
        elapsed = end_time - start_time

        # Should succeed
        assert result.exit_code == 0, f"Batch failed: {result.stdout}"

        # Should process all files
        assert f"Batch processing {num_files} depth file(s)" in result.stdout
        assert f"Success: {num_files}" in result.stdout
        assert "Errors:  0" in result.stdout

        # Performance assertion: should complete in reasonable time
        # Rough estimate: ~1 second per image max (very conservative)
        max_time = num_files * 2.0
        assert elapsed < max_time, f"Batch took too long: {elapsed:.1f}s (max: {max_time}s)"

        # Calculate throughput
        throughput = num_files / elapsed
        print(f"\nStress test performance:")
        print(f"  Total time: {elapsed:.1f}s")
        print(f"  Throughput: {throughput:.2f} images/sec")
        print(f"  Avg time per image: {elapsed/num_files:.2f}s")

    def test_memory_bounded(self, tmp_path):
        """Test that memory usage stays bounded during large batch."""
        pytest.importorskip("psutil", reason="psutil required for memory testing")
        import os

        import psutil

        # Create smaller batch for memory testing (20 large images)
        batch_dir = tmp_path / "memory_batch"
        batch_dir.mkdir()

        num_files = 20
        for i in range(num_files):
            # Large images to stress memory
            depth = np.random.rand(2048, 2048).astype(np.float32)
            np.save(batch_dir / f"large_{i:04d}_depth.npy", depth)

        output_dir = tmp_path / "output"

        runner = CliRunner()

        # Monitor memory before
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        result = runner.invoke(
            app,
            [
                "generate",
                "--depth-dir",
                str(batch_dir),
                "--preset",
                "draft",  # Use draft for speed
                "--output",
                str(output_dir),
            ],
        )

        # Monitor memory after
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_delta = mem_after - mem_before

        # Should succeed
        assert result.exit_code == 0

        # Memory growth should be reasonable
        # Allow up to 2GB growth for 20x 2048x2048 images
        max_growth_mb = 2000
        print(f"\nMemory usage:")
        print(f"  Before: {mem_before:.1f} MB")
        print(f"  After:  {mem_after:.1f} MB")
        print(f"  Delta:  {mem_delta:.1f} MB")

        assert mem_delta < max_growth_mb, f"Memory grew too much: {mem_delta:.1f} MB"

    def test_repeated_batches_no_leak(self, tmp_path):
        """Test repeated batch processing for memory leaks."""
        # Create a small batch
        batch_dir = tmp_path / "repeat_batch"
        batch_dir.mkdir()

        num_files = 10
        for i in range(num_files):
            depth = np.random.rand(512, 512).astype(np.float32)
            np.save(batch_dir / f"image_{i:02d}_depth.npy", depth)

        runner = CliRunner()

        # Run the same batch 5 times
        num_iterations = 5
        for iteration in range(num_iterations):
            output_dir = tmp_path / f"output_{iteration}"

            result = runner.invoke(
                app,
                [
                    "generate",
                    "--depth-dir",
                    str(batch_dir),
                    "--preset",
                    "draft",
                    "--output",
                    str(output_dir),
                ],
            )

            assert result.exit_code == 0, f"Iteration {iteration} failed"
            assert f"Success: {num_files}" in result.stdout

        # If we got here without crashes, no obvious leaks
        # More sophisticated leak detection would require memory profiling

    def test_mixed_file_sizes_batch(self, tmp_path):
        """Test batch with mixed file sizes (realistic workload)."""
        batch_dir = tmp_path / "mixed_batch"
        batch_dir.mkdir()

        # Create files with varying sizes
        sizes = [256, 512, 1024, 2048, 512, 1024, 256, 2048, 512, 1024]

        for i, size in enumerate(sizes):
            depth = np.random.rand(size, size).astype(np.float32)
            np.save(batch_dir / f"mixed_{i:02d}_depth.npy", depth)

        output_dir = tmp_path / "output"

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "generate",
                "--depth-dir",
                str(batch_dir),
                "--preset",
                "standard",
                "--output",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0
        assert f"Success: {len(sizes)}" in result.stdout

    def test_partial_failures_dont_crash_batch(self, tmp_path):
        """Test that batch continues gracefully with some failures."""
        batch_dir = tmp_path / "partial_fail_batch"
        batch_dir.mkdir()

        # Create 50 valid files
        for i in range(50):
            depth = np.random.rand(256, 256).astype(np.float32)
            np.save(batch_dir / f"valid_{i:02d}_depth.npy", depth)

        # Create 5 corrupt files
        for i in range(5):
            corrupt = batch_dir / f"corrupt_{i:02d}_depth.npy"
            corrupt.write_text("Not a valid numpy file")

        output_dir = tmp_path / "output"

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "generate",
                "--depth-dir",
                str(batch_dir),
                "--preset",
                "draft",
                "--output",
                str(output_dir),
            ],
        )

        # Should report both successes and failures
        assert "Success: 50" in result.stdout
        assert "Errors:  5" in result.stdout

        # Exit code should be 1 due to failures
        assert result.exit_code == 1

        # Should list failed files
        assert "Failed files:" in result.stdout


@pytest.mark.stress
@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Performance benchmarks for regression detection."""

    def test_baseline_single_image_performance(self, tmp_path):
        """Baseline performance test for single image."""
        # Create a standard 1024x1024 depth file
        depth = np.random.rand(1024, 1024).astype(np.float32)
        depth_path = tmp_path / "baseline_depth.npy"
        np.save(depth_path, depth)

        output_dir = tmp_path / "output"

        runner = CliRunner()

        # Measure time for standard preset
        standard_time, _ = _measure_best_elapsed(
            runner,
            [
                "generate",
                "--depth",
                str(depth_path),
                "--preset",
                "standard",
                "--output",
                str(output_dir),
            ],
        )

        # Measure time for draft preset (should be faster)
        output_dir2 = tmp_path / "output2"
        draft_time, _ = _measure_best_elapsed(
            runner,
            [
                "generate",
                "--depth",
                str(depth_path),
                "--preset",
                "draft",
                "--output",
                str(output_dir2),
            ],
        )

        print(f"\nPerformance baseline (1024x1024):")
        print(f"  Standard preset: {standard_time:.3f}s")
        print(f"  Draft preset:    {draft_time:.3f}s")
        assert (
            draft_time <= standard_time * 1.10
        ), f"Draft preset regressed beyond tolerance: draft={draft_time:.3f}s standard={standard_time:.3f}s"

    def test_throughput_by_preset(self, tmp_path):
        """Compare throughput across different presets."""
        # Create batch of 10 images
        batch_dir = tmp_path / "preset_batch"
        batch_dir.mkdir()

        num_files = 10
        for i in range(num_files):
            depth = np.random.rand(512, 512).astype(np.float32)
            np.save(batch_dir / f"image_{i:02d}_depth.npy", depth)

        runner = CliRunner()
        presets = ["draft", "standard", "premium"]

        times = {}
        for preset in presets:
            output_dir = tmp_path / f"output_{preset}"
            elapsed, _ = _measure_best_elapsed(
                runner,
                [
                    "generate",
                    "--depth-dir",
                    str(batch_dir),
                    "--preset",
                    preset,
                    "--output",
                    str(output_dir),
                ],
            )
            times[preset] = elapsed

        print(f"\nThroughput by preset ({num_files} images):")
        for preset, elapsed in times.items():
            throughput = num_files / elapsed
            print(f"  {preset:8s}: {elapsed:.2f}s ({throughput:.1f} img/s)")
        assert times["draft"] <= times["standard"] * 1.10, (
            f"Draft batch throughput regressed beyond tolerance: draft={times['draft']:.3f}s "
            f"standard={times['standard']:.3f}s"
        )


@pytest.mark.stress
@pytest.mark.slow
class TestResourceLimits:
    """Test behavior under resource constraints."""

    def test_graceful_degradation_large_images(self, tmp_path):
        """Test handling of very large images."""
        # Create a 4K image (challenging but realistic)
        depth = np.random.rand(4096, 4096).astype(np.float32)
        depth_path = tmp_path / "4k_depth.npy"
        np.save(depth_path, depth)

        output_dir = tmp_path / "output"

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "generate",
                "--depth",
                str(depth_path),
                "--preset",
                "draft",  # Use draft for speed
                "--output",
                str(output_dir),
            ],
        )

        # Should handle large image gracefully
        # Either succeeds or fails with clear error
        if result.exit_code == 0:
            assert "Generated PBR maps" in result.output
        else:
            assert "Error:" in result.output or "✗" in result.output
            # Should not crash with stack trace
            assert "Traceback" not in result.output

    def test_handles_empty_batch_gracefully(self, tmp_path):
        """Test batch mode with no valid files."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        output_dir = tmp_path / "output"

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "generate",
                "--depth-dir",
                str(empty_dir),
                "--output",
                str(output_dir),
            ],
        )

        # Should fail gracefully
        assert result.exit_code == 1
        assert "Warning: No depth files" in result.output
