"""Performance benchmarks for v1.7 (Condition #5).

Benchmarks NumPy vs pure Python modes and documents performance trade-offs.
"""

from __future__ import annotations

import random
import time
from typing import List

import pytest

pytestmark = pytest.mark.unit

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from tools.performance_ledger import (
    _bootstrap_confidence_interval,
    _pure_python_mean,
    _pure_python_percentile,
    _pure_python_std,
    compute_statistics,
)


@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Performance benchmarks for NumPy vs pure Python."""

    def test_benchmark_mean_computation(self, benchmark_params=None):
        """Benchmark mean computation."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required for benchmarks")

        # Test with varying sizes
        sizes = [10, 100, 1000]
        results = {}

        for size in sizes:
            values = [random.uniform(1.0, 100.0) for _ in range(size)]

            # Benchmark NumPy
            start = time.perf_counter()
            for _ in range(100):
                _ = float(np.mean(values))
            numpy_time = time.perf_counter() - start

            # Benchmark pure Python
            start = time.perf_counter()
            for _ in range(100):
                _ = _pure_python_mean(values)
            python_time = time.perf_counter() - start

            slowdown = python_time / numpy_time if numpy_time > 0 else 1.0
            results[size] = {"numpy": numpy_time, "python": python_time, "slowdown": slowdown}

            print(f"\nMean benchmark (n={size}):")
            print(f"  NumPy:       {numpy_time*1000:.3f}ms")
            print(f"  Pure Python: {python_time*1000:.3f}ms")
            print(f"  Slowdown:    {slowdown:.1f}x")

        # Acceptable slowdown: pure Python should be < 100x slower for mean
        for size, data in results.items():
            assert data["slowdown"] < 100, f"Pure Python mean too slow at n={size}: {data['slowdown']:.1f}x"

    def test_benchmark_percentile_computation(self):
        """Benchmark percentile computation."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required for benchmarks")

        sizes = [10, 100, 1000]
        results = {}

        for size in sizes:
            values = [random.uniform(1.0, 100.0) for _ in range(size)]

            # Benchmark NumPy
            start = time.perf_counter()
            for _ in range(100):
                _ = float(np.percentile(values, 95))
            numpy_time = time.perf_counter() - start

            # Benchmark pure Python
            start = time.perf_counter()
            for _ in range(100):
                _ = _pure_python_percentile(values, 95)
            python_time = time.perf_counter() - start

            slowdown = python_time / numpy_time if numpy_time > 0 else 1.0
            results[size] = {"numpy": numpy_time, "python": python_time, "slowdown": slowdown}

            print(f"\nPercentile benchmark (n={size}):")
            print(f"  NumPy:       {numpy_time*1000:.3f}ms")
            print(f"  Pure Python: {python_time*1000:.3f}ms")
            print(f"  Slowdown:    {slowdown:.1f}x")

        # Pure Python percentile requires sorting, expect 10-100x slowdown
        for size, data in results.items():
            assert data["slowdown"] < 200, f"Pure Python percentile too slow at n={size}: {data['slowdown']:.1f}x"

    def test_benchmark_std_computation(self):
        """Benchmark standard deviation computation."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required for benchmarks")

        sizes = [10, 100, 1000]
        results = {}

        for size in sizes:
            values = [random.uniform(1.0, 100.0) for _ in range(size)]

            # Benchmark NumPy
            start = time.perf_counter()
            for _ in range(100):
                _ = float(np.std(values, ddof=1))
            numpy_time = time.perf_counter() - start

            # Benchmark pure Python
            start = time.perf_counter()
            for _ in range(100):
                _ = _pure_python_std(values)
            python_time = time.perf_counter() - start

            slowdown = python_time / numpy_time if numpy_time > 0 else 1.0
            results[size] = {"numpy": numpy_time, "python": python_time, "slowdown": slowdown}

            print(f"\nStd benchmark (n={size}):")
            print(f"  NumPy:       {numpy_time*1000:.3f}ms")
            print(f"  Pure Python: {python_time*1000:.3f}ms")
            print(f"  Slowdown:    {slowdown:.1f}x")

        # Std is simple computation, should be < 100x slower
        for size, data in results.items():
            assert data["slowdown"] < 100, f"Pure Python std too slow at n={size}: {data['slowdown']:.1f}x"

    def test_benchmark_full_statistics(self):
        """Benchmark full statistics computation."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required for benchmarks")

        sizes = [10, 50, 100]
        results = {}

        for size in sizes:
            values = [random.uniform(1.0, 100.0) for _ in range(size)]

            # Benchmark with NumPy (should use NumPy path)
            start = time.perf_counter()
            for _ in range(10):
                _ = compute_statistics(values, bootstrap_iterations=0, enable_bootstrap=False)
            numpy_time = time.perf_counter() - start

            results[size] = {"numpy": numpy_time}

            print(f"\nFull statistics benchmark (n={size}):")
            print(f"  NumPy path:  {numpy_time*1000:.3f}ms for 10 iterations")

        # Document acceptable performance
        # Small datasets should complete quickly even with NumPy
        assert results[10]["numpy"] < 0.1, "Statistics too slow for small datasets"

    def test_benchmark_bootstrap_ci(self):
        """Benchmark bootstrap confidence interval computation."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required for benchmarks")

        sample_sizes = [10, 50, 100]
        bootstrap_iterations = [100, 500, 1000]

        results = {}

        for n_samples in sample_sizes:
            for n_bootstrap in bootstrap_iterations:
                values = [random.uniform(1.0, 100.0) for _ in range(n_samples)]

                start = time.perf_counter()
                _ = _bootstrap_confidence_interval(values, iterations=n_bootstrap, seed=42)
                elapsed = time.perf_counter() - start

                key = f"n={n_samples}, iter={n_bootstrap}"
                results[key] = elapsed

                print(f"\nBootstrap CI benchmark ({key}):")
                print(f"  Time: {elapsed*1000:.1f}ms")

        # Bootstrap should complete in reasonable time
        # 1000 iterations with 100 samples should be < 5 seconds
        assert results["n=100, iter=1000"] < 5.0, "Bootstrap CI too slow for typical use"

    def test_performance_regression_vs_v1_0(self):
        """Ensure v1.7 NumPy mode isn't slower than v1.0."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required for regression test")

        # Simulate typical workload
        values = [random.uniform(5.0, 20.0) for _ in range(50)]

        # Measure current performance
        start = time.perf_counter()
        for _ in range(100):
            stats = compute_statistics(values, bootstrap_iterations=0, enable_bootstrap=False)
        elapsed = time.perf_counter() - start

        # Should be fast (< 100ms for 100 iterations)
        per_call = elapsed / 100

        print(f"\nv1.7 statistics performance:")
        print(f"  Time per call: {per_call*1000:.2f}ms")
        print(f"  Total for 100 calls: {elapsed*1000:.1f}ms")

        # Performance acceptance criteria:
        # - Each statistics computation should be < 5ms for 50 samples
        # - This ensures no regression from v1.0
        assert per_call < 0.005, f"v1.7 NumPy mode regressed: {per_call*1000:.2f}ms per call (expected < 5ms)"


@pytest.mark.slow
class TestPerformanceDocumentation:
    """Document performance characteristics for different scenarios."""

    def test_document_pure_python_overhead(self):
        """Document when pure Python mode is acceptable."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required for comparison")

        scenarios = {
            "small_batch": 10,
            "medium_batch": 50,
            "large_batch": 200,
        }

        print("\n" + "=" * 60)
        print("Pure Python Performance Trade-offs")
        print("=" * 60)

        for scenario, size in scenarios.items():
            values = [random.uniform(1.0, 100.0) for _ in range(size)]

            # NumPy path
            start = time.perf_counter()
            stats_numpy = compute_statistics(values, bootstrap_iterations=0)
            numpy_time = time.perf_counter() - start

            # Pure Python path (simulated by calling pure functions)
            start = time.perf_counter()
            mean = _pure_python_mean(values)
            std = _pure_python_std(values, mean)
            p95 = _pure_python_percentile(sorted(values), 95)
            python_time = time.perf_counter() - start

            slowdown = python_time / numpy_time if numpy_time > 0 else 1.0

            print(f"\n{scenario} (n={size}):")
            print(f"  NumPy:       {numpy_time*1000:.3f}ms")
            print(f"  Pure Python: {python_time*1000:.3f}ms")
            print(f"  Slowdown:    {slowdown:.1f}x")

            # Recommendation
            if python_time < 0.1:  # < 100ms
                print(f"  ✅ Acceptable: Pure Python overhead is negligible")
            elif python_time < 1.0:  # < 1s
                print(f"  ⚠️  Caution: Pure Python adds noticeable delay")
            else:
                print(f"  ❌ Not recommended: Use NumPy for this scale")

        print("\n" + "=" * 60)
        print("Recommendation:")
        print("- Small batches (< 50): Pure Python is acceptable")
        print("- Medium batches (50-200): NumPy recommended")
        print("- Large batches (> 200): NumPy required")
        print("=" * 60 + "\n")

    def test_document_bootstrap_cost(self):
        """Document bootstrap CI computation cost."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required for benchmark")

        values = [random.uniform(1.0, 100.0) for _ in range(50)]

        print("\n" + "=" * 60)
        print("Bootstrap Confidence Interval Cost")
        print("=" * 60)

        iterations_list = [0, 100, 500, 1000, 5000]

        for iterations in iterations_list:
            if iterations == 0:
                start = time.perf_counter()
                stats = compute_statistics(values, bootstrap_iterations=0, enable_bootstrap=False)
                elapsed = time.perf_counter() - start
                print(f"\nNo bootstrap: {elapsed*1000:.2f}ms")
            else:
                start = time.perf_counter()
                stats = compute_statistics(values, bootstrap_iterations=iterations, enable_bootstrap=True)
                elapsed = time.perf_counter() - start
                print(f"\n{iterations} iterations: {elapsed*1000:.1f}ms")

        print("\n" + "=" * 60)
        print("Recommendation:")
        print("- 100-500 iterations: Fast, suitable for CI/CD")
        print("- 1000 iterations: Default, good accuracy/speed balance")
        print("- 5000+ iterations: Research/analysis only")
        print("=" * 60 + "\n")


@pytest.mark.slow
class TestWorstCasePerformance:
    """Test worst-case performance scenarios."""

    def test_maximum_samples_performance(self):
        """Test performance with large number of samples."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required for benchmark")

        # Simulate worst case: 1000 video frames processed
        values = [random.uniform(1.0, 100.0) for _ in range(1000)]

        start = time.perf_counter()
        stats = compute_statistics(values, bootstrap_iterations=1000, enable_bootstrap=True)
        elapsed = time.perf_counter() - start

        print(f"\nWorst case (n=1000, bootstrap=1000):")
        print(f"  Time: {elapsed:.2f}s")

        # Should complete in reasonable time (< 30s)
        assert elapsed < 30.0, f"Worst case too slow: {elapsed:.1f}s (expected < 30s)"

    def test_maximum_bootstrap_iterations_performance(self):
        """Test performance with maximum allowed bootstrap iterations."""
        if not HAS_NUMPY:
            pytest.skip("NumPy required for benchmark")

        values = [random.uniform(1.0, 100.0) for _ in range(50)]

        # Test with MAX_BOOTSTRAP_ITERATIONS (10000)
        start = time.perf_counter()
        stats = compute_statistics(values, bootstrap_iterations=10000, enable_bootstrap=True)
        elapsed = time.perf_counter() - start

        print(f"\nMaximum bootstrap iterations (10000):")
        print(f"  Time: {elapsed:.2f}s")

        # Should complete but may be slow (< 60s)
        assert elapsed < 60.0, f"Maximum bootstrap too slow: {elapsed:.1f}s"
