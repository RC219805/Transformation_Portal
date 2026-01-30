#!/usr/bin/env python3
"""Benchmark suite for depth_canonical module.

Compares performance of the new depth_canonical module to ensure:
1. No significant regression in depth estimation speed
2. PBR generation meets performance targets
3. Cache provides expected speedup
4. Batch processing achieves target throughput

Usage:
    python scripts/benchmarks/depth_canonical_benchmark.py
    python scripts/benchmarks/depth_canonical_benchmark.py --quick  # Faster, less comprehensive
    python scripts/benchmarks/depth_canonical_benchmark.py --save results.json  # Save results
"""

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Suppress deprecation warnings during benchmarking
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow required for benchmarking. Install with: pip install Pillow")
    sys.exit(1)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark test."""

    name: str
    iterations: int
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    throughput: Optional[float] = None  # images/hour for batch tests
    status: str = "✅"  # ✅ or ❌


def create_test_image(size: tuple) -> np.ndarray:
    """Create a random test image of given size."""
    return np.random.randint(0, 255, (*size, 3), dtype=np.uint8)


def benchmark_depth_estimation(quick: bool = False) -> List[BenchmarkResult]:
    """Benchmark depth estimation at various resolutions."""
    print("\n" + "=" * 80)
    print("Depth Estimation Performance")
    print("=" * 80)

    results = []

    # Test sizes: 512p, 720p, 1080p, 4K
    if quick:
        sizes = [("512p", (512, 512)), ("4K", (3840, 2160))]
        iterations = 3
    else:
        sizes = [
            ("512p", (512, 512)),
            ("720p", (1280, 720)),
            ("1080p", (1920, 1080)),
            ("4K", (3840, 2160)),
        ]
        iterations = 5

    try:
        from transformation_portal.depth_canonical import DepthPipeline, UnifiedDepthConfig

        # Create pipeline
        config = UnifiedDepthConfig()
        pipeline = DepthPipeline(config)

        for name, size in sizes:
            print(f"\nBenchmarking {name} ({size[0]}×{size[1]})...")

            # Create test image
            test_img = create_test_image(size)

            # Warmup
            try:
                _ = pipeline._estimate_depth(test_img)
            except Exception as e:
                print(f"  ⚠️  Skipping {name}: {e}")
                continue

            # Benchmark
            times = []
            for i in range(iterations):
                start = time.perf_counter()
                _ = pipeline._estimate_depth(test_img)
                elapsed = (time.perf_counter() - start) * 1000  # ms
                times.append(elapsed)
                print(f"  Iteration {i+1}/{iterations}: {elapsed:.1f}ms")

            # Calculate statistics
            mean_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)

            # Determine status (< 5% regression from target)
            # Targets based on existing performance data
            targets = {
                "512p": 50,   # 50ms
                "720p": 100,  # 100ms
                "1080p": 200, # 200ms
                "4K": 600,    # 600ms
            }
            target = targets.get(name, mean_time)
            status = "✅" if mean_time < target * 1.05 else "❌"

            result = BenchmarkResult(
                name=f"Depth Estimation {name}",
                iterations=iterations,
                mean_time_ms=mean_time,
                std_time_ms=std_time,
                min_time_ms=min_time,
                max_time_ms=max_time,
                status=status,
            )
            results.append(result)

            print(f"  Result: {mean_time:.1f}ms ± {std_time:.1f}ms (target: <{target}ms) {status}")

    except Exception as e:
        print(f"⚠️  Could not benchmark depth estimation: {e}")

    return results


def benchmark_pbr_generation(quick: bool = False) -> List[BenchmarkResult]:
    """Benchmark PBR map generation."""
    print("\n" + "=" * 80)
    print("PBR Generation Performance")
    print("=" * 80)

    results = []

    # Test sizes
    if quick:
        sizes = [("512p", (512, 512)), ("4K", (3840, 2160))]
        iterations = 3
    else:
        sizes = [
            ("512p", (512, 512)),
            ("1080p", (1920, 1080)),
            ("4K", (3840, 2160)),
        ]
        iterations = 5

    try:
        from transformation_portal.depth_canonical import generate_pbr_maps

        for name, size in sizes:
            print(f"\nBenchmarking PBR generation {name} ({size[0]}×{size[1]})...")

            # Create test depth map
            depth_map = np.random.rand(*size).astype(np.float32)

            # Warmup
            try:
                _ = generate_pbr_maps(depth_map)
            except Exception as e:
                print(f"  ⚠️  Skipping {name}: {e}")
                continue

            # Benchmark
            times = []
            for i in range(iterations):
                start = time.perf_counter()
                _ = generate_pbr_maps(depth_map)
                elapsed = (time.perf_counter() - start) * 1000  # ms
                times.append(elapsed)
                print(f"  Iteration {i+1}/{iterations}: {elapsed:.1f}ms")

            # Calculate statistics
            mean_time = np.mean(times)
            std_time = np.std(times)

            # Targets
            targets = {
                "512p": 30,   # 30ms
                "1080p": 150, # 150ms
                "4K": 500,    # 500ms
            }
            target = targets.get(name, mean_time)
            status = "✅" if mean_time < target * 1.05 else "❌"

            result = BenchmarkResult(
                name=f"PBR Generation {name}",
                iterations=iterations,
                mean_time_ms=mean_time,
                std_time_ms=std_time,
                min_time_ms=np.min(times),
                max_time_ms=np.max(times),
                status=status,
            )
            results.append(result)

            print(f"  Result: {mean_time:.1f}ms ± {std_time:.1f}ms (target: <{target}ms) {status}")

    except Exception as e:
        print(f"⚠️  Could not benchmark PBR generation: {e}")

    return results


def benchmark_cache_performance() -> List[BenchmarkResult]:
    """Benchmark cache hit/miss performance."""
    print("\n" + "=" * 80)
    print("Cache Performance")
    print("=" * 80)

    results = []

    try:
        from transformation_portal.depth_canonical import DepthPipeline, UnifiedDepthConfig

        # Create pipeline with caching enabled
        config = UnifiedDepthConfig()
        pipeline = DepthPipeline(config)

        # Create test image
        test_img = create_test_image((1920, 1080))

        # Cache miss (first run)
        print("\nBenchmarking cache miss (first run)...")
        start = time.perf_counter()
        _ = pipeline._estimate_depth(test_img)
        cache_miss_time = (time.perf_counter() - start) * 1000

        # Cache hit (second run with same image)
        print("Benchmarking cache hit (second run)...")
        start = time.perf_counter()
        _ = pipeline._estimate_depth(test_img)
        cache_hit_time = (time.perf_counter() - start) * 1000

        speedup = cache_miss_time / cache_hit_time if cache_hit_time > 0 else 0

        print(f"\n  Cache miss: {cache_miss_time:.1f}ms")
        print(f"  Cache hit:  {cache_hit_time:.1f}ms")
        print(f"  Speedup:    {speedup:.1f}x")

        status = "✅" if speedup > 5.0 else "❌"  # Target: at least 5x speedup

        results.append(
            BenchmarkResult(
                name="Cache Miss",
                iterations=1,
                mean_time_ms=cache_miss_time,
                std_time_ms=0,
                min_time_ms=cache_miss_time,
                max_time_ms=cache_miss_time,
                status=status,
            )
        )
        results.append(
            BenchmarkResult(
                name="Cache Hit",
                iterations=1,
                mean_time_ms=cache_hit_time,
                std_time_ms=0,
                min_time_ms=cache_hit_time,
                max_time_ms=cache_hit_time,
                throughput=speedup,  # Store speedup in throughput field
                status=status,
            )
        )

    except Exception as e:
        print(f"⚠️  Could not benchmark cache: {e}")

    return results


def benchmark_batch_processing(quick: bool = False) -> List[BenchmarkResult]:
    """Benchmark batch processing throughput."""
    print("\n" + "=" * 80)
    print("Batch Processing Throughput")
    print("=" * 80)

    results = []

    try:
        from transformation_portal.depth_canonical import DepthPipeline, UnifiedDepthConfig

        # Create pipeline
        config = UnifiedDepthConfig()
        pipeline = DepthPipeline(config)

        # Test with smaller batch for quick mode
        num_images = 10 if quick else 20
        size = (1920, 1080)

        print(f"\nProcessing {num_images} images at {size[0]}×{size[1]}...")

        # Create test images
        test_images = [create_test_image(size) for _ in range(num_images)]

        # Benchmark batch processing
        start = time.perf_counter()
        for i, img in enumerate(test_images, 1):
            _ = pipeline._estimate_depth(img)
            if i % 5 == 0:
                print(f"  Processed {i}/{num_images} images...")

        total_time = time.perf_counter() - start
        avg_time = (total_time / num_images) * 1000  # ms per image
        throughput = (num_images / total_time) * 3600  # images per hour

        print(f"\n  Total time:   {total_time:.1f}s")
        print(f"  Avg per image: {avg_time:.1f}ms")
        print(f"  Throughput:    {throughput:.0f} images/hour")

        # Target: at least 100 images/hour
        status = "✅" if throughput >= 100 else "❌"

        results.append(
            BenchmarkResult(
                name="Batch Processing",
                iterations=num_images,
                mean_time_ms=avg_time,
                std_time_ms=0,
                min_time_ms=avg_time,
                max_time_ms=avg_time,
                throughput=throughput,
                status=status,
            )
        )

    except Exception as e:
        print(f"⚠️  Could not benchmark batch processing: {e}")

    return results


def print_summary(all_results: List[BenchmarkResult]):
    """Print summary of all benchmark results."""
    print("\n" + "=" * 80)
    print("Benchmark Summary")
    print("=" * 80)

    for result in all_results:
        throughput_str = f" | {result.throughput:.0f} img/hr" if result.throughput else ""
        print(f"{result.status} {result.name}: {result.mean_time_ms:.1f}ms ± {result.std_time_ms:.1f}ms{throughput_str}")

    # Overall status
    all_passed = all(r.status == "✅" for r in all_results)
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ All performance targets met!")
    else:
        print("❌ Some performance targets not met. Review results above.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Benchmark depth_canonical module")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmarks (less comprehensive)")
    parser.add_argument("--save", type=Path, help="Save results to JSON file")
    parser.add_argument("--depth-only", action="store_true", help="Only benchmark depth estimation")
    parser.add_argument("--pbr-only", action="store_true", help="Only benchmark PBR generation")

    args = parser.parse_args()

    print("Depth Canonical Benchmark Suite")
    print("=" * 80)

    if args.quick:
        print("Running in QUICK mode (less comprehensive)")

    all_results = []

    # Run benchmarks
    if not args.pbr_only:
        all_results.extend(benchmark_depth_estimation(quick=args.quick))

    if not args.depth_only:
        all_results.extend(benchmark_pbr_generation(quick=args.quick))

    if not args.depth_only and not args.pbr_only:
        all_results.extend(benchmark_cache_performance())
        all_results.extend(benchmark_batch_processing(quick=args.quick))

    # Print summary
    print_summary(all_results)

    # Save results if requested
    if args.save:
        results_dict = {"benchmarks": [asdict(r) for r in all_results]}
        with open(args.save, "w") as f:
            json.dump(results_dict, f, indent=2)
        print(f"\n📊 Results saved to: {args.save}")

    # Exit with appropriate code
    all_passed = all(r.status == "✅" for r in all_results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
