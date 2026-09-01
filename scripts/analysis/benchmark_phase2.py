#!/usr/bin/env python3
"""Benchmark script for Phase 2 parallelization optimizations.

Measures:
1. Sequential vs parallel batch processing throughput
2. Cache hit rates and effectiveness
3. Memory usage during parallel processing
4. Worker scalability (1, 2, 4, 8 workers)

Usage:
    python scripts/benchmark_phase2.py --input-dir <dir> --workers 4
    python scripts/benchmark_phase2.py --synthetic --num-images 100
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from transformation_portal.lux_depth_v3.config import EnhanceConfig
from transformation_portal.lux_depth_v3.execution_lifecycle import prepare_lux_execution
from transformation_portal.lux_depth_v3.input_manager import ImageInput
from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _prepared_orchestrator(
    config: EnhanceConfig,
    output_root: Path,
    input_root: Path,
    image_paths: List[Path],
) -> EnhanceOrchestrator:
    """Build a benchmark executor from one frozen input/runtime plan."""

    prepared = prepare_lux_execution(config, input_root, image_paths)
    return EnhanceOrchestrator.from_prepared(
        prepared,
        output_root,
        verify_outputs=False,
    )


def create_synthetic_images(output_dir: Path, count: int) -> List[Path]:
    """Create synthetic test images for benchmarking.

    Args:
        output_dir: Directory to create images in
        count: Number of images to create

    Returns:
        List of created image paths
    """
    import numpy as np
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    images = []

    logger.info(f"Creating {count} synthetic images...")
    for i in range(count):
        # Create random RGB image (1024x768)
        array = np.random.randint(0, 255, (768, 1024, 3), dtype=np.uint8)
        img = Image.fromarray(array)

        img_path = output_dir / f"synthetic_{i:04d}.jpg"
        img.save(img_path, quality=95)
        images.append(img_path)

    logger.info(f"Created {len(images)} synthetic images")
    return images


def benchmark_sequential(
    orchestrator: EnhanceOrchestrator,
    image_paths: List[Path],
    input_root: Path,
) -> Dict[str, Any]:
    """Benchmark sequential processing.

    Args:
        orchestrator: Orchestrator with parallel processing disabled
        image_paths: List of images to process
        input_root: Root directory for images

    Returns:
        Benchmark results
    """
    logger.info("=== Sequential Processing Benchmark ===")

    start_time = time.time()
    results = []

    for img_path in image_paths:
        try:
            result = orchestrator.enhance_image(ImageInput(img_path), input_root)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed {img_path}: {e}")
            results.append({"status": "error", "error": str(e)})

    elapsed = time.time() - start_time

    successful = sum(1 for r in results if r.get("status") == "ok")

    return {
        "mode": "sequential",
        "total_images": len(image_paths),
        "successful": successful,
        "elapsed_seconds": elapsed,
        "throughput_images_per_second": successful / elapsed if elapsed > 0 else 0,
    }


def benchmark_parallel(
    orchestrator: EnhanceOrchestrator, image_paths: List[Path], input_root: Path, workers: int
) -> Dict[str, Any]:
    """Benchmark parallel processing.

    Args:
        orchestrator: Orchestrator with parallel processing enabled
        image_paths: List of images to process
        input_root: Root directory for images
        workers: Number of parallel workers

    Returns:
        Benchmark results
    """
    logger.info(f"=== Parallel Processing Benchmark ({workers} workers) ===")

    start_time = time.time()

    image_inputs = [ImageInput(p) for p in image_paths]
    results = orchestrator.enhance_batch_parallel(image_inputs, input_root)

    elapsed = time.time() - start_time

    successful = sum(1 for r in results if r.get("status") == "ok")

    return {
        "mode": "parallel",
        "workers": workers,
        "total_images": len(image_paths),
        "successful": successful,
        "elapsed_seconds": elapsed,
        "throughput_images_per_second": successful / elapsed if elapsed > 0 else 0,
    }


def benchmark_cache_effectiveness(
    orchestrator: EnhanceOrchestrator, image_paths: List[Path], input_root: Path
) -> Dict[str, Any]:
    """Benchmark depth cache effectiveness.

    Args:
        orchestrator: Orchestrator with cache enabled
        image_paths: List of images to process (will process twice)
        input_root: Root directory for images

    Returns:
        Cache benchmark results
    """
    logger.info("=== Cache Effectiveness Benchmark ===")

    if not orchestrator.depth_cache:
        return {"mode": "cache", "error": "Cache not enabled"}

    # First pass: populate cache
    logger.info("First pass: populating cache...")
    start_time = time.time()
    image_inputs = [ImageInput(p) for p in image_paths]
    results_pass1 = orchestrator.enhance_batch_parallel(image_inputs, input_root)
    elapsed_pass1 = time.time() - start_time

    cache_stats_after_pass1 = orchestrator.depth_cache.stats()

    # Second pass: should hit cache
    logger.info("Second pass: testing cache hits...")
    start_time = time.time()
    results_pass2 = orchestrator.enhance_batch_parallel(image_inputs, input_root)
    elapsed_pass2 = time.time() - start_time

    cache_stats_after_pass2 = orchestrator.depth_cache.stats()

    speedup = elapsed_pass1 / elapsed_pass2 if elapsed_pass2 > 0 else 0

    return {
        "mode": "cache",
        "total_images": len(image_paths),
        "pass1_elapsed_seconds": elapsed_pass1,
        "pass2_elapsed_seconds": elapsed_pass2,
        "speedup_ratio": speedup,
        "cache_entries_after_pass1": cache_stats_after_pass1["entry_count"],
        "cache_entries_after_pass2": cache_stats_after_pass2["entry_count"],
        "cache_size_gb": cache_stats_after_pass2["size_gb"],
    }


def benchmark_worker_scalability(
    output_dir: Path, image_paths: List[Path], input_root: Path, worker_counts: List[int]
) -> List[Dict[str, Any]]:
    """Benchmark scalability across different worker counts.

    Args:
        output_dir: Output directory for results
        image_paths: List of images to process
        input_root: Root directory for images
        worker_counts: List of worker counts to test

    Returns:
        List of benchmark results for each worker count
    """
    logger.info("=== Worker Scalability Benchmark ===")

    results = []

    for workers in worker_counts:
        logger.info(f"Testing with {workers} workers...")

        # Create fresh output directory for this run
        run_output = output_dir / f"workers_{workers}"

        config = EnhanceConfig(
            model_key="da3-metric",
            enable_parallel_processing=True,
            max_parallel_workers=workers,
            enable_depth_cache=False,
            enable_v2=False,
        )

        orchestrator = _prepared_orchestrator(config, run_output, input_root, image_paths)

        benchmark = benchmark_parallel(orchestrator, image_paths, input_root, workers)
        results.append(benchmark)

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Phase 2 parallelization")
    parser.add_argument("--input-dir", type=Path, help="Directory with test images")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic images")
    parser.add_argument("--num-images", type=int, default=20, help="Number of synthetic images")
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Worker counts to test (default: 1 2 4)",
    )
    parser.add_argument("--test-cache", action="store_true", help="Test cache effectiveness")
    parser.add_argument("--output", type=Path, help="Output directory for results")

    args = parser.parse_args()

    # Setup output directory
    if args.output:
        output_base = args.output
    else:
        output_base = Path(tempfile.mkdtemp(prefix="phase2_benchmark_"))

    output_base.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_base}")

    # Get image paths
    if args.synthetic:
        synthetic_dir = output_base / "synthetic_images"
        image_paths = create_synthetic_images(synthetic_dir, args.num_images)
        input_root = synthetic_dir
    elif args.input_dir:
        input_root = args.input_dir
        extensions = [".jpg", ".jpeg", ".png"]
        image_paths = []
        for ext in extensions:
            image_paths.extend(input_root.rglob(f"*{ext}"))
            image_paths.extend(input_root.rglob(f"*{ext.upper()}"))
        logger.info(f"Found {len(image_paths)} images in {input_root}")
    else:
        logger.error("Must specify --input-dir or --synthetic")
        return 1

    if not image_paths:
        logger.error("No images found to process")
        return 1

    all_results = []

    # Benchmark 1: Sequential processing
    logger.info("\n" + "=" * 60)
    seq_output = output_base / "sequential"
    seq_config = EnhanceConfig(
        model_key="da3-metric",
        enable_parallel_processing=False,
        enable_v2=False,
    )
    seq_orchestrator = _prepared_orchestrator(seq_config, seq_output, input_root, image_paths)
    seq_results = benchmark_sequential(seq_orchestrator, image_paths, input_root)
    all_results.append(seq_results)

    logger.info(f"Sequential: {seq_results['throughput_images_per_second']:.2f} images/sec")

    # Benchmark 2: Parallel processing with different worker counts
    logger.info("\n" + "=" * 60)
    scalability_results = benchmark_worker_scalability(output_base, image_paths, input_root, args.workers)
    all_results.extend(scalability_results)

    for result in scalability_results:
        logger.info(
            f"Parallel ({result['workers']} workers): "
            f"{result['throughput_images_per_second']:.2f} images/sec "
            f"({result['throughput_images_per_second'] / seq_results['throughput_images_per_second']:.2f}x)"
        )

    # Benchmark 3: Cache effectiveness (optional)
    if args.test_cache:
        logger.info("\n" + "=" * 60)
        cache_output = output_base / "cache_test"
        cache_config = EnhanceConfig(
            model_key="da3-metric",
            enable_parallel_processing=True,
            max_parallel_workers=4,
            enable_depth_cache=True,
            depth_cache_max_size_gb=5.0,
            enable_v2=False,
        )
        cache_orchestrator = _prepared_orchestrator(cache_config, cache_output, input_root, image_paths)
        cache_results = benchmark_cache_effectiveness(cache_orchestrator, image_paths, input_root)
        all_results.append(cache_results)

        logger.info(f"Cache speedup: {cache_results.get('speedup_ratio', 0):.2f}x")
        logger.info(f"Cache entries: {cache_results.get('cache_entries_after_pass2', 0)}")

    # Write summary
    logger.info("\n" + "=" * 60)
    logger.info("=== Benchmark Summary ===")
    summary_path = output_base / "benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "config": {
                    "num_images": len(image_paths),
                    "worker_counts_tested": args.workers,
                    "cache_tested": args.test_cache,
                },
                "results": all_results,
            },
            f,
            indent=2,
        )

    logger.info(f"Results saved to: {summary_path}")
    logger.info(f"Output directory: {output_base}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
