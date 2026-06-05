#!/usr/bin/env python3
"""
Benchmark script for Phase 3 Advanced Optimizations.

Tests performance of:
1. CoreML ANE vs PyTorch MPS (depth inference)
2. PBR GPU batching vs sequential
3. MessagePack vs JSON (manifest I/O)
4. xxHash vs SHA-1 (output key generation)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def benchmark_coreml():
    """Benchmark CoreML vs PyTorch MPS for depth inference."""
    print("\n" + "=" * 80)
    print("Benchmark 1: CoreML ANE vs PyTorch MPS (Depth Inference)")
    print("=" * 80)

    import platform

    if platform.system() != "Darwin" or platform.machine() != "arm64":
        print("⚠️  Skipped: CoreML only available on Apple Silicon")
        return None

    try:
        from transformation_portal.lux_depth_v3.config import DA3Config, DeviceConfig
        from transformation_portal.lux_depth_v3.inference import DA3InferenceEngine
    except ImportError as e:
        print(f"⚠️  Skipped: {e}")
        return None

    # Create test image
    test_image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)

    results = {}

    # Test PyTorch MPS
    print("\nTesting PyTorch MPS...")
    try:
        config_mps = DA3Config(device=DeviceConfig(device="mps", use_coreml=False))
        engine_mps = DA3InferenceEngine(config_mps)

        # Warmup
        _ = engine_mps.infer(test_image)

        # Benchmark (5 runs)
        times = []
        for i in range(5):
            start = time.time()
            _ = engine_mps.infer(test_image)
            times.append((time.time() - start) * 1000)

        avg_time = np.mean(times)
        print(f"  PyTorch MPS: {avg_time:.1f}ms/image (avg of 5 runs)")
        results["pytorch_mps_ms"] = avg_time
    except Exception as e:
        print(f"  PyTorch MPS failed: {e}")

    # Test CoreML ANE
    print("\nTesting CoreML ANE...")
    try:
        import coremltools

        config_coreml = DA3Config(device=DeviceConfig(device="mps", use_coreml=True))
        engine_coreml = DA3InferenceEngine(config_coreml)

        # Warmup
        _ = engine_coreml.infer(test_image)

        # Benchmark (5 runs)
        times = []
        for i in range(5):
            start = time.time()
            _ = engine_coreml.infer(test_image)
            times.append((time.time() - start) * 1000)

        avg_time = np.mean(times)
        print(f"  CoreML ANE: {avg_time:.1f}ms/image (avg of 5 runs)")
        results["coreml_ane_ms"] = avg_time

        if "pytorch_mps_ms" in results:
            speedup = results["pytorch_mps_ms"] / avg_time
            print(f"\n  ✓ Speedup: {speedup:.1f}x")
            results["speedup"] = speedup
    except ImportError:
        print("  ⚠️  CoreML skipped: coremltools not installed")
    except Exception as e:
        print(f"  CoreML ANE failed: {e}")

    return results


def benchmark_pbr_batching(num_images=10):
    """Benchmark PBR GPU batching vs sequential."""
    print("\n" + "=" * 80)
    print(f"Benchmark 2: PBR GPU Batching vs Sequential ({num_images} images)")
    print("=" * 80)

    try:
        from transformation_portal.lux_depth_v3.pbr import PBRConfig, generate_pbr_maps, generate_pbr_maps_batched
    except ImportError as e:
        print(f"⚠️  Skipped: {e}")
        return None

    # Create test depth maps
    depths = [np.random.rand(512, 512).astype(np.float32) for _ in range(num_images)]
    config = PBRConfig()

    results = {}

    # Sequential
    print("\nTesting sequential PBR generation...")
    start = time.time()
    seq_results = [generate_pbr_maps(depth, config) for depth in depths]
    seq_time = (time.time() - start) * 1000
    print(f"  Sequential: {seq_time:.1f}ms total ({seq_time/num_images:.1f}ms/image)")
    results["sequential_ms"] = seq_time

    # Batched (CPU)
    print("\nTesting batched PBR generation (CPU)...")
    start = time.time()
    batch_results = generate_pbr_maps_batched(depths, config, device="cpu")
    batch_time = (time.time() - start) * 1000
    print(f"  Batched (CPU): {batch_time:.1f}ms total ({batch_time/num_images:.1f}ms/image)")
    results["batched_cpu_ms"] = batch_time

    speedup_cpu = seq_time / batch_time
    print(f"  CPU speedup: {speedup_cpu:.2f}x")

    # Batched (GPU) if available
    try:
        import torch

        if torch.backends.mps.is_available():
            print("\nTesting batched PBR generation (MPS)...")
            start = time.time()
            batch_results_gpu = generate_pbr_maps_batched(depths, config, device="mps")
            batch_time_gpu = (time.time() - start) * 1000
            print(f"  Batched (MPS): {batch_time_gpu:.1f}ms total ({batch_time_gpu/num_images:.1f}ms/image)")
            results["batched_mps_ms"] = batch_time_gpu

            speedup_gpu = seq_time / batch_time_gpu
            print(f"\n  ✓ MPS speedup: {speedup_gpu:.2f}x")
            results["speedup"] = speedup_gpu
        elif torch.cuda.is_available():
            print("\nTesting batched PBR generation (CUDA)...")
            start = time.time()
            batch_results_gpu = generate_pbr_maps_batched(depths, config, device="cuda")
            batch_time_gpu = (time.time() - start) * 1000
            print(f"  Batched (CUDA): {batch_time_gpu:.1f}ms total ({batch_time_gpu/num_images:.1f}ms/image)")
            results["batched_cuda_ms"] = batch_time_gpu

            speedup_gpu = seq_time / batch_time_gpu
            print(f"\n  ✓ CUDA speedup: {speedup_gpu:.2f}x")
            results["speedup"] = speedup_gpu
        else:
            print("\n  ⚠️  No GPU available for batching test")
    except ImportError:
        print("\n  ⚠️  torch not available for GPU batching test")

    return results


def benchmark_msgpack(num_manifests=1000):
    """Benchmark MessagePack vs JSON for manifests."""
    print("\n" + "=" * 80)
    print(f"Benchmark 3: MessagePack vs JSON ({num_manifests} manifests)")
    print("=" * 80)

    try:
        from transformation_portal.lux_depth_v3.manifest import CombinedManifest, InputMetadata
    except ImportError as e:
        print(f"⚠️  Skipped: {e}")
        return None

    import tempfile

    # Create test manifests
    manifests = []
    for i in range(num_manifests):
        manifest = CombinedManifest()
        manifest.input = InputMetadata(
            image_path=f"/test/images/image_{i:04d}.jpg",
            image_sha256="abc123def456" * 5,
            image_size_bytes=1024000,
            image_dimensions=(1920, 1080),
        )
        manifest.start_time = "2024-01-01T00:00:00Z"
        manifest.end_time = "2024-01-01T00:01:00Z"
        manifests.append(manifest)

    results = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # JSON
        print("\nTesting JSON serialization...")
        json_files = []
        start = time.time()
        for i, manifest in enumerate(manifests):
            json_path = tmpdir / f"manifest_{i:04d}.json"
            manifest.save(json_path)
            json_files.append(json_path)
        json_write_time = (time.time() - start) * 1000

        json_size = sum(f.stat().st_size for f in json_files)
        print(f"  JSON write: {json_write_time:.1f}ms, size: {json_size/1024/1024:.2f}MB")
        results["json_write_ms"] = json_write_time
        results["json_size_mb"] = json_size / 1024 / 1024

        # JSON read
        start = time.time()
        for json_path in json_files:
            _ = CombinedManifest.load(json_path)
        json_read_time = (time.time() - start) * 1000
        print(f"  JSON read: {json_read_time:.1f}ms")
        results["json_read_ms"] = json_read_time

        # MessagePack
        try:
            import msgpack

            print("\nTesting MessagePack serialization...")
            msgpack_files = []
            start = time.time()
            for i, manifest in enumerate(manifests):
                msgpack_path = tmpdir / f"manifest_{i:04d}.msgpack"
                manifest.save_msgpack(msgpack_path)
                msgpack_files.append(msgpack_path)
            msgpack_write_time = (time.time() - start) * 1000

            msgpack_size = sum(f.stat().st_size for f in msgpack_files)
            print(f"  MessagePack write: {msgpack_write_time:.1f}ms, size: {msgpack_size/1024/1024:.2f}MB")
            results["msgpack_write_ms"] = msgpack_write_time
            results["msgpack_size_mb"] = msgpack_size / 1024 / 1024

            # MessagePack read
            start = time.time()
            for msgpack_path in msgpack_files:
                _ = CombinedManifest.load_msgpack(msgpack_path)
            msgpack_read_time = (time.time() - start) * 1000
            print(f"  MessagePack read: {msgpack_read_time:.1f}ms")
            results["msgpack_read_ms"] = msgpack_read_time

            # Comparison
            size_reduction = (1 - msgpack_size / json_size) * 100
            read_speedup = json_read_time / msgpack_read_time
            print(f"\n  ✓ Size reduction: {size_reduction:.1f}%")
            print(f"  ✓ Read speedup: {read_speedup:.1f}x")
            results["size_reduction_pct"] = size_reduction
            results["read_speedup"] = read_speedup
        except ImportError:
            print("\n  ⚠️  msgpack not available")

    return results


def benchmark_xxhash(num_operations=10000):
    """Benchmark xxHash vs SHA-1 for output keys."""
    print("\n" + "=" * 80)
    print(f"Benchmark 4: xxHash vs SHA-1 ({num_operations} operations)")
    print("=" * 80)

    import hashlib

    # Test data
    test_paths = [
        f"photos/scene{i:04d}/subfolder/image_{j:04d}.jpg".encode()
        for i in range(10)
        for j in range(num_operations // 10)
    ]

    results = {}

    # SHA-1
    print("\nTesting SHA-1...")
    start = time.time()
    for path in test_paths:
        _ = hashlib.sha1(path).hexdigest()[:8]
    sha1_time = (time.time() - start) * 1000
    print(f"  SHA-1: {sha1_time:.1f}ms ({sha1_time/num_operations*1000:.2f}µs/op)")
    results["sha1_ms"] = sha1_time

    # xxHash
    try:
        import xxhash

        print("\nTesting xxHash...")
        start = time.time()
        for path in test_paths:
            _ = xxhash.xxh64(path).hexdigest()[:8]
        xxhash_time = (time.time() - start) * 1000
        print(f"  xxHash: {xxhash_time:.1f}ms ({xxhash_time/num_operations*1000:.2f}µs/op)")
        results["xxhash_ms"] = xxhash_time

        speedup = sha1_time / xxhash_time
        print(f"\n  ✓ Speedup: {speedup:.1f}x")
        results["speedup"] = speedup
    except ImportError:
        print("\n  ⚠️  xxhash not available")

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Performance Benchmarks")
    parser.add_argument("--coreml", action="store_true", help="Benchmark CoreML ANE")
    parser.add_argument("--pbr-batch", action="store_true", help="Benchmark PBR batching")
    parser.add_argument("--msgpack", action="store_true", help="Benchmark MessagePack")
    parser.add_argument("--xxhash", action="store_true", help="Benchmark xxHash")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--test-images", type=int, default=10, help="Number of test images for PBR")
    parser.add_argument("--output", type=Path, help="Save results to JSON file")

    args = parser.parse_args()

    if not any([args.coreml, args.pbr_batch, args.msgpack, args.xxhash, args.all]):
        parser.print_help()
        return

    all_results = {}

    if args.all or args.coreml:
        results = benchmark_coreml()
        if results:
            all_results["coreml"] = results

    if args.all or args.pbr_batch:
        results = benchmark_pbr_batching(num_images=args.test_images)
        if results:
            all_results["pbr_batching"] = results

    if args.all or args.msgpack:
        results = benchmark_msgpack(num_manifests=1000)
        if results:
            all_results["msgpack"] = results

    if args.all or args.xxhash:
        results = benchmark_xxhash(num_operations=10000)
        if results:
            all_results["xxhash"] = results

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for category, results in all_results.items():
        print(f"\n{category.upper()}:")
        for key, value in results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")

    # Save results
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✓ Results saved to {args.output}")


if __name__ == "__main__":
    main()
