#!/usr/bin/env python3
"""
Phase 3 Combined Features Example

Demonstrates using all three Phase 3 features together:
- Parallel processing for batch operations
- CoreML depth estimation for speed
- Incremental caching for iteration efficiency
"""

import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from utils.parallel_processor import ParallelProcessor, WorkerConfig
    from utils.incremental_cache import IncrementalCache, CacheConfig
    from depth_pipeline.coreml_exporter import CoreMLDepthEstimator
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    MODULES_AVAILABLE = False


def setup_test_images(count=20):
    """Create test images if they don't exist"""
    input_dir = Path("input_images")
    input_dir.mkdir(exist_ok=True)
    
    existing = list(input_dir.glob("combined_*.jpg"))
    if len(existing) >= count:
        return sorted(existing)[:count]
    
    print(f"Creating {count} test images...")
    for i in range(count):
        # Create image with some structure (gradient + noise)
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        
        # Vertical gradient (depth cue)
        for y in range(512):
            base = int(200 * (1 - y / 512))
            img[y, :, :] = base
        
        # Add some noise and color
        noise = np.random.randint(-20, 20, (512, 512, 3))
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        img[:, :, 1] += 20  # Slight green tint
        
        Image.fromarray(img).save(input_dir / f"combined_{i:03d}.jpg")
    
    return sorted(input_dir.glob("combined_*.jpg"))


def example_1_sequential_baseline():
    """Baseline: Sequential processing without optimizations"""
    print("="*60)
    print("Example 1: Sequential Baseline (No Optimizations)")
    print("="*60)
    
    if not MODULES_AVAILABLE:
        print("Required modules not available")
        return None
    
    image_paths = setup_test_images(10)
    output_dir = Path("output_sequential")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nProcessing {len(image_paths)} images sequentially...")
    
    # Simple depth estimator (mocked for demo)
    def estimate_depth_simple(image):
        """Simple depth estimation (grayscale conversion)"""
        time.sleep(0.5)  # Simulate slow depth estimation
        gray = np.mean(image, axis=2)
        return (gray - gray.min()) / (gray.max() - gray.min())
    
    start_time = time.time()
    
    for i, image_path in enumerate(image_paths, 1):
        print(f"\rProcessing {i}/{len(image_paths)}", end="")
        
        # Load image
        image = np.array(Image.open(image_path))
        
        # Estimate depth (slow)
        depth = estimate_depth_simple(image)
        
        # Apply simple effect
        enhanced = image * 1.2
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        # Save
        output_path = output_dir / f"result_{image_path.stem}.jpg"
        Image.fromarray(enhanced).save(output_path)
    
    elapsed = time.time() - start_time
    throughput = len(image_paths) / elapsed * 3600
    
    print(f"\n\n✓ Sequential processing complete")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Time per image: {elapsed/len(image_paths):.2f}s")
    print(f"  Throughput: {throughput:.0f} images/hour")
    
    return elapsed


def example_2_parallel_only():
    """Use parallel processing but no caching or CoreML"""
    print("\n" + "="*60)
    print("Example 2: Parallel Processing Only")
    print("="*60)
    
    if not MODULES_AVAILABLE:
        print("Required modules not available")
        return None
    
    image_paths = setup_test_images(10)
    output_dir = Path("output_parallel_only")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nProcessing {len(image_paths)} images in parallel...")
    
    def process_image(image_path):
        """Process a single image"""
        image = np.array(Image.open(image_path))
        
        # Simulate depth estimation
        time.sleep(0.5)
        gray = np.mean(image, axis=2)
        depth = (gray - gray.min()) / (gray.max() - gray.min())
        
        # Apply effect
        enhanced = image * 1.2
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        # Save
        output_path = output_dir / f"result_{image_path.stem}.jpg"
        Image.fromarray(enhanced).save(output_path)
        
        return output_path
    
    # Use parallel processor
    config = WorkerConfig(num_workers=4)
    processor = ParallelProcessor(config)
    
    start_time = time.time()
    results = processor.process_batch(image_paths, process_image)
    elapsed = time.time() - start_time
    
    print(f"\n✓ Parallel processing complete")
    processor.print_summary()
    
    return elapsed


def example_3_with_caching():
    """Use parallel processing + caching"""
    print("\n" + "="*60)
    print("Example 3: Parallel + Caching")
    print("="*60)
    
    if not MODULES_AVAILABLE:
        print("Required modules not available")
        return None
    
    image_paths = setup_test_images(10)
    output_dir = Path("output_parallel_cached")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize cache
    cache_config = CacheConfig(
        cache_dir=Path(".cache/phase3_example"),
        max_size_gb=1.0
    )
    cache = IncrementalCache(cache_config)
    
    print(f"\nProcessing {len(image_paths)} images with caching...")
    print("First run will be slow, second run will be fast!\n")
    
    def process_image_cached(image_path):
        """Process with caching"""
        image = np.array(Image.open(image_path))
        
        # Get or compute depth (cached)
        depth = cache.get_or_compute(
            "depth_maps",
            lambda: compute_depth(image),
            inputs={"image": image_path}
        )
        
        # Apply effect (not cached)
        enhanced = image * 1.2
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        output_path = output_dir / f"result_{image_path.stem}.jpg"
        Image.fromarray(enhanced).save(output_path)
        
        return output_path
    
    def compute_depth(image):
        """Slow depth computation"""
        time.sleep(0.5)
        gray = np.mean(image, axis=2)
        return (gray - gray.min()) / (gray.max() - gray.min())
    
    # First run
    processor = ParallelProcessor(WorkerConfig(num_workers=4))
    
    print("First run (computing depth):")
    start_time = time.time()
    processor.process_batch(image_paths, process_image_cached)
    first_run = time.time() - start_time
    print(f"  Time: {first_run:.2f}s\n")
    
    # Second run (cached depth)
    print("Second run (depth cached):")
    start_time = time.time()
    processor.process_batch(image_paths, process_image_cached)
    second_run = time.time() - start_time
    print(f"  Time: {second_run:.2f}s\n")
    
    speedup = first_run / second_run if second_run > 0 else 0
    print(f"✓ Cache speedup: {speedup:.2f}×")
    
    cache.print_stats()
    
    return (first_run, second_run)


def example_4_full_optimization():
    """Use all three features: Parallel + CoreML + Caching"""
    print("\n" + "="*60)
    print("Example 4: Full Optimization (Parallel + CoreML + Caching)")
    print("="*60)
    
    if not MODULES_AVAILABLE:
        print("Required modules not available")
        return None
    
    image_paths = setup_test_images(10)
    output_dir = Path("output_fully_optimized")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nProcessing {len(image_paths)} images with full optimization...")
    
    # Initialize all components
    cache = IncrementalCache(CacheConfig(cache_dir=Path(".cache/phase3_full")))
    
    try:
        depth_estimator = CoreMLDepthEstimator(prefer_coreml=True)
        print(f"Depth backend: {'CoreML' if depth_estimator.use_coreml else 'PyTorch'}")
    except:
        print("CoreML/PyTorch not available, using mock depth estimator")
        depth_estimator = None
    
    processor = ParallelProcessor(WorkerConfig(num_workers=4))
    
    def process_image_optimized(image_path):
        """Fully optimized processing"""
        image = np.array(Image.open(image_path))
        
        # Get or compute depth with caching
        depth = cache.get_or_compute(
            "depth_maps_coreml",
            lambda: estimate_depth_fast(image),
            inputs={"image": image_path, "model": "coreml"}
        )
        
        # Apply depth-aware enhancement
        enhanced = apply_depth_enhancement(image, depth)
        
        # Save
        output_path = output_dir / f"result_{image_path.stem}.jpg"
        Image.fromarray(enhanced).save(output_path)
        
        return output_path
    
    def estimate_depth_fast(image):
        """Fast depth estimation"""
        if depth_estimator:
            return depth_estimator.estimate(image)
        else:
            # Mock fast depth
            time.sleep(0.05)  # Much faster than 0.5s
            gray = np.mean(image, axis=2)
            return (gray - gray.min()) / (gray.max() - gray.min())
    
    def apply_depth_enhancement(image, depth):
        """Apply depth-aware effects"""
        # Simple depth-based adjustment
        depth_3d = np.stack([depth] * 3, axis=2)
        enhanced = image.astype(np.float32)
        enhanced = enhanced * (0.8 + 0.4 * depth_3d)
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    
    # First run
    print("\nFirst run:")
    start_time = time.time()
    processor.process_batch(image_paths, process_image_optimized)
    first_run = time.time() - start_time
    print(f"  Time: {first_run:.2f}s")
    
    # Second run (fully cached)
    print("\nSecond run (cached):")
    start_time = time.time()
    processor.process_batch(image_paths, process_image_optimized)
    second_run = time.time() - start_time
    print(f"  Time: {second_run:.2f}s")
    
    print("\n✓ Full optimization complete")
    processor.print_summary()
    cache.print_stats()
    
    speedup = first_run / second_run if second_run > 0 else 0
    print(f"\nCache speedup: {speedup:.2f}×")
    
    return (first_run, second_run)


def example_5_performance_comparison():
    """Compare all approaches"""
    print("\n" + "="*60)
    print("Example 5: Performance Comparison")
    print("="*60)
    
    if not MODULES_AVAILABLE:
        print("Required modules not available")
        return
    
    print("\nThis example compares all optimization strategies:")
    print("1. Sequential baseline")
    print("2. Parallel processing")
    print("3. Parallel + Caching")
    print("4. Parallel + CoreML + Caching")
    
    results = {}
    
    # Run baseline
    print("\n" + "-"*60)
    results['sequential'] = example_1_sequential_baseline()
    
    # Run parallel only
    print("\n" + "-"*60)
    results['parallel'] = example_2_parallel_only()
    
    # Run with caching
    print("\n" + "-"*60)
    cache_times = example_3_with_caching()
    if cache_times:
        results['parallel_cached'] = cache_times[1]  # Use second run time
    
    # Run fully optimized
    print("\n" + "-"*60)
    full_times = example_4_full_optimization()
    if full_times:
        results['full_optimized'] = full_times[1]  # Use second run time
    
    # Print comparison
    print("\n" + "="*60)
    print("Performance Comparison Summary")
    print("="*60)
    
    if 'sequential' in results and results['sequential']:
        baseline = results['sequential']
        
        print(f"\nBaseline (Sequential):          {baseline:.2f}s  (1.00×)")
        
        if 'parallel' in results and results['parallel']:
            speedup = baseline / results['parallel']
            print(f"Parallel Processing:            {results['parallel']:.2f}s  ({speedup:.2f}×)")
        
        if 'parallel_cached' in results and results['parallel_cached']:
            speedup = baseline / results['parallel_cached']
            print(f"Parallel + Caching:             {results['parallel_cached']:.2f}s  ({speedup:.2f}×)")
        
        if 'full_optimized' in results and results['full_optimized']:
            speedup = baseline / results['full_optimized']
            print(f"Parallel + CoreML + Caching:    {results['full_optimized']:.2f}s  ({speedup:.2f}×)")
    
    print("="*60)


def main():
    """Run examples"""
    print("\n" + "="*60)
    print("Phase 3 Combined Features Examples")
    print("="*60)
    
    if not MODULES_AVAILABLE:
        print("\n✗ Required modules not available")
        return
    
    examples = [
        ("Sequential Baseline", example_1_sequential_baseline),
        ("Parallel Processing Only", example_2_parallel_only),
        ("Parallel + Caching", example_3_with_caching),
        ("Full Optimization", example_4_full_optimization),
        ("Performance Comparison", example_5_performance_comparison),
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    print("  0. Run performance comparison")
    
    choice = input("\nSelect example (0-5): ").strip()
    
    if choice == "0":
        example_5_performance_comparison()
    elif choice.isdigit() and 1 <= int(choice) <= len(examples):
        name, func = examples[int(choice) - 1]
        print(f"\nRunning: {name}")
        func()
    else:
        print("Invalid choice. Running performance comparison.")
        example_5_performance_comparison()


if __name__ == "__main__":
    main()
