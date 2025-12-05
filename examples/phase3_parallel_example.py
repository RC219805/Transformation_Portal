#!/usr/bin/env python3
"""
Phase 3 Parallel Processing Example

Demonstrates multi-GPU/CPU batch processing with automatic load balancing.
"""

import time
from pathlib import Path

import numpy as np
from PIL import Image

from utils.parallel_processor import (
    ParallelProcessor,
    WorkerConfig,
    ProcessingMode,
    process_images_parallel
)


def example_1_basic_parallel():
    """Basic parallel processing example"""
    print("="*60)
    print("Example 1: Basic Parallel Processing")
    print("="*60)
    
    # Create some dummy image paths
    image_paths = [Path(f"image_{i:03d}.jpg") for i in range(20)]
    
    def process_image(path):
        """Simulate image processing"""
        time.sleep(0.1)
        return f"Processed {path.name}"
    
    # Use convenience function
    results = process_images_parallel(
        image_paths,
        process_image,
        num_workers=4,
        use_gpu=False,
        progress=True
    )
    
    print(f"\nProcessed {len(results)} images")
    print(f"Success: {sum(1 for _, err in results if err is None)}")
    print(f"Failed: {sum(1 for _, err in results if err is not None)}")


def example_2_custom_config():
    """Custom configuration example"""
    print("\n" + "="*60)
    print("Example 2: Custom Configuration")
    print("="*60)
    
    # Configure for specific hardware
    config = WorkerConfig(
        mode=ProcessingMode.MULTI_CPU,
        num_workers=4,
        memory_limit_gb=8.0,
        batch_size=2,
        timeout_seconds=300.0
    )
    
    processor = ParallelProcessor(config)
    
    print(f"Mode: {processor.mode.value}")
    print(f"Workers: {processor.num_workers}")
    print(f"GPUs available: {processor.num_gpus}")
    print(f"CPUs available: {processor.num_cpus}")
    
    items = list(range(50))
    
    def compute_intensive(x):
        time.sleep(0.05)
        return x ** 2
    
    results = processor.process_batch(items, compute_intensive)
    processor.print_summary()


def example_3_error_handling():
    """Error handling example"""
    print("\n" + "="*60)
    print("Example 3: Error Handling")
    print("="*60)
    
    processor = ParallelProcessor(WorkerConfig(num_workers=2))
    
    items = list(range(20))
    
    def process_with_errors(x):
        """Function that fails on certain inputs"""
        if x % 7 == 0:
            raise ValueError(f"Cannot process {x}")
        return x * 2
    
    results = processor.process_batch(items, process_with_errors)
    
    print("\nResults:")
    for idx, (result, error) in enumerate(results):
        if error:
            print(f"  Item {idx}: FAILED - {error}")
        else:
            print(f"  Item {idx}: SUCCESS - {result}")
    
    processor.print_summary()


def example_4_real_images():
    """Real image processing example"""
    print("\n" + "="*60)
    print("Example 4: Real Image Processing")
    print("="*60)
    
    # Check if we have test images
    input_dir = Path("input_images")
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        print("Creating dummy images for demonstration...")
        input_dir.mkdir(exist_ok=True)
        
        # Create dummy images
        for i in range(10):
            img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            Image.fromarray(img).save(input_dir / f"test_{i:02d}.jpg")
    
    image_paths = sorted(input_dir.glob("*.jpg"))[:10]
    
    if not image_paths:
        print("No images found")
        return
    
    output_dir = Path("output_parallel")
    output_dir.mkdir(exist_ok=True)
    
    def process_image(image_path):
        """Process a single image"""
        # Load image
        image = np.array(Image.open(image_path))
        
        # Simple enhancement (brightness increase)
        enhanced = np.clip(image * 1.2, 0, 255).astype(np.uint8)
        
        # Save result
        output_path = output_dir / f"enhanced_{image_path.name}"
        Image.fromarray(enhanced).save(output_path)
        
        return output_path
    
    # Process in parallel
    results = process_images_parallel(
        image_paths,
        process_image,
        num_workers=4,
        progress=True
    )
    
    print(f"\nProcessed {len(results)} images")
    print(f"Output directory: {output_dir}")


def example_5_gpu_comparison():
    """Compare CPU vs GPU processing"""
    print("\n" + "="*60)
    print("Example 5: CPU vs GPU Comparison")
    print("="*60)
    
    items = list(range(100))
    
    def compute_task(x):
        time.sleep(0.01)
        return x ** 2
    
    # Test CPU mode
    config_cpu = WorkerConfig(mode=ProcessingMode.MULTI_CPU, num_workers=4)
    processor_cpu = ParallelProcessor(config_cpu)
    
    start = time.time()
    processor_cpu.process_batch(items, compute_task)
    cpu_time = time.time() - start
    
    print(f"\nCPU Mode (4 workers):")
    print(f"  Time: {cpu_time:.2f}s")
    print(f"  Throughput: {processor_cpu.stats.throughput_per_hour:.1f} items/hr")
    
    # Test GPU mode (may fall back to CPU if no GPU)
    config_gpu = WorkerConfig(mode=ProcessingMode.AUTO)
    processor_gpu = ParallelProcessor(config_gpu)
    
    start = time.time()
    processor_gpu.process_batch(items, compute_task)
    gpu_time = time.time() - start
    
    print(f"\nAuto Mode ({processor_gpu.mode.value}):")
    print(f"  Time: {gpu_time:.2f}s")
    print(f"  Throughput: {processor_gpu.stats.throughput_per_hour:.1f} items/hr")
    
    if cpu_time > gpu_time:
        speedup = cpu_time / gpu_time
        print(f"\nSpeedup: {speedup:.2f}×")


def example_6_progress_tracking():
    """Progress tracking example"""
    print("\n" + "="*60)
    print("Example 6: Custom Progress Tracking")
    print("="*60)
    
    processor = ParallelProcessor()
    items = list(range(50))
    
    progress_data = {'completed': 0, 'total': len(items), 'last_update': time.time()}
    
    def progress_callback(completed, total):
        """Custom progress callback"""
        now = time.time()
        elapsed = now - progress_data['last_update']
        
        if elapsed > 0.5 or completed == total:
            percent = completed / total * 100
            rate = completed / (now - progress_data['last_update']) if elapsed > 0 else 0
            
            print(f"\rProgress: {completed}/{total} ({percent:.1f}%) - {rate:.1f} items/s", end="")
            progress_data['last_update'] = now
    
    def slow_task(x):
        time.sleep(0.05)
        return x * 2
    
    results = processor.process_batch(items, slow_task, progress_callback)
    print()  # New line after progress
    
    processor.print_summary()


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("Phase 3 Parallel Processing Examples")
    print("="*60)
    
    examples = [
        ("Basic Parallel Processing", example_1_basic_parallel),
        ("Custom Configuration", example_2_custom_config),
        ("Error Handling", example_3_error_handling),
        ("Real Image Processing", example_4_real_images),
        ("CPU vs GPU Comparison", example_5_gpu_comparison),
        ("Progress Tracking", example_6_progress_tracking),
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    print("  0. Run all examples")
    
    choice = input("\nSelect example (0-6): ").strip()
    
    if choice == "0":
        for name, func in examples:
            print(f"\n\nRunning: {name}")
            func()
    elif choice.isdigit() and 1 <= int(choice) <= len(examples):
        name, func = examples[int(choice) - 1]
        print(f"\nRunning: {name}")
        func()
    else:
        print("Invalid choice. Running all examples.")
        for name, func in examples:
            func()


if __name__ == "__main__":
    main()
