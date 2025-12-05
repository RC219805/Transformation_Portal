#!/usr/bin/env python3
"""
Phase 3 CoreML Export Example

Demonstrates CoreML depth estimation with Apple Neural Engine optimization.
"""

import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from transformation_portal.depth.models.coreml_exporter import (
        CoreMLExporter,
        CoreMLDepthEstimator
    )
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    MODULES_AVAILABLE = False


def example_1_export_model():
    """Export a single model to CoreML"""
    print("="*60)
    print("Example 1: Export Model to CoreML")
    print("="*60)
    
    if not MODULES_AVAILABLE:
        print("CoreML modules not available")
        return
    
    exporter = CoreMLExporter(cache_dir=Path("weights/coreml"))
    
    print("\nExporting depth_anything_v2_small...")
    print("(This may take several minutes on first run)")
    
    coreml_path = exporter.export_depth_model(
        model_name="depth_anything_v2_small",
        optimize_for_ane=True
    )
    
    if coreml_path:
        print(f"\n✓ Model exported successfully")
        print(f"  Path: {coreml_path}")
        print(f"  Size: {exporter._get_model_size(coreml_path):.1f} MB")
    else:
        print("\n✗ Export failed (may require PyTorch and coremltools)")


def example_2_list_models():
    """List available CoreML models"""
    print("\n" + "="*60)
    print("Example 2: List CoreML Models")
    print("="*60)
    
    if not MODULES_AVAILABLE:
        print("CoreML modules not available")
        return
    
    exporter = CoreMLExporter()
    models = exporter.list_models()
    
    if models:
        print(f"\nFound {len(models)} CoreML model(s):")
        for model in models:
            print(f"  - {model['name']}: {model['size_mb']:.1f} MB")
    else:
        print("\nNo CoreML models found. Run example 1 to export models.")


def example_3_depth_estimation():
    """Estimate depth from an image"""
    print("\n" + "="*60)
    print("Example 3: Depth Estimation")
    print("="*60)
    
    if not MODULES_AVAILABLE:
        print("CoreML modules not available")
        return
    
    # Create or load test image
    input_dir = Path("input_images")
    if not input_dir.exists():
        print("Creating dummy test image...")
        input_dir.mkdir(exist_ok=True)
        
        # Create a simple test image with depth cues
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        # Gradient from top to bottom (sky to ground)
        for y in range(512):
            intensity = int(255 * (1 - y / 512))
            img[y, :, :] = intensity
        
        Image.fromarray(img).save(input_dir / "test_depth.jpg")
    
    image_path = input_dir / "test_depth.jpg"
    if not image_path.exists():
        jpg_files = list(input_dir.glob("*.jpg"))
        image_path = jpg_files[0] if jpg_files else None
        
    if not image_path or not image_path.exists():
        print(f"No test image found in {input_dir}")
        return
    
    print(f"\nLoading image: {image_path}")
    image = np.array(Image.open(image_path))
    
    print("\nInitializing depth estimator...")
    print("(Will use CoreML if available, PyTorch otherwise)")
    
    try:
        estimator = CoreMLDepthEstimator(
            model_name="depth_anything_v2_small",
            prefer_coreml=True
        )
        
        print(f"Backend: {'CoreML' if estimator.use_coreml else 'PyTorch'}")
        
        print("\nEstimating depth...")
        start = time.time()
        depth = estimator.estimate(image)
        elapsed = time.time() - start
        
        print(f"\n✓ Depth estimation complete")
        print(f"  Time: {elapsed*1000:.1f}ms")
        print(f"  Depth shape: {depth.shape}")
        print(f"  Depth range: [{depth.min():.3f}, {depth.max():.3f}]")
        
        # Save depth map visualization
        output_dir = Path("output_depth")
        output_dir.mkdir(exist_ok=True)
        
        depth_viz = (depth * 255).astype(np.uint8)
        Image.fromarray(depth_viz).save(output_dir / f"depth_{image_path.stem}.png")
        print(f"  Saved: {output_dir}/depth_{image_path.stem}.png")
        
    except Exception as e:
        print(f"\n✗ Depth estimation failed: {e}")


def example_4_benchmark():
    """Benchmark depth estimation performance"""
    print("\n" + "="*60)
    print("Example 4: Performance Benchmark")
    print("="*60)
    
    if not MODULES_AVAILABLE:
        print("CoreML modules not available")
        return
    
    try:
        print("\nInitializing estimator...")
        estimator = CoreMLDepthEstimator(prefer_coreml=True)
        
        print(f"Backend: {'CoreML' if estimator.use_coreml else 'PyTorch'}")
        print("\nRunning benchmark (50 iterations)...")
        print("This may take a minute...")
        
        results = estimator.benchmark(num_iterations=50)
        
        print("\n" + "="*60)
        print("Benchmark Results")
        print("="*60)
        print(f"Backend:     {results['backend']}")
        print(f"Model:       {results['model']}")
        print(f"Iterations:  {results['iterations']}")
        print(f"Mean:        {results['mean_ms']:.1f}ms")
        print(f"Std:         {results['std_ms']:.1f}ms")
        print(f"Min:         {results['min_ms']:.1f}ms")
        print(f"Max:         {results['max_ms']:.1f}ms")
        print(f"Median:      {results['median_ms']:.1f}ms")
        print(f"Throughput:  {results['throughput_per_hour']:.0f} images/hour")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Benchmark failed: {e}")


def example_5_batch_processing():
    """Batch process multiple images"""
    print("\n" + "="*60)
    print("Example 5: Batch Depth Estimation")
    print("="*60)
    
    if not MODULES_AVAILABLE:
        print("CoreML modules not available")
        return
    
    # Create test images if needed
    input_dir = Path("input_images")
    if not input_dir.exists() or len(list(input_dir.glob("*.jpg"))) == 0:
        print("Creating test images...")
        input_dir.mkdir(exist_ok=True)
        
        for i in range(5):
            img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            Image.fromarray(img).save(input_dir / f"batch_{i:02d}.jpg")
    
    image_paths = sorted(input_dir.glob("*.jpg"))[:5]
    
    if not image_paths:
        print(f"No images found in {input_dir}")
        return
    
    print(f"\nProcessing {len(image_paths)} images...")
    
    try:
        estimator = CoreMLDepthEstimator(prefer_coreml=True)
        print(f"Backend: {'CoreML' if estimator.use_coreml else 'PyTorch'}")
        
        output_dir = Path("output_depth_batch")
        output_dir.mkdir(exist_ok=True)
        
        start = time.time()
        
        for i, image_path in enumerate(image_paths, 1):
            print(f"\rProcessing {i}/{len(image_paths)}: {image_path.name}", end="")
            
            image = np.array(Image.open(image_path))
            depth = estimator.estimate(image)
            
            depth_viz = (depth * 255).astype(np.uint8)
            output_path = output_dir / f"depth_{image_path.stem}.png"
            Image.fromarray(depth_viz).save(output_path)
        
        elapsed = time.time() - start
        print(f"\n\n✓ Batch processing complete")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Time per image: {elapsed/len(image_paths)*1000:.1f}ms")
        print(f"  Throughput: {len(image_paths)/elapsed*3600:.0f} images/hour")
        print(f"  Output: {output_dir}")
        
    except Exception as e:
        print(f"\n✗ Batch processing failed: {e}")


def example_6_compare_backends():
    """Compare CoreML vs PyTorch performance"""
    print("\n" + "="*60)
    print("Example 6: CoreML vs PyTorch Comparison")
    print("="*60)
    
    if not MODULES_AVAILABLE:
        print("CoreML modules not available")
        return
    
    print("\nThis example compares CoreML and PyTorch backends")
    print("on the same hardware (if both are available).\n")
    
    results = {}
    
    # Test PyTorch
    try:
        print("Testing PyTorch backend...")
        estimator_pt = CoreMLDepthEstimator(prefer_coreml=False)
        results['pytorch'] = estimator_pt.benchmark(num_iterations=20)
        print(f"✓ PyTorch: {results['pytorch']['mean_ms']:.1f}ms")
    except Exception as e:
        print(f"✗ PyTorch failed: {e}")
    
    # Test CoreML
    try:
        print("\nTesting CoreML backend...")
        estimator_cm = CoreMLDepthEstimator(prefer_coreml=True)
        if estimator_cm.use_coreml:
            results['coreml'] = estimator_cm.benchmark(num_iterations=20)
            print(f"✓ CoreML: {results['coreml']['mean_ms']:.1f}ms")
        else:
            print("✗ CoreML not available (using PyTorch)")
    except Exception as e:
        print(f"✗ CoreML failed: {e}")
    
    # Compare results
    if len(results) == 2:
        print("\n" + "="*60)
        print("Performance Comparison")
        print("="*60)
        
        pt_time = results['pytorch']['mean_ms']
        cm_time = results['coreml']['mean_ms']
        speedup = pt_time / cm_time
        
        print(f"PyTorch:  {pt_time:.1f}ms")
        print(f"CoreML:   {cm_time:.1f}ms")
        print(f"Speedup:  {speedup:.2f}× faster")
        print("="*60)


def main():
    """Run examples"""
    print("\n" + "="*60)
    print("Phase 3 CoreML Depth Estimation Examples")
    print("="*60)
    
    if not MODULES_AVAILABLE:
        print("\n✗ CoreML modules not available")
        print("Please ensure depth_pipeline is properly installed.")
        return
    
    examples = [
        ("Export Model to CoreML", example_1_export_model),
        ("List CoreML Models", example_2_list_models),
        ("Depth Estimation", example_3_depth_estimation),
        ("Performance Benchmark", example_4_benchmark),
        ("Batch Processing", example_5_batch_processing),
        ("CoreML vs PyTorch", example_6_compare_backends),
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
        print("Invalid choice. Running depth estimation demo.")
        example_3_depth_estimation()


if __name__ == "__main__":
    main()
