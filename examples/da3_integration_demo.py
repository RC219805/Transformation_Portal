#!/usr/bin/env python3
"""
DA3 Integration Demo for Transformation Portal

This script demonstrates various ways to use the DA3 integration
for luxury real estate rendering workflows.
"""

import os
from pathlib import Path
from lux_depth_v3.da3_integration import (
    DA3DepthEstimator,
    estimate_depth,
    convert_to_metric_depth
)

# Set environment for Mac compatibility
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def example_1_quick_start():
    """Example 1: Quick depth estimation."""
    print("=" * 60)
    print("Example 1: Quick Depth Estimation")
    print("=" * 60)
    
    result = estimate_depth(
        "input_images/750_Picacho/Kitchen_2K_test.png",
        "test_output/demo_example1",
        model="large-1.1",
        device="cpu"
    )
    
    if result.success:
        print("✅ Success!")
        print(f"   GLB: {result.glb_path}")
        print(f"   Scene JPG: {result.scene_jpg}")
        if result.depth_array is not None:
            depth = result.depth_array
            print(f"   Depth shape: {depth.shape}")
            print(f"   Depth range: [{depth.min():.3f}, {depth.max():.3f}]")
    else:
        print(f"❌ Failed: {result.stderr}")
    
    print()


def example_2_advanced_processing():
    """Example 2: Advanced processing with multiple outputs."""
    print("=" * 60)
    print("Example 2: Advanced Multi-Format Export")
    print("=" * 60)
    
    estimator = DA3DepthEstimator(
        model="large-1.1",
        device="cpu",
        verbose=True,
        auto_cleanup=True
    )
    
    result = estimator.process_image(
        input_path="input_images/750_Picacho/Kitchen_2K_test.png",
        output_dir="test_output/demo_example2",
        export_format="glb-depth_vis-mini_npz",
        process_res=1024  # Higher resolution
    )
    
    if result.success:
        print("\n✅ Processing complete!")
        print(f"\n📂 Output files:")
        print(f"   Directory: {result.output_dir}")
        print(f"   3D Model: {result.glb_path}")
        print(f"   Depth Vis: {result.depth_vis_dir}")
        print(f"   Data: {result.npz_path}")
        
        # Load and analyze depth
        depth = result.depth_array
        conf = result.confidence_array
        
        if depth is not None:
            print(f"\n📊 Depth Statistics:")
            print(f"   Shape: {depth.shape}")
            print(f"   Min: {depth.min():.3f}")
            print(f"   Max: {depth.max():.3f}")
            print(f"   Mean: {depth.mean():.3f}")
            print(f"   Std: {depth.std():.3f}")
        
        if conf is not None:
            print(f"\n📊 Confidence Statistics:")
            print(f"   Mean confidence: {conf.mean():.3f}")
            print(f"   Min confidence: {conf.min():.3f}")
    else:
        print(f"❌ Failed: {result.stderr}")
    
    print()


def example_3_batch_processing():
    """Example 3: Batch process directory."""
    print("=" * 60)
    print("Example 3: Batch Directory Processing")
    print("=" * 60)
    
    estimator = DA3DepthEstimator(model="large-1.1", device="cpu")
    
    result = estimator.process_directory(
        input_dir="input_images/750_Picacho",
        output_dir="test_output/demo_example3_batch",
        extensions=["png", "jpg"],
        export_format="depth_vis-mini_npz"
    )
    
    if result.success:
        print("✅ Batch processing complete!")
        print(f"   Output: {result.output_dir}")
        print(f"   Check depth_vis/ folder for visualizations")
    else:
        print(f"❌ Failed: {result.stderr}")
    
    print()


def example_4_metric_conversion():
    """Example 4: Metric depth conversion."""
    print("=" * 60)
    print("Example 4: Metric Depth Conversion")
    print("=" * 60)
    
    # Note: This example uses DA3-LARGE (relative depth)
    # For real metric depth, use model="metric-large"
    
    estimator = DA3DepthEstimator(model="large-1.1", device="cpu")
    
    result = estimator.process_image(
        "input_images/750_Picacho/Kitchen_2K_test.png",
        "test_output/demo_example4",
        export_format="mini_npz"
    )
    
    if result.success and result.depth_array is not None:
        depth = result.depth_array
        
        # Assume focal length (in pixels)
        # For real applications, extract from EXIF or camera calibration
        focal_length_px = 1000.0
        
        # Convert to metric (this is approximate for relative depth models)
        depth_meters = convert_to_metric_depth(
            depth,
            focal_length_px,
            model_type="relative"  # Use "metric" for DA3METRIC models
        )
        
        print("✅ Depth conversion example:")
        print(f"   Original range: [{depth.min():.3f}, {depth.max():.3f}]")
        print(f"   Metric range: [{depth_meters.min():.3f}, {depth_meters.max():.3f}]")
        print(f"\n   Note: For accurate metric depth, use model='metric-large'")
    else:
        print(f"❌ Failed to load depth array")
    
    print()


def example_5_model_comparison():
    """Example 5: Compare different models."""
    print("=" * 60)
    print("Example 5: Model Comparison")
    print("=" * 60)
    
    models_to_test = ["small", "base", "large-1.1"]
    
    print(f"Testing models: {models_to_test}")
    print("(This may take a few minutes...)\n")
    
    for model_name in models_to_test:
        print(f"Testing {model_name}...")
        
        estimator = DA3DepthEstimator(model=model_name, device="cpu")
        
        result = estimator.process_image(
            "input_images/750_Picacho/Kitchen_2K_test.png",
            f"test_output/demo_example5_{model_name}",
            export_format="mini_npz"
        )
        
        if result.success and result.depth_array is not None:
            depth = result.depth_array
            print(f"   ✅ {model_name}: Shape={depth.shape}, "
                  f"Range=[{depth.min():.3f}, {depth.max():.3f}]")
        else:
            print(f"   ❌ {model_name}: Failed")
    
    print()


def example_6_pipeline_integration():
    """Example 6: Integration with rendering pipeline."""
    print("=" * 60)
    print("Example 6: Rendering Pipeline Integration")
    print("=" * 60)
    
    # Simulate a rendering pipeline workflow
    class MockRenderingPipeline:
        """Mock rendering pipeline for demonstration."""
        
        def __init__(self):
            self.depth_estimator = DA3DepthEstimator(
                model="large-1.1",
                device="cpu"
            )
        
        def process_property(self, image_path, output_dir):
            """Process a property image with depth estimation."""
            output_dir = Path(output_dir)
            
            print(f"   Processing: {Path(image_path).name}")
            
            # Step 1: Estimate depth
            depth_result = self.depth_estimator.process_image(
                image_path,
                output_dir / "depth",
                export_format="mini_npz-depth_vis"
            )
            
            if not depth_result.success:
                print(f"   ❌ Depth estimation failed")
                return None
            
            # Step 2: Load depth for further processing
            depth = depth_result.depth_array
            
            # Step 3: Use depth for zone-based enhancement (mock)
            foreground_mask = depth[0] < depth[0].mean()  # Simple threshold
            background_mask = ~foreground_mask
            
            print(f"   ✅ Depth estimated")
            print(f"      Foreground: {foreground_mask.sum()} pixels")
            print(f"      Background: {background_mask.sum()} pixels")
            print(f"      Saved to: {output_dir / 'depth'}")
            
            return {
                "depth": depth,
                "foreground_mask": foreground_mask,
                "background_mask": background_mask,
                "output_dir": output_dir
            }
    
    # Use the pipeline
    pipeline = MockRenderingPipeline()
    
    result = pipeline.process_property(
        "input_images/750_Picacho/Kitchen_2K_test.png",
        "test_output/demo_example6_pipeline"
    )
    
    if result:
        print(f"\n✅ Pipeline processing complete!")
    
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("DA3 Integration Demo for Transformation Portal")
    print("=" * 60 + "\n")
    
    # Check if test image exists
    test_image = Path("input_images/750_Picacho/Kitchen_2K_test.png")
    if not test_image.exists():
        print(f"❌ Test image not found: {test_image}")
        print("   Please ensure test images are available.")
        return
    
    # Run examples
    try:
        example_1_quick_start()
        example_2_advanced_processing()
        example_3_batch_processing()
        example_4_metric_conversion()
        # example_5_model_comparison()  # Uncomment to test multiple models
        example_6_pipeline_integration()
        
        print("=" * 60)
        print("✅ All examples completed successfully!")
        print("=" * 60)
        print("\nCheck test_output/demo_* directories for results.")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
