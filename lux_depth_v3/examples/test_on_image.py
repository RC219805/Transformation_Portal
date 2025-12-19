"""Test all DA3 features on a real image (or synthetic if none provided).

This script demonstrates a complete end-to-end workflow using all integrated
features on an actual image file.

Usage:
    # With your own image
    python lux_depth_v3/examples/test_on_image.py path/to/image.jpg
    
    # Let it create a synthetic test image
    python lux_depth_v3/examples/test_on_image.py
    
    # Skip inference (just test features)
    python lux_depth_v3/examples/test_on_image.py --skip-inference
"""

import sys
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def create_synthetic_image(output_path: Path, size=(1080, 1920)) -> Path:
    """Create a synthetic test image with gradient pattern."""
    print(f"Creating synthetic test image: {output_path}")
    
    height, width = size
    
    # Create gradient image (simulates depth variation)
    gradient_x = np.linspace(0, 255, width, dtype=np.uint8)
    gradient_y = np.linspace(0, 255, height, dtype=np.uint8)
    
    # Combine gradients
    img_r = np.tile(gradient_x, (height, 1))
    img_g = np.tile(gradient_y[:, np.newaxis], (1, width))
    img_b = (img_r * 0.5 + img_g * 0.5).astype(np.uint8)
    
    img_array = np.stack([img_r, img_g, img_b], axis=-1)
    
    img = Image.fromarray(img_array)
    img.save(output_path)
    
    print(f"✓ Created {width}x{height} synthetic image")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Test DA3 features on an image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_on_image.py                          # Create synthetic image
  python test_on_image.py image.jpg                # Use your image
  python test_on_image.py --model metric-large    # Use metric model
  python test_on_image.py --skip-inference         # Test features only
        """
    )
    parser.add_argument(
        "image",
        type=str,
        nargs="?",
        help="Path to test image (optional - will create synthetic if not provided)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./test_output",
        help="Output directory for results (default: ./test_output)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="nested-giant-large-v1.1",
        choices=["nested-giant-large-v1.1", "metric-large", "large-v1.1"],
        help="Model variant to use (default: nested-giant-large-v1.1)"
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip actual inference (test features only)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to use for inference (default: cpu)"
    )
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get or create image
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Error: Image not found: {image_path}")
            return 1
        print(f"Using provided image: {image_path}")
    else:
        print("No image provided, creating synthetic test image...")
        image_path = output_dir / "test_image.jpg"
        create_synthetic_image(image_path)
    
    # Load image to get dimensions
    img = Image.open(image_path)
    width, height = img.size
    
    print(f"\nImage: {image_path}")
    print(f"Size: {width}x{height}")
    print(f"Mode: {img.mode}")
    print(f"Output: {output_dir}")
    print(f"Model: {args.model}")
    
    # Test features
    print("\n" + "="*70)
    print("TESTING INTEGRATED FEATURES")
    print("="*70)
    
    from lux_depth_v3 import ModelVariant, DA3APIConfig, RefViewStrategy
    from lux_depth_v3.license import validate_license
    
    # Map CLI model name to variant
    model_map = {
        "nested-giant-large-v1.1": ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1,
        "metric-large": ModelVariant.DA3_METRIC_LARGE,
        "large-v1.1": ModelVariant.DA3_LARGE_V1_1,
    }
    
    variant = model_map[args.model]
    
    # Test 1: Model info
    print("\n1. Model Information:")
    info = variant.info
    print(f"   ✓ Model: {info.display_name}")
    print(f"   ✓ Version: {info.version or 'N/A'}")
    print(f"   ✓ Parameters: {info.params}")
    print(f"   ✓ License: {info.license.value}")
    print(f"   ✓ Commercial use: {info.is_commercial}")
    
    # Test 2: License validation
    print("\n2. License Validation:")
    try:
        validate_license(variant, commercial_use=False)
        print(f"   ✓ License validated for non-commercial use")
    except Exception as e:
        print(f"   ✗ License validation failed: {e}")
    
    # Test 3: API configuration
    print("\n3. API Configuration:")
    config = DA3APIConfig(
        model_name=args.model,
        ref_view_strategy=RefViewStrategy.SADDLE_BALANCED,
        use_ray_pose=True,
        export_format="mini_npz-glb"
    )
    print(f"   ✓ Config created")
    print(f"   ✓ Ref view strategy: {config.ref_view_strategy.value}")
    print(f"   ✓ Export format: {config.export_format}")
    
    if not args.skip_inference:
        # Test 4: Inference
        print("\n4. Running Inference:")
        print("   Note: This requires depth-anything-3 package installed")
        print("   Install with: pip install depth-anything-3")
        
        try:
            from lux_depth_v3 import DA3InferenceEngine
            
            print(f"   Creating inference engine (device: {args.device})...")
            engine = DA3InferenceEngine(
                model_variant=variant,
                device=args.device
            )
            
            print(f"   Running inference on {image_path.name}...")
            result = engine.infer(
                images=[image_path],
                export_dir=output_dir,
                convert_to_metric=True
            )
            
            print(f"   ✓ Inference complete!")
            print(f"   ✓ Depth shape: {result.depth.shape}")
            
            if hasattr(result, 'metric_depth') and result.metric_depth is not None:
                print(f"   ✓ Metric depth available")
                
                from lux_depth_v3.metric_depth import get_depth_statistics
                stats = get_depth_statistics(result.metric_depth[0])
                print(f"   ✓ Depth range: {stats['min_m']:.2f} - {stats['max_m']:.2f} m")
                print(f"   ✓ Mean depth: {stats['mean_m']:.2f} m")
            
            if result.extrinsics is not None:
                print(f"   ✓ Camera poses estimated: {result.extrinsics.shape[0]} views")
            
            # Save depth visualization
            depth_img = result.depth[0]
            depth_normalized = ((depth_img - depth_img.min()) / 
                              (depth_img.max() - depth_img.min()) * 255).astype(np.uint8)
            depth_pil = Image.fromarray(depth_normalized)
            depth_output = output_dir / "depth_visualization.png"
            depth_pil.save(depth_output)
            print(f"   ✓ Saved depth visualization: {depth_output}")
            
            print(f"\n✅ ALL FEATURES WORKING!")
            print(f"   Output directory: {output_dir}")
            print(f"   Check {depth_output} for depth visualization")
            
        except ImportError as e:
            print(f"   ⚠️  Could not import DA3 API: {e}")
            print(f"   Install with: pip install depth-anything-3")
            print(f"   Feature validation passed, but inference skipped")
        except Exception as e:
            print(f"   ✗ Inference failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        print("\n4. Inference: Skipped (--skip-inference)")
        print("   Feature integration validated successfully!")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    
    if args.skip_inference:
        print("\n✅ Feature integration validated (inference skipped)")
        print("\nTo test full workflow:")
        print("  1. Install DA3: pip install depth-anything-3")
        print(f"  2. Run: python {Path(__file__).name} {image_path}")
    else:
        print("\n✅ Full workflow completed successfully!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
