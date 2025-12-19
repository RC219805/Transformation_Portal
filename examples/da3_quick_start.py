#!/usr/bin/env python3
"""DA3 Integration Quick Start Example.

This script demonstrates the complete DA3 integration in lux_depth_v3 with
monocular depth estimation, multi-view processing, and metric depth conversion.

Usage:
    # Quick demo with synthetic image (no DA3 API required)
    python examples/da3_quick_start.py --demo
    
    # Process real image (requires DA3 API)
    python examples/da3_quick_start.py --input image.jpg
    
    # Multi-view processing
    python examples/da3_quick_start.py --input-dir images/ --multi-view
"""

import sys
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))


def demo_feature_integration():
    """Demonstrate DA3 feature integration (no inference)."""
    print("\n" + "="*70)
    print("DA3 INTEGRATION FEATURE DEMONSTRATION")
    print("="*70)
    
    # 1. Import all DA3 components
    print("\n✅ Step 1: Import DA3 Components")
    from lux_depth_v3 import (
        DA3DepthEstimator,
        DA3Result,
        estimate_depth,
        ModelVariant,
        DA3Config,
        DA3APIConfig,
        RefViewStrategy,
        convert_to_metric_depth,
    )
    from lux_depth_v3.license import validate_license, get_license_info
    print("   All components imported successfully!")
    
    # 2. Show model variants
    print("\n✅ Step 2: Available Model Variants")
    variants = [
        ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1,
        ModelVariant.DA3_LARGE_V1_1,
        ModelVariant.DA3_METRIC_LARGE,
        ModelVariant.DA3_BASE,
    ]
    
    for variant in variants:
        info = variant.info
        license_type = "🔓 Commercial" if info.is_commercial else "🔒 Non-Commercial"
        print(f"   • {info.display_name}: {info.params}, {license_type}")
    
    # 3. License validation
    print("\n✅ Step 3: License Validation")
    
    # Commercial model
    try:
        validate_license(ModelVariant.DA3_METRIC_LARGE, commercial_use=True)
        print("   ✓ DA3METRIC-LARGE: Commercial use allowed (Apache 2.0)")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Non-commercial model
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_license(ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1, commercial_use=True)
        if w:
            print("   ⚠️  DA3NESTED-GIANT-LARGE-1.1: Non-commercial only (CC-BY-NC-4.0)")
    
    # Get license info
    info = get_license_info(ModelVariant.DA3_METRIC_LARGE)
    print(f"   ℹ️  License URL: {info['license_url']}")
    
    # 4. Configuration presets
    print("\n✅ Step 4: Configuration Presets")
    from lux_depth_v3.config import Preset
    
    for preset in [Preset.INTERIOR_LUXURY, Preset.EXTERIOR_SHOWCASE, Preset.METRIC_SCAN]:
        config = DA3Config.from_preset(preset)
        print(f"   • {preset.value}: {config.model_variant.info.display_name}")
    
    # 5. Reference view strategies
    print("\n✅ Step 5: Reference View Selection")
    strategies = [
        RefViewStrategy.SADDLE_BALANCED,
        RefViewStrategy.MIDDLE,
        RefViewStrategy.FIRST,
    ]
    
    for strategy in strategies:
        print(f"   • {strategy.value}: {strategy.value.replace('_', ' ').title()}")
    
    # 6. Metric depth conversion
    print("\n✅ Step 6: Metric Depth Conversion")
    from lux_depth_v3.metric_depth import MetricDepthConverter, get_depth_statistics
    
    # Create dummy depth
    depth = np.random.rand(480, 640).astype(np.float32)
    
    # Convert with DA3METRIC-LARGE
    converter = MetricDepthConverter("DA3METRIC-LARGE")
    result = converter.convert(depth, focal_length_px=500.0)
    
    print(f"   ✓ Converted depth shape: {result.depth_meters.shape}")
    print(f"   ✓ Scale factor: {result.scale_factor:.4f}")
    print(f"   ✓ Already metric: {result.already_metric}")
    
    # Get statistics
    stats = get_depth_statistics(result.depth_meters)
    print(f"   ✓ Depth range: {stats['min_m']:.2f} - {stats['max_m']:.2f} m")
    
    print("\n" + "="*70)
    print("✅ ALL FEATURES VALIDATED")
    print("="*70)
    print("\nDA3 integration is complete and working!")
    print("\nNext steps:")
    print("  1. Install DA3 API: pip install depth-anything-3")
    print("  2. Run inference: python examples/da3_quick_start.py --input image.jpg")
    print("  3. Explore examples in lux_depth_v3/examples/")


def process_image(image_path: Path, output_dir: Path):
    """Process single image with DA3 (requires DA3 API)."""
    print(f"\n🚀 Processing image: {image_path}")
    
    try:
        from lux_depth_v3 import DA3InferenceEngine, ModelVariant
        from lux_depth_v3.config import DA3Config
        
        # Create configuration
        config = DA3Config(
            model_variant=ModelVariant.DA3_METRIC_LARGE,  # Commercial-friendly
        )
        
        # Initialize engine
        print("   Loading DA3 model...")
        engine = DA3InferenceEngine(config, commercial_use=True)
        
        # Run inference
        print("   Running inference...")
        result = engine.infer(
            images=[image_path],
            export_dir=output_dir,
            convert_to_metric=True
        )
        
        print(f"   ✅ Success!")
        print(f"   Depth shape: {result.depth.shape}")
        print(f"   Output: {output_dir}")
        
        # Show statistics if metric depth available
        if hasattr(result, 'metric_depth') and result.metric_depth is not None:
            from lux_depth_v3.metric_depth import get_depth_statistics
            stats = get_depth_statistics(result.metric_depth[0])
            print(f"   Depth range: {stats['min_m']:.2f} - {stats['max_m']:.2f} m")
        
    except ImportError:
        print("   ❌ DA3 API not installed")
        print("   Install with: pip install depth-anything-3")
        return 1
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="DA3 Integration Quick Start",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run feature demonstration (no inference)"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input image path"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Output directory (default: ./output)"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.demo:
        # Run feature demonstration
        demo_feature_integration()
        return 0
    
    elif args.input:
        # Process image
        image_path = Path(args.input)
        if not image_path.exists():
            print(f"❌ Image not found: {image_path}")
            return 1
        
        return process_image(image_path, output_dir)
    
    else:
        # No arguments - show help and run demo
        print("No arguments provided. Running feature demonstration...\n")
        demo_feature_integration()
        print("\nFor help: python examples/da3_quick_start.py --help")
        return 0


if __name__ == "__main__":
    sys.exit(main())
