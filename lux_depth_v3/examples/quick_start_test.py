"""Quick start script to test all features on a sample image.

This script validates that all integrated DA3 features are working correctly
without requiring the full DA3 package installation.

Usage:
    python lux_depth_v3/examples/quick_start_test.py
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lux_depth_v3 import (
    ModelVariant,
    RefViewStrategy,
    DA3APIConfig,
)
from lux_depth_v3.metric_depth import convert_to_metric_depth, get_depth_statistics
from lux_depth_v3.license import validate_license
from lux_depth_v3.reference_view import select_reference_view


def test_feature_1_model_versioning():
    """Test 1: Model versioning with v1.1 support."""
    print("\n" + "="*70)
    print("TEST 1: Model Versioning (v1.1 Support)")
    print("="*70)
    
    # Test v1.1 model
    variant = ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1
    info = variant.info
    
    print(f"✓ Model: {info.display_name}")
    print(f"✓ Version: {info.version}")
    print(f"✓ Parameters: {info.params}")
    print(f"✓ License: {info.license.value}")
    print(f"✓ Capabilities: {list(info.capabilities.keys())}")
    
    # Test v1.0 legacy model
    legacy = ModelVariant.DA3_NESTED_GIANT_LARGE
    print(f"✓ Legacy model: {legacy.info.display_name} (v{legacy.info.version})")
    
    return True


def test_feature_2_license_validation():
    """Test 2: License validation system."""
    print("\n" + "="*70)
    print("TEST 2: License Validation")
    print("="*70)
    
    # Test non-commercial model
    nc_variant = ModelVariant.DA3_NESTED_GIANT_LARGE_V1_1
    print(f"✓ Non-commercial model: {nc_variant.info.display_name}")
    print(f"  License: {nc_variant.info.license.value}")
    print(f"  Commercial allowed: {nc_variant.info.is_commercial}")
    
    # Test commercial alternative
    commercial = ModelVariant.get_commercial_alternative(nc_variant)
    print(f"✓ Commercial alternative: {commercial.info.display_name}")
    print(f"  License: {commercial.info.license.value}")
    
    # Validate (should warn)
    print("\n  Testing commercial use warning...")
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_license(nc_variant, commercial_use=True, strict=False)
        if len(w) > 0:
            print(f"  ✓ Warning issued: {w[0].category.__name__}")
    
    return True


def test_feature_3_reference_view_selection():
    """Test 3: Reference view selection strategies."""
    print("\n" + "="*70)
    print("TEST 3: Reference View Selection")
    print("="*70)
    
    # Simulate class tokens
    num_views = 5
    class_tokens = np.random.randn(num_views, 768)
    
    strategies = ["saddle_balanced", "saddle_sim_range", "middle", "first"]
    
    for strategy in strategies:
        result = select_reference_view(
            num_views=num_views,
            strategy=strategy,
            class_tokens=class_tokens if "saddle" in strategy else None
        )
        print(f"✓ Strategy '{strategy}': selected view {result.selected_index}")
        if result.metrics:
            print(f"  Metrics: {list(result.metrics.keys())[:3]}...")
    
    return True


def test_feature_4_metric_depth_conversion():
    """Test 4: Metric depth conversion."""
    print("\n" + "="*70)
    print("TEST 4: Metric Depth Conversion")
    print("="*70)
    
    # Create synthetic depth
    depth = np.random.rand(480, 640) * 10.0
    
    # Create intrinsics
    intrinsics = np.array([
        [500.0, 0.0, 320.0],
        [0.0, 500.0, 240.0],
        [0.0, 0.0, 1.0]
    ])
    
    # Test conversion
    result = convert_to_metric_depth(
        depth,
        model_name="DA3METRIC-LARGE",
        intrinsics=intrinsics
    )
    
    print(f"✓ Conversion successful")
    print(f"  Focal length: {result.focal_length_px:.2f} px")
    print(f"  Scale factor: {result.scale_factor:.4f}")
    print(f"  Already metric: {result.already_metric}")
    
    # Get statistics
    stats = get_depth_statistics(result.depth_meters)
    print(f"  Depth range: {stats['min_m']:.2f} - {stats['max_m']:.2f} m")
    print(f"  Mean depth: {stats['mean_m']:.2f} m")
    print(f"  Median depth: {stats['median_m']:.2f} m")
    
    # Test nested model (already metric)
    print("\n  Testing nested model (already metric)...")
    result_nested = convert_to_metric_depth(
        depth,
        model_name="DA3NESTED-GIANT-LARGE-1.1"
    )
    print(f"✓ Already metric: {result_nested.already_metric}")
    
    return True


def test_feature_5_api_config():
    """Test 5: API configuration."""
    print("\n" + "="*70)
    print("TEST 5: DA3 API Configuration")
    print("="*70)
    
    config = DA3APIConfig(
        model_name="da3-large",
        ref_view_strategy=RefViewStrategy.SADDLE_BALANCED,
        use_ray_pose=True,
        infer_gs=False,
        export_format="mini_npz-glb",
        process_res=1024,
        conf_thresh_percentile=40.0
    )
    
    print(f"✓ Config created")
    print(f"  Model: {config.model_name}")
    print(f"  Ref view strategy: {config.ref_view_strategy.value}")
    print(f"  Ray pose: {config.use_ray_pose}")
    print(f"  Export format: {config.export_format}")
    print(f"  Process resolution: {config.process_res}")
    
    # Convert to API kwargs
    kwargs = config.to_api_kwargs()
    print(f"  API kwargs: {len(kwargs)} parameters")
    print(f"  Keys: {list(kwargs.keys())[:5]}...")
    
    return True


def test_feature_6_cli_availability():
    """Test 6: CLI command availability."""
    print("\n" + "="*70)
    print("TEST 6: CLI Availability")
    print("="*70)
    
    try:
        from lux_depth_v3.cli import app
        
        commands = [cmd.name for cmd in app.registered_commands]
        print(f"✓ CLI app loaded")
        print(f"  Registered commands: {len(commands)} commands")
        
        return True
    except Exception as e:
        print(f"✗ CLI import failed: {e}")
        return False


def test_feature_7_all_imports():
    """Test 7: All public API imports."""
    print("\n" + "="*70)
    print("TEST 7: Public API Imports")
    print("="*70)
    
    from lux_depth_v3 import __all__
    
    print(f"✓ __all__ exports: {len(__all__)} items")
    print(f"  Key exports:")
    for item in __all__[:10]:
        print(f"    - {item}")
    if len(__all__) > 10:
        print(f"    ... and {len(__all__) - 10} more")
    
    return True


def main():
    """Run all feature tests."""
    print("\n" + "="*70)
    print("LUX DEPTH V3 - FEATURE INTEGRATION TEST")
    print("="*70)
    print("\nThis script validates that all DA3 features are integrated")
    print("and accessible without requiring the full DA3 package.")
    
    tests = [
        ("Model Versioning", test_feature_1_model_versioning),
        ("License Validation", test_feature_2_license_validation),
        ("Reference View Selection", test_feature_3_reference_view_selection),
        ("Metric Depth Conversion", test_feature_4_metric_depth_conversion),
        ("API Configuration", test_feature_5_api_config),
        ("CLI Availability", test_feature_6_cli_availability),
        ("Public API Imports", test_feature_7_all_imports),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL FEATURES INTEGRATED AND WORKING!")
        print("\nNext steps:")
        print("  1. Run: pytest tests/test_integration_e2e.py")
        print("  2. Try: python lux_depth_v3/examples/test_on_image.py [image.jpg]")
        print("  3. Install DA3: pip install depth-anything-3")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        print("Review the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
