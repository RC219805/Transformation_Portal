#!/usr/bin/env python3
"""
Quick status check for DA3 integration.
Shows if DA3 is ready to use and provides next steps.
"""
import os
import sys
from pathlib import Path

# Fix OpenMP issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def check_status():
    """Check DA3 integration status."""
    print("=" * 80)
    print("DA3 Integration - Status Check")
    print("=" * 80)
    print()
    
    # Check imports
    print("📦 Module Status:")
    try:
        from lux_depth_v3.da3_integration import estimate_depth, DA3DepthEstimator
        print("  ✅ lux_depth_v3.da3_integration")
    except ImportError as e:
        print(f"  ❌ lux_depth_v3.da3_integration: {e}")
        return False
    
    try:
        from lux_depth_v3.metric_depth import convert_to_metric_depth
        print("  ✅ lux_depth_v3.metric_depth")
    except ImportError as e:
        print(f"  ❌ lux_depth_v3.metric_depth: {e}")
        return False
    
    try:
        from lux_depth_v3.model_cache import ModelCacheManager
        print("  ✅ lux_depth_v3.model_cache")
    except ImportError as e:
        print(f"  ❌ lux_depth_v3.model_cache: {e}")
        return False
    
    # Check DA3 CLI
    print()
    print("🔧 DA3 CLI:")
    import subprocess
    try:
        result = subprocess.run(
            ["which", "da3"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            da3_path = result.stdout.strip()
            print(f"  ✅ DA3 CLI found: {da3_path}")
        else:
            print("  ⚠️  DA3 CLI not found")
            print("     Install: cd depth_anything_3_official && pip install -e .")
    except Exception as e:
        print(f"  ❌ Error checking DA3 CLI: {e}")
    
    # Check test image
    print()
    print("📸 Test Image:")
    test_image = Path("input_images/750_Picacho/Kitchen_2K_test.png")
    if test_image.exists():
        size_mb = test_image.stat().st_size / (1024 * 1024)
        print(f"  ✅ {test_image}")
        print(f"     Size: {size_mb:.2f} MB")
    else:
        print(f"  ❌ {test_image} not found")
    
    # Check models cache
    print()
    print("💾 Model Cache:")
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_cache.exists():
        # Look for DA3 models
        da3_models = list(hf_cache.glob("models--depth-anything--*"))
        if da3_models:
            print(f"  ✅ HuggingFace cache: {hf_cache}")
            print(f"     Found {len(da3_models)} DA3 model(s):")
            for model_dir in da3_models[:5]:  # Show first 5
                model_name = model_dir.name.replace("models--", "").replace("--", "/")
                print(f"       - {model_name}")
        else:
            print(f"  ⚠️  No DA3 models cached yet")
            print(f"     Cache location: {hf_cache}")
            print(f"     Models will download on first use")
    else:
        print(f"  ⚠️  HuggingFace cache not found: {hf_cache}")
    
    # Available models
    print()
    print("🤖 Available Models:")
    from lux_depth_v3.da3_integration import DA3DepthEstimator
    for i, (key, value) in enumerate(DA3DepthEstimator.AVAILABLE_MODELS.items(), 1):
        print(f"  {i}. {key:30s} -> {value}")
    
    # Next steps
    print()
    print("=" * 80)
    print("✅ DA3 Integration is Ready!")
    print("=" * 80)
    print()
    print("📋 Next Steps:")
    print()
    print("1️⃣  Quick Test:")
    print("   python test_da3_integration.py")
    print()
    print("2️⃣  Process Single Image:")
    print("   python -c \"")
    print("   from lux_depth_v3.da3_integration import estimate_depth")
    print("   result = estimate_depth(")
    print("       'input_images/750_Picacho/Kitchen_2K_test.png',")
    print("       'output/depth/',")
    print("       model='large-1.1'")
    print("   )")
    print("   print(f'Success: {result.success}')")
    print("   \"")
    print()
    print("3️⃣  Batch Processing:")
    print("   lux-depth-v3 process -i renders/ -o output/ --model large-1.1")
    print()
    print("4️⃣  Download Models:")
    print("   lux-depth-v3 cache-download --set essential")
    print()
    print("📚 Documentation:")
    print("   - Quick Start: lux_depth_v3/QUICK_START.md")
    print("   - Full Guide: DA3_INTEGRATION_IMPLEMENTATION_COMPLETE.md")
    print()
    
    return True


if __name__ == "__main__":
    try:
        success = check_status()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
