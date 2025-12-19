#!/usr/bin/env python3
"""
Quick test script for DA3 integration.
Tests the user's example: estimate_depth("input_images/750_Picacho/Kitchen_2K_test.png", "output/depth/", model="large-1.1")
"""
import os
import sys
from pathlib import Path

# Fix OpenMP duplicate library issue on Mac
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add lux_depth_v3 to path
sys.path.insert(0, str(Path(__file__).parent))

from lux_depth_v3.da3_integration import estimate_depth, DA3DepthEstimator


def test_basic_estimate():
    """Test the basic estimate_depth() function."""
    print("=" * 80)
    print("Testing DA3 Integration - Basic estimate_depth()")
    print("=" * 80)
    
    input_path = "input_images/750_Picacho/Kitchen_2K_test.png"
    output_dir = "output/depth/"
    model = "large-1.1"
    
    if not Path(input_path).exists():
        print(f"❌ Test image not found: {input_path}")
        return False
    
    print(f"\n📸 Input: {input_path}")
    print(f"📁 Output: {output_dir}")
    print(f"🤖 Model: {model}")
    print(f"\n⏳ Running depth estimation...")
    
    try:
        result = estimate_depth(
            image_path=input_path,
            output_dir=output_dir,
            model=model,
            device="cpu"  # Use CPU for safety
        )
        
        print(f"\n{'='*80}")
        print("Results:")
        print(f"{'='*80}")
        print(f"Success: {result.success}")
        print(f"Output dir: {result.output_dir}")
        
        if result.success:
            print(f"\n✅ Processing succeeded!")
            
            if result.npz_path:
                print(f"   NPZ file: {result.npz_path}")
                depth = result.depth_array
                if depth is not None:
                    print(f"   Depth shape: {depth.shape}")
                    print(f"   Depth range: [{depth.min():.2f}, {depth.max():.2f}]")
            
            if result.glb_path:
                print(f"   GLB file: {result.glb_path}")
            
            if result.depth_vis_dir:
                print(f"   Depth vis: {result.depth_vis_dir}")
            
            print(f"\n{'='*80}")
            return True
        else:
            print(f"\n❌ Processing failed!")
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            return False
            
    except Exception as e:
        print(f"\n❌ Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_estimator_class():
    """Test the DA3DepthEstimator class directly."""
    print("\n" + "=" * 80)
    print("Testing DA3DepthEstimator class")
    print("=" * 80)
    
    input_path = "input_images/750_Picacho/Kitchen_2K_test.png"
    output_dir = "output/depth_class_test/"
    
    if not Path(input_path).exists():
        print(f"❌ Test image not found: {input_path}")
        return False
    
    print(f"\n📸 Input: {input_path}")
    print(f"📁 Output: {output_dir}")
    print(f"🤖 Available models:")
    for key, value in DA3DepthEstimator.AVAILABLE_MODELS.items():
        print(f"   - {key}: {value}")
    
    try:
        estimator = DA3DepthEstimator(
            model="large-1.1",
            device="cpu",
            verbose=True
        )
        
        print(f"\n⏳ Processing image...")
        result = estimator.process_image(
            input_path=input_path,
            output_dir=output_dir,
            export_format="mini_npz",  # Lightweight format
            process_res=504
        )
        
        print(f"\n{'='*80}")
        print("Results:")
        print(f"{'='*80}")
        print(f"Success: {result.success}")
        
        if result.success:
            print(f"✅ Processing succeeded!")
            
            if result.npz_path:
                print(f"   NPZ: {result.npz_path}")
                depth = result.depth_array
                conf = result.confidence_array
                
                if depth is not None:
                    print(f"   Depth: {depth.shape}, range=[{depth.min():.2f}, {depth.max():.2f}]")
                
                if conf is not None:
                    print(f"   Confidence: {conf.shape}, range=[{conf.min():.2f}, {conf.max():.2f}]")
            
            return True
        else:
            print(f"❌ Processing failed!")
            print(f"STDERR:\n{result.stderr}")
            return False
            
    except Exception as e:
        print(f"\n❌ Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n🚀 DA3 Integration Test Suite\n")
    
    results = []
    
    # Test 1: Basic estimate_depth function
    results.append(("estimate_depth()", test_basic_estimate()))
    
    # Test 2: DA3DepthEstimator class
    results.append(("DA3DepthEstimator", test_estimator_class()))
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {name}")
    
    all_passed = all(r[1] for r in results)
    
    print("=" * 80)
    if all_passed:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
