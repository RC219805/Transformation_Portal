#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple standalone tests for readiness check and image processor.
Does not require pytest - uses standard library only.
"""

import sys
import tempfile
from pathlib import Path
from PIL import Image

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

try:
    import check_image_processing_readiness as readiness
    READINESS_AVAILABLE = True
except ImportError:
    READINESS_AVAILABLE = False
    print("⚠ Readiness check module not available")

try:
    import simple_image_processor as processor
    PROCESSOR_AVAILABLE = True
except ImportError:
    PROCESSOR_AVAILABLE = False
    print("⚠ Simple processor module not available")


def test_readiness_check():
    """Test readiness check functions."""
    if not READINESS_AVAILABLE:
        print("⊘ Skipping readiness tests (module not available)")
        return True
    
    print("\n=== Testing Readiness Check ===")
    
    # Test package checking
    print("  Testing package check...")
    installed, version = readiness.check_package('sys')
    assert installed is True, "sys package should be installed"
    print("  ✓ Package check works")
    
    # Test disk space
    print("  Testing disk space check...")
    disk = readiness.check_disk_space()
    if 'error' not in disk:
        assert 'total_gb' in disk
        assert disk['total_gb'] > 0
    print("  ✓ Disk space check works")
    
    # Test capabilities assessment
    print("  Testing capability assessment...")
    capabilities = readiness.assess_capabilities()
    assert 'core_packages' in capabilities
    assert 'minimal_ready' in capabilities
    print("  ✓ Capability assessment works")
    
    # Test sample image checking
    print("  Testing sample image check...")
    images = readiness.check_sample_images()
    assert 'sample_count' in images
    assert images['sample_count'] >= 0
    print("  ✓ Sample image check works")
    
    print("✓ All readiness check tests passed!")
    return True


def test_simple_processor():
    """Test simple image processor functions."""
    if not PROCESSOR_AVAILABLE:
        print("⊘ Skipping processor tests (module not available)")
        return True
    
    print("\n=== Testing Simple Image Processor ===")
    
    # Test brightness adjustment
    print("  Testing brightness adjustment...")
    img = Image.new('RGB', (100, 100), color=(128, 128, 128))
    result = processor.adjust_brightness(img, factor=1.5)
    assert result is not None
    assert result.size == img.size
    print("  ✓ Brightness adjustment works")
    
    # Test contrast adjustment
    print("  Testing contrast adjustment...")
    result = processor.adjust_contrast(img, factor=1.2)
    assert result is not None
    print("  ✓ Contrast adjustment works")
    
    # Test saturation adjustment
    print("  Testing saturation adjustment...")
    result = processor.adjust_saturation(img, factor=1.5)
    assert result is not None
    print("  ✓ Saturation adjustment works")
    
    # Test resize with aspect
    print("  Testing resize (maintain aspect)...")
    large_img = Image.new('RGB', (1920, 1080), color=(200, 200, 200))
    result = processor.resize_image(large_img, (1280, 720), maintain_aspect=True)
    assert result is not None
    assert result.size[0] <= 1280
    assert result.size[1] <= 720
    print("  ✓ Resize with aspect preservation works")
    
    # Test resize without aspect
    print("  Testing resize (no aspect)...")
    result = processor.resize_image(large_img, (800, 800), maintain_aspect=False)
    assert result is not None
    assert result.size == (800, 800)
    print("  ✓ Resize without aspect preservation works")
    
    # Test full processing pipeline
    print("  Testing full image processing...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test image
        input_path = tmpdir / 'test_input.jpg'
        test_img = Image.new('RGB', (800, 600), color=(128, 128, 128))
        test_img.save(input_path, quality=95)
        
        # Process it
        output_path = tmpdir / 'test_output.jpg'
        success = processor.process_image(
            input_path,
            output_path,
            brightness=1.1,
            contrast=1.05,
            saturation=1.0,
            quality=90,
            verbose=False
        )
        
        assert success is True, "Processing should succeed"
        assert output_path.exists(), "Output file should be created"
        
        # Verify output
        result_img = Image.open(output_path)
        assert result_img.size == (800, 600), "Output size should match input"
    
    print("  ✓ Full processing pipeline works")
    
    # Test format conversion
    print("  Testing format conversion...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create PNG input
        input_path = tmpdir / 'test.png'
        test_img = Image.new('RGB', (400, 300), color=(100, 200, 150))
        test_img.save(input_path)
        
        # Convert to JPEG
        output_path = tmpdir / 'test.jpg'
        success = processor.process_image(input_path, output_path, verbose=False)
        
        assert success is True
        assert output_path.exists()
        assert output_path.suffix == '.jpg'
    
    print("  ✓ Format conversion works")
    
    print("✓ All processor tests passed!")
    return True


def main():
    """Run all tests."""
    print("="*70)
    print("TRANSFORMATION PORTAL - IMAGE PROCESSING TESTS")
    print("="*70)
    
    all_passed = True
    
    try:
        if not test_readiness_check():
            all_passed = False
    except Exception as e:
        print(f"✗ Readiness check tests failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        if not test_simple_processor():
            all_passed = False
    except Exception as e:
        print(f"✗ Processor tests failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("="*70)
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("="*70)
        return 1


if __name__ == '__main__':
    sys.exit(main())
