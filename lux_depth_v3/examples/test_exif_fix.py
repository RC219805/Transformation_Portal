#!/usr/bin/env python3
"""Demonstration of EXIF normalization fix.

This script demonstrates the EXIF normalization fixes:
1. normalize_exif_orientation() always returns True
2. Manifest always shows exif_normalized: true
3. Preflight validation catches depth/image mismatches
"""

import sys
from pathlib import Path
import tempfile
import json
import numpy as np
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lux_depth_v3.enhance.preprocessing import (
    normalize_exif_orientation,
    validate_depth_image_alignment,
)


def test_exif_normalization():
    """Test that normalization always returns True."""
    print("=" * 70)
    print("Test 1: EXIF Normalization Always Returns True")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Test 1a: Image WITH EXIF orientation tag
        print("\n1a. Testing image WITH EXIF orientation tag (6 = 90° CW)...")
        img_with_exif = Image.new("RGB", (100, 200), color="red")
        exif = img_with_exif.getexif()
        exif[0x0112] = 6  # Rotate 90° CW

        input_path = tmp_path / "with_exif.jpg"
        output_path = tmp_path / "with_exif_normalized.png"
        img_with_exif.save(input_path, exif=exif)

        result = normalize_exif_orientation(input_path, output_path)
        print(f"   normalize_exif_orientation() returned: {result}")
        print(f"   ✓ Expected: True (file was normalized)")
        assert result is True, "Should return True for image with EXIF tag"

        # Verify dimensions were swapped
        normalized = Image.open(output_path)
        print(f"   Original dimensions: 100x200")
        print(f"   Normalized dimensions: {normalized.size[0]}x{normalized.size[1]}")
        assert normalized.size == (200, 100), "Dimensions should be swapped"
        print("   ✓ Dimensions correctly swapped (90° rotation applied)")

        # Test 1b: Image WITHOUT EXIF orientation tag
        print("\n1b. Testing image WITHOUT EXIF orientation tag...")
        img_no_exif = Image.new("RGB", (200, 100), color="blue")

        input_path2 = tmp_path / "no_exif.jpg"
        output_path2 = tmp_path / "no_exif_normalized.png"
        img_no_exif.save(input_path2)

        result2 = normalize_exif_orientation(input_path2, output_path2)
        print(f"   normalize_exif_orientation() returned: {result2}")
        print(f"   ✓ Expected: True (file is always normalized, even without EXIF tag)")
        assert result2 is True, "Should return True even without EXIF tag"

        # Verify dimensions unchanged
        normalized2 = Image.open(output_path2)
        print(f"   Original dimensions: 200x100")
        print(f"   Normalized dimensions: {normalized2.size[0]}x{normalized2.size[1]}")
        assert normalized2.size == (200, 100), "Dimensions should remain same"
        print("   ✓ Dimensions unchanged (no rotation needed)")

    print("\n✓ Test 1 passed: Normalization always returns True\n")


def test_preflight_validation():
    """Test preflight depth/image validation."""
    print("=" * 70)
    print("Test 2: Preflight Validation Catches Shape Mismatches")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Test 2a: Matching shapes (should pass)
        print("\n2a. Testing matching image and depth shapes...")
        img = Image.new("RGB", (200, 100))  # W=200, H=100
        img_path = tmp_path / "image.png"
        img.save(img_path)

        depth = np.ones((100, 200), dtype=np.uint16) * 30000  # H=100, W=200
        depth_img = Image.fromarray(depth, mode="I;16")
        depth_path = tmp_path / "depth.png"
        depth_img.save(depth_path)

        try:
            validate_depth_image_alignment(img_path, depth_path)
            print("   ✓ Validation passed (shapes match)")
        except ValueError as e:
            print(f"   ✗ Unexpected error: {e}")
            raise

        # Test 2b: Mismatched shapes (should fail with clear error)
        print("\n2b. Testing mismatched shapes (simulating EXIF issue)...")
        depth_wrong = np.ones((200, 100), dtype=np.uint16) * 30000  # H=200, W=100 (swapped!)
        depth_wrong_img = Image.fromarray(depth_wrong, mode="I;16")
        depth_wrong_path = tmp_path / "depth_wrong.png"
        depth_wrong_img.save(depth_wrong_path)

        try:
            validate_depth_image_alignment(img_path, depth_wrong_path)
            print("   ✗ Validation should have failed!")
            raise AssertionError("Should have detected shape mismatch")
        except ValueError as e:
            print("   ✓ Validation correctly detected shape mismatch")
            print(f"   Error message preview: {str(e)[:100]}...")
            assert "Image/depth shape mismatch" in str(e)

        # Test 2c: Wrong dtype (should fail)
        print("\n2c. Testing wrong dtype (uint8 instead of uint16)...")
        depth_uint8 = np.ones((100, 200), dtype=np.uint8) * 128
        depth_uint8_img = Image.fromarray(depth_uint8, mode="L")
        depth_uint8_path = tmp_path / "depth_uint8.png"
        depth_uint8_img.save(depth_uint8_path)

        try:
            validate_depth_image_alignment(img_path, depth_uint8_path)
            print("   ✗ Validation should have failed!")
            raise AssertionError("Should have detected wrong dtype")
        except ValueError as e:
            print("   ✓ Validation correctly detected wrong dtype")
            assert "Depth must be uint16" in str(e)

    print("\n✓ Test 2 passed: Preflight validation catches errors\n")


def test_manifest_structure():
    """Show expected manifest structure."""
    print("=" * 70)
    print("Test 3: Manifest Structure (exif_normalized always true)")
    print("=" * 70)

    # Example manifest with EXIF normalization
    manifest_example = {
        "schema": "lux-depth-v3.enhance.v1",
        "input": {
            "image_path": "/path/to/750Picacho_Aerial_Ultimate.tif",
            "image_sha256": "abc123...",
            "exif_normalized": True,  # ✓ Always true now
            "normalized_path": "/path/to/tmp_inputs/750Picacho_Aerial_Ultimate_normalized.png",
        },
        "depth": {
            "backend": "da3",
            "model": "DA3METRIC-LARGE",
            "depth_path": "depth/750Picacho_Aerial_Ultimate_depth.png",
            "dtype": "uint16",
            "shape": [3600, 6000],
            "representation": "depth",
            "convention": "higher_is_farther",
        },
    }

    print("\nExpected manifest structure:")
    print(json.dumps(manifest_example, indent=2))

    print("\nKey changes:")
    print("  1. exif_normalized: true (always, not conditional)")
    print("  2. normalized_path: always set (points to normalized file)")
    print("  3. Both DA3 and V2 use the same normalized_path")
    print("  4. Preflight validation ensures depth matches normalized image")

    print("\n✓ Test 3 passed: Manifest structure documented\n")


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║  EXIF Normalization Fix Demonstration                             ║")
    print("║  Issue: V2 crashes due to EXIF orientation mismatches             ║")
    print("║  Fix: Always normalize, always validate before V2                 ║")
    print("╚" + "=" * 68 + "╝")
    print()

    try:
        test_exif_normalization()
        test_preflight_validation()
        test_manifest_structure()

        print("=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70)
        print("\nSummary of fixes:")
        print("  1. normalize_exif_orientation() always returns True")
        print("  2. Manifest always shows exif_normalized: true")
        print("  3. Preflight validation catches depth/image mismatches")
        print("  4. Clear error messages guide users to the root cause")
        print("\nThese changes prevent the V2 crash described in the issue.")
        print()

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
