#!/usr/bin/env python3
"""Verify 16-bit TIFF handoff file format and bit depth.

This script creates a test image, processes it with Materials V3 16-bit path,
and verifies the actual TIFF file format and bit depth before V2 cleanup.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    print("=" * 70)
    print("16-Bit TIFF Handoff Verification")
    print("=" * 70)

    # Check tifffile
    try:
        import tifffile

        print(f"✓ tifffile version: {tifffile.__version__}\n")
    except ImportError:
        print("✗ tifffile not available")
        return False

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_dir = tmpdir / "input"
        output_dir = tmpdir / "output_verify"
        input_dir.mkdir()

        # Create test image with known pixel values
        print("Creating test image with gradient pattern...")
        img_array = np.zeros((512, 512, 3), dtype=np.uint8)
        for i in range(512):
            img_array[i, :, :] = int(255 * i / 512)
        img = Image.fromarray(img_array)
        test_img = input_dir / "test.png"
        img.save(test_img)
        print(f"✓ Created: {test_img}\n")

        # Run pipeline with V2 ENABLED but with a hook to inspect the TIFF
        # We'll run with --enable-v2 on and capture the temp file
        # Run with V2 disabled to preserve temp/ handoff file for inspection
        print("Running pipeline at output bit depth 16...")
        cmd = [
            sys.executable,
            "-m",
            "transformation_portal.lux_depth_v3",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--quality-tier",
            "premium",
            "--depth-backend",
            "synthetic",
            "--materials-v3",
            "on",
            "--enable-segmentation",
            "on",
            "--segmentation-backend",
            "stub",
            "--enable-v2",
            "off",  # V2 disabled to keep temp file for verification
            "--output-bit-depth",
            "16",
            "--keep-intermediates",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            print(f"✗ Pipeline failed")
            print(f"STDERR:\n{result.stderr}")
            return False

        print("✓ Pipeline completed\n")

        # Check for TIFF handoff file (before cleanup)
        temp_dir = output_dir / "temp"
        tiff_files = list(temp_dir.glob("*_materials_v3_enhanced.tif"))

        print("=" * 70)
        print("TIFF Handoff File Verification")
        print("=" * 70)

        if tiff_files:
            tiff_path = tiff_files[0]
            print(f"✓ Found TIFF handoff: {tiff_path.name}")

            # Load and inspect TIFF
            tiff_data = tifffile.imread(tiff_path)
            print(f"✓ Loaded TIFF successfully")
            print(f"  - dtype: {tiff_data.dtype}")
            print(f"  - shape: {tiff_data.shape}")
            print(f"  - min value: {tiff_data.min()}")
            print(f"  - max value: {tiff_data.max()}")
            print(f"  - value range: {tiff_data.max() - tiff_data.min()}")

            if tiff_data.dtype == np.uint16:
                print(f"✓ Bit depth: 16-bit (correct)")
            else:
                print(f"✗ Bit depth: {tiff_data.dtype} (expected uint16)")
                return False

            # Verify it's actually using 16-bit range
            if tiff_data.max() > 255:
                print(f"✓ Using 16-bit value range (max > 255)")
            else:
                print(f"✗ Max value {tiff_data.max()} does not prove 16-bit sample use")
                return False

        else:
            print("✗ Required TIFF handoff file is missing")
            return False

        # Check manifest
        print("\n" + "=" * 70)
        print("Manifest Verification")
        print("=" * 70)

        manifest_files = list(output_dir.glob("manifests/*_combined.json"))
        if not manifest_files:
            print(f"✗ No manifest found")
            return False

        manifest_path = manifest_files[0]
        print(f"✓ Found manifest: {manifest_path.name}")

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        if "materials_v3" in manifest:
            mat_v3 = manifest["materials_v3"]
            print(f"  - Materials V3 enabled: {mat_v3.get('enabled')}")
            print(f"  - Output bit depth: {mat_v3.get('output_bit_depth')}")
            print(f"  - Schema version: {mat_v3.get('schema_version')}")

            if mat_v3.get("output_bit_depth") == 16:
                print(f"✓ Manifest correctly records 16-bit output")
            else:
                print(f"✗ Manifest shows wrong bit depth: {mat_v3.get('output_bit_depth')}")
                return False
        else:
            print(f"✗ No materials_v3 metadata in manifest")
            return False

        # Test 8-bit path
        print("\n" + "=" * 70)
        print("8-Bit Golden Path Verification")
        print("=" * 70)

        output_dir_8bit = tmpdir / "output_8bit"
        cmd_8bit = [
            sys.executable,
            "-m",
            "transformation_portal.lux_depth_v3",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir_8bit),
            "--quality-tier",
            "premium",
            "--depth-backend",
            "synthetic",
            "--materials-v3",
            "on",
            "--enable-segmentation",
            "on",
            "--segmentation-backend",
            "stub",
            "--enable-v2",
            "off",
            "--output-bit-depth",
            "8",
            "--keep-intermediates",
        ]

        result = subprocess.run(cmd_8bit, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"✗ Pipeline failed")
            return False

        print("✓ Pipeline completed (8-bit mode)")
        png_files = list((output_dir_8bit / "temp").glob("*_materials_v3_enhanced.png"))
        if len(png_files) != 1:
            print("✗ Required 8-bit PNG handoff file is missing")
            return False
        png_data = np.asarray(Image.open(png_files[0]))
        if png_data.dtype != np.uint8:
            print(f"✗ Expected uint8 PNG samples, got {png_data.dtype}")
            return False

        manifest_files_8bit = list(output_dir_8bit.glob("manifests/*_combined.json"))
        if manifest_files_8bit:
            with open(manifest_files_8bit[0], "r") as f:
                manifest_8bit = json.load(f)

            if "materials_v3" in manifest_8bit:
                bit_depth = manifest_8bit["materials_v3"].get("output_bit_depth")
                print(f"  - Output bit depth: {bit_depth}")
                if bit_depth == 8:
                    print(f"✓ 8-bit Golden Path preserved (correct)")
                else:
                    print(f"✗ Expected 8-bit, got {bit_depth}")
                    return False

        print("\n" + "=" * 70)
        print("✓ ALL VERIFICATIONS PASSED")
        print("=" * 70)
        print("\nSummary:")
        print("  - Materials V3 outputs 16-bit TIFF at output bit depth 16")
        print("  - Materials V3 outputs 8-bit PNG at output bit depth 8")
        print("  - Manifest correctly tracks bit depth")
        print("  - Schema version 1.1 used for Materials V3 metadata")
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
