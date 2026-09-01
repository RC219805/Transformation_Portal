#!/usr/bin/env python3
"""Validate Lux Depth V3 16-bit output path behavior.

Validates:
1. Materials V3 outputs 16-bit TIFF at output bit depth 16
2. Materials V3 outputs 8-bit PNG at output bit depth 8 (Golden Path)
3. Bit depth tracking in manifest
4. File format and dtype verification
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np


def validate_encoded_image(path: Path, expected_bits: int) -> bool:
    """Reopen an emitted image and verify its actual sample encoding."""
    if not path.is_file():
        print(f"✗ Expected encoded image is missing: {path}")
        return False
    if expected_bits == 16:
        import tifffile

        pixels = tifffile.imread(path)
        valid = pixels.dtype == np.uint16 and int(pixels.max()) > 255
    else:
        from PIL import Image

        pixels = np.asarray(Image.open(path))
        valid = pixels.dtype == np.uint8
    if not valid:
        print(f"✗ {path} has dtype={pixels.dtype}, max={int(pixels.max())}; expected {expected_bits}-bit samples")
        return False
    print(f"✓ Reopened {path.name}: dtype={pixels.dtype}, max={int(pixels.max())}")
    return True


def create_test_image(output_path: Path, size=(512, 512)):
    """Create a simple test image."""
    from PIL import Image

    # Create gradient image
    img_array = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for i in range(size[1]):
        img_array[i, :, :] = int(255 * i / size[1])

    img = Image.fromarray(img_array)
    img.save(output_path)
    print(f"✓ Created test image: {output_path}")


def validate_16bit_path():
    """Validate the canonical 16-bit output path."""
    print("\n=== Validation 1: 16-Bit TIFF Path ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_dir = tmpdir / "input"
        output_dir = tmpdir / "output_16bit"
        input_dir.mkdir()

        # Create test image
        test_img = input_dir / "test.png"
        create_test_image(test_img)

        # Run the pipeline with canonical 16-bit output.
        import subprocess

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
            "synthetic",  # Use synthetic backend for fast testing
            "--materials-v3",
            "on",
            "--enable-segmentation",
            "on",
            "--segmentation-backend",
            "stub",
            "--enable-v2",
            "off",  # Disable V2 for faster test
            "--output-bit-depth",
            "16",
            "--keep-intermediates",
        ]

        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"✗ Pipeline failed with return code {result.returncode}")
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            return False

        print(f"✓ Pipeline completed successfully")

        # Check for 16-bit TIFF handoff file
        temp_dir = output_dir / "temp"
        if not temp_dir.exists():
            print(f"✗ Temp directory not found: {temp_dir}")
            return False

        tiff_files = list(temp_dir.glob("*_materials_v3_enhanced.tif"))
        if not tiff_files:
            print(f"✗ No 16-bit TIFF handoff file found in {temp_dir}")
            print(f"Files in temp dir: {list(temp_dir.iterdir())}")
            return False
        if not validate_encoded_image(tiff_files[0], 16):
            return False

        # Check manifest for bit depth tracking
        manifest_files = list(output_dir.glob("manifests/*_combined.json"))
        if not manifest_files:
            print(f"✗ No manifest files found in {output_dir}")
            return False

        manifest_path = manifest_files[0]
        print(f"✓ Found manifest: {manifest_path}")

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        # Check Materials V3 bit depth
        if "materials_v3" in manifest:
            mat_v3 = manifest["materials_v3"]
            bit_depth = mat_v3.get("output_bit_depth")
            if bit_depth == 16:
                print(f"✓ Materials V3 output_bit_depth = 16 (correct)")
            else:
                print(f"✗ Materials V3 output_bit_depth = {bit_depth} (expected 16)")
                return False

            schema_version = mat_v3.get("schema_version")
            if schema_version == "1.1":
                print(f"✓ Materials V3 schema_version = 1.1 (correct)")
            else:
                print(f"✗ Materials V3 schema_version = {schema_version} (expected 1.1)")
                return False
        else:
            print(f"✗ No materials_v3 metadata in manifest")
            return False

        print("✓ Validation 1 PASSED: 16-bit path works correctly\n")
        return True


def validate_8bit_golden_path():
    """Validate canonical 8-bit PNG output."""
    print("\n=== Validation 2: 8-Bit PNG Golden Path ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_dir = tmpdir / "input"
        output_dir = tmpdir / "output_8bit"
        input_dir.mkdir()

        # Create test image
        test_img = input_dir / "test.png"
        create_test_image(test_img)

        # Run the pipeline with canonical 8-bit output.
        import subprocess

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
            "off",
            "--output-bit-depth",
            "8",
            "--keep-intermediates",
        ]

        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"✗ Pipeline failed with return code {result.returncode}")
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            return False

        print(f"✓ Pipeline completed successfully")

        png_files = list((output_dir / "temp").glob("*_materials_v3_enhanced.png"))
        if len(png_files) != 1 or not validate_encoded_image(png_files[0], 8):
            return False

        # Check manifest for bit depth tracking
        manifest_files = list(output_dir.glob("manifests/*_combined.json"))
        if not manifest_files:
            print(f"✗ No manifest files found in {output_dir}")
            return False

        manifest_path = manifest_files[0]
        print(f"✓ Found manifest: {manifest_path}")

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        # Check Materials V3 bit depth
        if "materials_v3" in manifest:
            mat_v3 = manifest["materials_v3"]
            bit_depth = mat_v3.get("output_bit_depth")
            if bit_depth == 8:
                print(f"✓ Materials V3 output_bit_depth = 8 (correct)")
            else:
                print(f"✗ Materials V3 output_bit_depth = {bit_depth} (expected 8)")
                return False
        else:
            print(f"✗ No materials_v3 metadata in manifest")
            return False

        print("✓ Validation 2 PASSED: 8-bit Golden Path preserved\n")
        return True


def validate_bit_depth_tracking():
    """Validate V2 bit depth tracking in manifest."""
    print("\n=== Validation 3: V2 Bit Depth Tracking ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_dir = tmpdir / "input"
        output_dir = tmpdir / "output_v2"
        input_dir.mkdir()

        # Create test image
        test_img = input_dir / "test.png"
        create_test_image(test_img)

        # Run the pipeline with V2 enabled and canonical 16-bit output.
        import subprocess

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
            "on",
            "--v2-preset",
            "default",  # Use default preset
            "--output-bit-depth",
            "16",
        ]

        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            print(f"✗ Pipeline failed with return code {result.returncode}")
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            return False

        print(f"✓ Pipeline completed successfully")

        # Check manifest for V2 bit depth tracking
        manifest_files = list(output_dir.glob("manifests/*_combined.json"))
        if not manifest_files:
            print(f"✗ No manifest files found in {output_dir}/manifests")
            print(f"Output structure: {list(output_dir.rglob('*.json'))}")
            return False

        manifest_path = manifest_files[0]
        print(f"✓ Found manifest: {manifest_path}")

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        # Check V2 bit depth
        if "v2" in manifest:
            v2_meta = manifest["v2"]
            input_bit_depth = v2_meta.get("input_bit_depth")
            output_bit_depth = v2_meta.get("output_bit_depth")

            if input_bit_depth == 16:
                print(f"✓ V2 input_bit_depth = 16 (correct)")
            else:
                print(f"✗ V2 input_bit_depth = {input_bit_depth} (expected 16)")
                return False

            if output_bit_depth == 16:
                print(f"✓ V2 output_bit_depth = 16 (correct)")
            else:
                print(f"✗ V2 output_bit_depth = {output_bit_depth} (expected 16)")
                return False
            output_paths = [Path(path) for path in (v2_meta.get("output_paths") or [])]
            if not output_paths:
                print("✗ V2 manifest contains no emitted output path")
                return False
            resolved_outputs = [path if path.is_absolute() else output_dir / path for path in output_paths]
            if not any(validate_encoded_image(path, 16) for path in resolved_outputs):
                return False
        else:
            print(f"✗ No v2 metadata in manifest")
            return False

        print("✓ Validation 3 PASSED: V2 bit depth tracking works correctly\n")
        return True


def main():
    """Run all validations."""
    print("=" * 70)
    print("16-Bit Output Path Implementation Validation")
    print("=" * 70)

    # Import check
    try:
        import tifffile

        print(f"✓ tifffile available: version {tifffile.__version__}")
    except ImportError:
        print("✗ tifffile not available - install with: pip install tifffile")
        return False

    results = []

    # Run validations
    results.append(("16-bit TIFF path", validate_16bit_path()))
    results.append(("8-bit PNG Golden Path", validate_8bit_golden_path()))
    results.append(("V2 bit depth tracking", validate_bit_depth_tracking()))

    # Print summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(result[1] for result in results)
    print("=" * 70)
    if all_passed:
        print("✓ ALL VALIDATIONS PASSED")
        return True
    else:
        print("✗ SOME VALIDATIONS FAILED")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
