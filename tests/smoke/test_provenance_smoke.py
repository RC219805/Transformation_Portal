#!/usr/bin/env python
"""Smoke test for provenance capture functionality.

This script validates that provenance capture works end-to-end without
requiring the full pipeline or ML dependencies.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image


def check_exiftool():
    """Check if exiftool is available."""
    try:
        result = subprocess.run(
            ["exiftool", "-ver"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode == 0:
            print(f"✓ exiftool available: {result.stdout.decode().strip()}")
            return True
        else:
            print("✗ exiftool not available")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("✗ exiftool not found in PATH")
        return False


def create_test_tiff(path: Path) -> None:
    """Create a simple test TIFF file."""
    img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    img.save(path, format="TIFF")
    print(f"✓ Created test TIFF: {path}")


def test_provenance_capture():
    """Test provenance capture end-to-end."""
    print("\n=== Provenance Capture Smoke Test ===\n")

    # Check exiftool
    if not check_exiftool():
        print("\n⚠ Skipping test: exiftool not available")
        print("Install with: apt-get install libimage-exiftool-perl (Ubuntu/Debian)")
        return 1

    # Import provenance module
    try:
        from transformation_portal.lux_depth_v3.provenance import (
            PROVENANCE_SCHEMA_VERSION,
            capture_provenance,
        )

        print("✓ Imported provenance module")
    except ImportError as e:
        print(f"✗ Failed to import provenance module: {e}")
        return 1

    # Create temporary test file
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        test_tiff = tmpdir_path / "test.tif"
        create_test_tiff(test_tiff)

        # Capture provenance
        try:
            provenance = capture_provenance(
                image_path=test_tiff,
                config_fingerprint="sha256:test_smoke_test",
                cli_args=["--smoke-test"],
                repo_root=Path.cwd(),
            )
            print("✓ Captured provenance metadata")
        except Exception as e:
            print(f"✗ Failed to capture provenance: {e}")
            return 1

        # Validate required fields
        try:
            provenance.validate_required_fields()
            print("✓ Validated required fields")
        except Exception as e:
            print(f"✗ Validation failed: {e}")
            return 1

        # Write sidecar
        sidecar_path = tmpdir_path / "test_provenance.json"
        try:
            provenance.write_sidecar(sidecar_path)
            print(f"✓ Wrote sidecar: {sidecar_path}")
        except Exception as e:
            print(f"✗ Failed to write sidecar: {e}")
            return 1

        # Verify sidecar content
        try:
            with open(sidecar_path) as f:
                data = json.load(f)

            assert data["schema_version"] == PROVENANCE_SCHEMA_VERSION
            assert data["input"]["file_path"] == str(test_tiff)
            assert data["input"]["file_sha256"]
            assert data["input"]["file_size_bytes"] > 0
            assert data["exif"]
            assert data["toolchain"]["python_version"]
            assert data["toolchain"]["exiftool_version"]
            assert data["ingest_context"]["config_fingerprint"] == "sha256:test_smoke_test"

            print("✓ Verified sidecar content")
            print(f"  - Schema: {data['schema_version']}")
            print(f"  - File SHA256: {data['input']['file_sha256'][:16]}...")
            print(f"  - exiftool: {data['toolchain']['exiftool_version']}")
            print(f"  - Python: {data['toolchain']['python_version'].split()[0]}")

        except Exception as e:
            print(f"✗ Sidecar verification failed: {e}")
            return 1

    print("\n✓ All smoke tests passed!\n")
    return 0


if __name__ == "__main__":
    sys.exit(test_provenance_capture())
