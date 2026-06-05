#!/usr/bin/env python3
"""Validate Depth Pro checkpoint and run basic inference test.

This script verifies:
1. Checkpoint file exists and has correct size
2. SHA-256 hash matches expected value
3. depth-pro package is installed
4. Basic inference works correctly

Usage:
    Public compatibility path:
    python scripts/validate_depth_pro_checkpoint.py [--checkpoint PATH]
"""

import argparse
import hashlib
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))

# Expected values
EXPECTED_SHA256 = "3eb35ca68168ad3d14cb150f8947a4edf85589941661fdb2686259c80685c0ce"
EXPECTED_SIZE_GB_MIN = 1.5
EXPECTED_SIZE_GB_MAX = 2.5
DEFAULT_CHECKPOINT = Path("checkpoints/depth_pro.pt")


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_step(num, text):
    """Print formatted step."""
    print(f"\n[{num}] {text}")


def print_success(text):
    """Print success message."""
    print(f"  ✓ {text}")


def print_error(text):
    """Print error message."""
    print(f"  ✗ {text}")


def check_file_exists(checkpoint_path):
    """Check if checkpoint file exists."""
    print_step(1, "Checking checkpoint file existence")

    if not checkpoint_path.exists():
        print_error(f"Checkpoint not found: {checkpoint_path}")
        print("\n  Download with:")
        print(f"    mkdir -p {checkpoint_path.parent}")
        print("    curl -L https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt \\")
        print(f"      -o {checkpoint_path}")
        return False

    print_success(f"Checkpoint found: {checkpoint_path}")
    return True


def check_file_size(checkpoint_path):
    """Check if checkpoint has expected size."""
    print_step(2, "Checking checkpoint file size")

    size_bytes = checkpoint_path.stat().st_size
    size_gb = size_bytes / (1024**3)
    size_mb = size_bytes / (1024**2)

    print(f"  File size: {size_gb:.2f} GB ({size_mb:.0f} MB)")

    if not (EXPECTED_SIZE_GB_MIN < size_gb < EXPECTED_SIZE_GB_MAX):
        print_error(f"Unexpected size! Expected ~1.9 GB, got {size_gb:.2f} GB")
        print("  File may be corrupted or incomplete.")
        return False

    print_success(f"Size is within expected range ({EXPECTED_SIZE_GB_MIN}-{EXPECTED_SIZE_GB_MAX} GB)")
    return True


def check_sha256(checkpoint_path):
    """Verify checkpoint SHA-256 hash."""
    print_step(3, "Verifying SHA-256 hash")
    print("  This may take 1-2 minutes for a 1.9 GB file...")

    start_time = time.time()
    h = hashlib.sha256()

    with open(checkpoint_path, "rb") as f:
        # Read in 1 MB chunks
        chunk_size = 1024 * 1024
        while chunk := f.read(chunk_size):
            h.update(chunk)

    actual_hash = h.hexdigest()
    elapsed = time.time() - start_time

    print(f"  Computed in {elapsed:.1f}s")
    print(f"  Expected: {EXPECTED_SHA256}")
    print(f"  Actual:   {actual_hash}")

    if actual_hash != EXPECTED_SHA256:
        print_error("SHA-256 mismatch!")
        print("\n  This indicates:")
        print("    - File corruption during download")
        print("    - Wrong checkpoint version")
        print("    - File tampering")
        print("\n  Please re-download the checkpoint:")
        print("    curl -L https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt \\")
        print(f"      -o {checkpoint_path}")
        return False

    print_success("SHA-256 verified - checkpoint is authentic")
    return True


def check_depth_pro_package():
    """Check if depth-pro package is installed."""
    print_step(4, "Checking depth-pro package")

    try:
        import depth_pro

        print_success(f"depth-pro package installed: version {getattr(depth_pro, '__version__', 'unknown')}")
        return True
    except ImportError:
        print_error("depth-pro package not installed")
        print("\n  Install with:")
        print("    ./scripts/setup/install_depth_pro_runtime.sh --skip-verify")
        return False


def run_basic_inference(checkpoint_path):
    """Run basic inference test."""
    print_step(5, "Running basic inference test")

    try:
        # Import required modules
        import numpy as np
        from PIL import Image

        from transformation_portal.stage_graph.stage import StageContext, StageStatus
        from transformation_portal.stage_graph.stages.depth_pro import DepthProStage

        print("  Creating test image (640x480)...")
        test_image = Image.new("RGB", (640, 480), color=(120, 150, 180))

        print("  Initializing DepthProStage...")
        stage = DepthProStage(
            checkpoint_path=checkpoint_path,
            device="cpu",  # Use CPU for compatibility
            strict_validation=True,
        )

        print("  Running inference (this may take 10-30 seconds on CPU)...")
        context = StageContext(artifacts={"image": test_image})

        start_time = time.time()
        result = stage.compute(context)
        inference_time = time.time() - start_time

        if result.status != StageStatus.COMPLETED:
            print_error(f"Inference failed: {result.error}")
            if result.error_traceback:
                print(f"\nTraceback:\n{result.error_traceback}")
            return False

        # Verify outputs
        if "depth_map" not in result.artifacts:
            print_error("Missing depth_map in output")
            return False

        depth_map = result.artifacts["depth_map"]

        # Check depth map properties
        if not isinstance(depth_map, np.ndarray):
            print_error(f"Depth map is not ndarray (got {type(depth_map)})")
            return False

        if depth_map.dtype != np.float32:
            print_error(f"Depth map dtype is not float32 (got {depth_map.dtype})")
            return False

        if depth_map.shape != (480, 640):
            print_error(f"Unexpected depth map shape: {depth_map.shape} (expected (480, 640))")
            return False

        # Verify metric depth properties
        if not np.all(np.isfinite(depth_map)):
            print_error("Depth map contains non-finite values")
            return False

        if depth_map.min() < 0:
            print_error(f"Depth map contains negative values: min={depth_map.min()}")
            return False

        # Verify provenance
        if "depth_provenance" not in result.artifacts:
            print_error("Missing depth_provenance in output")
            return False

        prov = result.artifacts["depth_provenance"]
        if prov.get("status") != "ok":
            print_error(f"Provenance status is not 'ok': {prov.get('status')}")
            return False

        # Success!
        print_success(f"Inference successful in {inference_time:.2f}s")
        print("\n  Depth Statistics:")
        print(f"    Shape:  {depth_map.shape}")
        print(f"    Range:  {depth_map.min():.2f} - {depth_map.max():.2f} meters")
        print(f"    Median: {np.median(depth_map):.2f} meters")
        print(f"    P95:    {np.percentile(depth_map, 95):.2f} meters")

        return True

    except Exception as e:
        print_error(f"Inference test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main validation workflow."""
    parser = argparse.ArgumentParser(description="Validate Depth Pro checkpoint and run basic inference")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Path to checkpoint file (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip inference test (only validate file)",
    )

    args = parser.parse_args()

    print_header("Depth Pro Checkpoint Validation")
    print(f"\nCheckpoint path: {args.checkpoint.absolute()}")

    # Run validation steps
    all_passed = True

    # Step 1: File exists
    if not check_file_exists(args.checkpoint):
        return 1

    # Step 2: File size
    if not check_file_size(args.checkpoint):
        all_passed = False

    # Step 3: SHA-256
    if not check_sha256(args.checkpoint):
        all_passed = False

    # Step 4: Package installed
    if not check_depth_pro_package():
        all_passed = False
        # Can't run inference without package
        args.skip_inference = True

    # Step 5: Basic inference (optional)
    if not args.skip_inference:
        if not run_basic_inference(args.checkpoint):
            all_passed = False
    else:
        print_step(5, "Skipping inference test (--skip-inference)")

    # Final summary
    print_header("Validation Summary")

    if all_passed:
        print("\n✅ All validation checks passed!")
        print("\nYour Depth Pro checkpoint is ready to use.")
        print("\nExample usage:")
        print("  python -m transformation_portal.lux_depth_v3 \\")
        print("    --input-dir ./images \\")
        print("    --output-dir ./output \\")
        print("    --preset depth-pro-example \\")
        print("    --non-commercial-ok \\")
        print("    --accept-apple-depth-pro-research-license")
        return 0
    else:
        print("\n❌ Some validation checks failed.")
        print("\nPlease resolve the issues above before using the checkpoint.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
