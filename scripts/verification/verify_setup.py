#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify Transformation Portal installation and dependencies.

This script checks:
- Required Python packages
- Optional ML packages
- Model files
- GPU/MPS availability
- Dimension validation

Usage:
    Public compatibility path:
    python scripts/verify_setup.py [--verbose]
"""

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Dict, Tuple


def check_package(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """Check if a package is installed.

    Args:
        package_name: Name of the package (for display)
        import_name: Name to use for import (defaults to package_name)

    Returns:
        Tuple of (is_installed, version_or_error)
    """
    if import_name is None:
        import_name = package_name

    try:
        module = importlib.import_module(import_name)
        version = getattr(module, "__version__", "unknown")
        return True, version
    except ImportError as e:
        return False, str(e)


def check_torch_backend() -> Dict[str, bool]:
    """Check available PyTorch backends (CUDA, MPS, CPU).

    Returns:
        Dictionary of backend availability
    """
    backends = {
        "cpu": True,  # Always available
        "cuda": False,
        "mps": False,
    }

    try:
        import torch

        backends["cuda"] = torch.cuda.is_available()
        backends["mps"] = torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False
    except ImportError:
        # PyTorch is not installed; report 'cuda' and 'mps' as unavailable.
        pass

    return backends


def check_model_files() -> Dict[str, bool]:
    """Check if required model files exist.

    Returns:
        Dictionary of model file status
    """
    repo_root = Path(__file__).resolve().parents[2]

    models = {
        "Depth Anything V2 (CoreML)": repo_root / "DepthAnythingV2SmallF16.mlpackage",
        "Real-ESRGAN-compatible weights": repo_root / "weights" / "RealESRGAN_x4plus.pth",
    }

    return {name: path.exists() for name, path in models.items()}


def verify_dimension_validation():
    """Test dimension validation function."""
    print("\nTesting dimension validation:")
    print("-" * 70)

    # Try to import constants from main module
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
        from transformation_portal.pipelines.lux_render_pipeline import (
            MIN_SD_DIMENSION,
            SD_DIMENSION_MULTIPLE,
            validate_sd_dimensions,
        )

        print(f"Using imported constants: SD_DIMENSION_MULTIPLE={SD_DIMENSION_MULTIPLE}, MIN_SD_DIMENSION={MIN_SD_DIMENSION}")
    except ImportError as e:
        print("✗ Could not import dimension validation constants from main module.")
        print("  Please ensure all dependencies are installed and the source tree is complete.")
        print("  Install core dependencies with: make install-core")
        print(f"  Import error: {e}")
        print("  Aborting dimension validation test to avoid using potentially outdated fallback values.")
        return False

    test_cases = [
        (MIN_SD_DIMENSION, MIN_SD_DIMENSION, True, f"Minimum {MIN_SD_DIMENSION}x{MIN_SD_DIMENSION}"),
        (768, MIN_SD_DIMENSION, True, f"Standard 768x{MIN_SD_DIMENSION}"),
        (1024, 768, True, "Standard 1024x768"),
        (1024, 770, False, f"Invalid 1024x770 (not multiple of {SD_DIMENSION_MULTIPLE})"),
        (800, 600, False, f"Invalid 800x600 (not multiple of {SD_DIMENSION_MULTIPLE})"),
    ]

    try:
        try:
            import typer

            validation_exception = typer.BadParameter
        except ImportError:
            validation_exception = ValueError  # fallback for fallback implementation
        for width, height, should_pass, description in test_cases:
            try:
                result_w, result_h = validate_sd_dimensions(width, height, auto_correct=False)
                status = "✓ PASS" if should_pass else "✗ FAIL (should have raised error)"
                print(f"{status}: {description} -> {result_w}x{result_h}")
            except validation_exception as e:
                status = "✗ FAIL" if should_pass else "✓ PASS"
                print(f"{status}: {description} -> {type(e).__name__}")

        print("\n✓ Dimension validation function is working correctly")
        return True

    except ImportError as e:
        print(f"✗ Could not import validation function: {e}")
        return False


def main():
    """Main verification routine."""
    parser = argparse.ArgumentParser(description="Verify Transformation Portal installation")
    parser.add_argument("--verbose", action="store_true", help="Show detailed error messages")

    args = parser.parse_args()

    print("=" * 70)
    print("TRANSFORMATION PORTAL - INSTALLATION VERIFICATION")
    print("=" * 70)

    # Check required packages
    print("\nRequired Packages:")
    print("-" * 70)

    required_packages = [
        ("numpy", "numpy"),
        ("Pillow", "PIL"),
        ("scipy", "scipy"),
        ("typer", "typer"),
    ]

    required_ok = True
    for pkg_name, import_name in required_packages:
        installed, version = check_package(pkg_name, import_name)
        symbol = "✓" if installed else "✗"
        status = f"{version}" if installed else "NOT INSTALLED"
        print(f"{symbol} {pkg_name:20s} {status}")
        if not installed:
            required_ok = False
            if args.verbose:
                print(f"   Error: {version}")

    # Check optional ML packages
    print("\nOptional ML Packages:")
    print("-" * 70)

    ml_packages = [
        ("torch", "torch"),
        ("diffusers", "diffusers"),
        ("transformers", "transformers"),
        ("controlnet-aux", "controlnet_aux"),
        ("accelerate", "accelerate"),
    ]

    for pkg_name, import_name in ml_packages:
        installed, version = check_package(pkg_name, import_name)
        symbol = "✓" if installed else "○"
        status = f"{version}" if installed else "Not installed (optional)"
        print(f"{symbol} {pkg_name:20s} {status}")
        if not installed and args.verbose:
            print(f"   Note: {version}")

    # Check PyTorch backends
    print("\nPyTorch Backends:")
    print("-" * 70)

    backends = check_torch_backend()
    for backend, available in backends.items():
        symbol = "✓" if available else "○"
        status = "Available" if available else "Not available"
        print(f"{symbol} {backend.upper():20s} {status}")

    if backends["mps"]:
        print("   → Apple Silicon detected - CoreML models recommended for best performance")
    elif backends["cuda"]:
        print("   → CUDA available - GPU acceleration enabled")
    else:
        print("   → CPU only - processing will be slower")

    # Check model files
    print("\nModel Files:")
    print("-" * 70)

    models = check_model_files()
    for model_name, exists in models.items():
        symbol = "✓" if exists else "○"
        status = "Found" if exists else "Not found (run scripts/setup/download_depth_models.py)"
        print(f"{symbol} {model_name:30s} {status}")

    # Test dimension validation
    verify_dimension_validation()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if required_ok:
        print("✓ All required packages are installed")
    else:
        print("✗ Some required packages are missing")
        print("  Run: make install-core")

    ml_installed = sum(1 for pkg_name, import_name in ml_packages if check_package(pkg_name, import_name)[0])
    print(f"○ {ml_installed}/{len(ml_packages)} optional ML packages installed")

    models_found = sum(1 for exists in models.values() if exists)
    print(f"○ {models_found}/{len(models)} model files found")

    if not required_ok:
        print("\n⚠ Installation incomplete - some features may not work")
        return 1

    print("\n✓ Installation verified - ready to use!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
