#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Processing Readiness Check for Transformation Portal.

This script provides a comprehensive assessment of the current environment's
readiness for image processing operations. It categorizes functionality into
tiers and provides actionable next steps.

Usage:
    .venv/bin/python scripts/check_image_processing_readiness.py [--verbose] [--quick-start]

Features:
    - Tiered capability assessment (Minimal, Standard, Full)
    - Clear actionable recommendations
    - Quick start guide for immediate processing
    - Disk space and dependency checks
    - Sample data verification
"""

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple


class Colors:
    """ANSI color codes for terminal output."""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def colored(text: str, color: str) -> str:
    """Return colored text for terminal output."""
    return f"{color}{text}{Colors.ENDC}"


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
    except ImportError:
        return False, "Not installed"


def check_disk_space() -> Dict[str, any]:
    """Check available disk space.

    Returns:
        Dictionary with disk space information
    """
    try:
        total, used, free = shutil.disk_usage("/")

        # Convert to GB
        total_gb = total / (1024**3)
        used_gb = used / (1024**3)
        free_gb = free / (1024**3)
        used_percent = (used / total) * 100

        return {
            "total_gb": total_gb,
            "used_gb": used_gb,
            "free_gb": free_gb,
            "used_percent": used_percent,
            "sufficient": free_gb >= 5.0,  # Need at least 5GB for ML packages
        }
    except Exception as e:
        return {"error": str(e), "sufficient": False}


def check_sample_images() -> Dict[str, any]:
    """Check for sample images.

    Returns:
        Dictionary with sample image status
    """
    repo_root = Path(__file__).resolve().parents[2]
    sample_dir = repo_root / "data" / "sample_images"
    input_dir = repo_root / "input_images"

    # Count image files
    image_extensions = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}

    sample_images = []
    if sample_dir.exists():
        sample_images = [f for f in sample_dir.rglob("*") if f.is_file() and f.suffix.lower() in image_extensions]

    input_images = []
    if input_dir.exists():
        input_images = [f for f in input_dir.rglob("*") if f.is_file() and f.suffix.lower() in image_extensions]

    return {
        "sample_dir_exists": sample_dir.exists(),
        "input_dir_exists": input_dir.exists(),
        "sample_count": len(sample_images),
        "input_count": len(input_images),
        "total_count": len(sample_images) + len(input_images),
        "has_images": len(sample_images) + len(input_images) > 0,
    }


def check_ffmpeg() -> Tuple[bool, str]:
    """Check if FFmpeg is installed.

    Returns:
        Tuple of (is_installed, version_or_error)
    """
    try:
        import subprocess

        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # Extract version from first line
            version_line = result.stdout.split("\n")[0]
            return True, version_line
        else:
            return False, "FFmpeg found but failed to run"
    except FileNotFoundError:
        return False, "Not installed"
    except Exception as e:
        return False, str(e)


def assess_capabilities() -> Dict[str, any]:
    """Assess current processing capabilities.

    Returns:
        Dictionary with capability tiers and recommendations
    """
    # Check core packages
    core_packages = {
        "numpy": check_package("numpy", "numpy")[0],
        "Pillow": check_package("Pillow", "PIL")[0],
        "scipy": check_package("scipy", "scipy")[0],
        "PyYAML": check_package("PyYAML", "yaml")[0],
        "typer": check_package("typer", "typer")[0],
        "tqdm": check_package("tqdm", "tqdm")[0],
    }

    # Check supported ML packages. The external ``realesrgan`` package is
    # hard-blocked by dependency policy; upscaling must use governed/local
    # backends or Pillow fallback instead of ad hoc package installation.
    ml_packages = {
        "torch": check_package("torch", "torch")[0],
        "diffusers": check_package("diffusers", "diffusers")[0],
        "transformers": check_package("transformers", "transformers")[0],
        "controlnet_aux": check_package("controlnet-aux", "controlnet_aux")[0],
    }

    # Check image processing packages
    image_packages = {
        "tifffile": check_package("tifffile", "tifffile")[0],
        "imagecodecs": check_package("imagecodecs", "imagecodecs")[0],
        "scikit-image": check_package("scikit-image", "skimage")[0],
        "opencv": check_package("opencv-python", "cv2")[0],
    }

    # Determine capability tier
    minimal_ready = core_packages["numpy"] and core_packages["Pillow"]
    standard_ready = minimal_ready and core_packages["scipy"] and image_packages["tifffile"]
    full_ready = standard_ready and ml_packages["torch"] and ml_packages["diffusers"]

    return {
        "core_packages": core_packages,
        "ml_packages": ml_packages,
        "image_packages": image_packages,
        "minimal_ready": minimal_ready,
        "standard_ready": standard_ready,
        "full_ready": full_ready,
    }


def print_tier_status(capabilities: Dict) -> None:
    """Print capability tier status."""
    print("\n" + "=" * 70)
    print(colored("CAPABILITY ASSESSMENT", Colors.BOLD))
    print("=" * 70)

    # Minimal Tier
    print("\n" + colored("📦 MINIMAL TIER", Colors.OKBLUE))
    print("   Basic image operations (resize, format conversion, metadata)")
    if capabilities["minimal_ready"]:
        print(colored("   ✓ READY", Colors.OKGREEN))
        print("   → Can process images with basic operations")
    else:
        print(colored("   ✗ NOT READY", Colors.FAIL))
        print("   → Install: make install-core")

    # Standard Tier
    print("\n" + colored("📦 STANDARD TIER", Colors.OKBLUE))
    print("   Professional image processing (color grading, batch, 16-bit TIFF)")
    if capabilities["standard_ready"]:
        print(colored("   ✓ READY", Colors.OKGREEN))
        print("   → Can process with LUTs, batch operations, metadata preservation")
    else:
        print(colored("   ✗ NOT READY", Colors.WARNING))
        missing = []
        if not capabilities["core_packages"]["scipy"]:
            missing.append("scipy")
        if not capabilities["image_packages"]["tifffile"]:
            missing.append("tifffile")
        print(f"   → Missing standard packages: {', '.join(missing)}")
        print("   → Install: make install-core")

    # Full Tier
    print("\n" + colored("📦 FULL TIER", Colors.OKBLUE))
    print("   AI-powered processing (depth maps, upscaling, enhancement)")
    if capabilities["full_ready"]:
        print(colored("   ✓ READY", Colors.OKGREEN))
        print("   → Can use all AI/ML pipelines")
    else:
        print(colored("   ✗ NOT READY", Colors.WARNING))
        ml_count = sum(capabilities["ml_packages"].values())
        print(f"   → {ml_count}/{len(capabilities['ml_packages'])} supported ML packages installed")
        print("   → Install: make install-ml-core")
        print("   → Advanced Apple Silicon bootstrap: ./scripts/bootstrap/install_ml_stack.sh --profile core-cpu")
        print("   → Note: Requires ~5GB disk space")


def print_available_operations(capabilities: Dict) -> None:
    """Print operations available with current setup."""
    print("\n" + "=" * 70)
    print(colored("AVAILABLE OPERATIONS", Colors.BOLD))
    print("=" * 70)

    operations = []

    if capabilities["minimal_ready"]:
        operations.extend(
            [
                ("✓", "Image format conversion (JPG, PNG, TIFF)", Colors.OKGREEN),
                ("✓", "Basic resize and crop operations", Colors.OKGREEN),
                ("✓", "EXIF/IPTC metadata reading", Colors.OKGREEN),
            ]
        )

    if capabilities["standard_ready"]:
        operations.extend(
            [
                ("✓", "LUT-based color grading", Colors.OKGREEN),
                ("✓", "16-bit TIFF batch processing", Colors.OKGREEN),
                ("✓", "Professional metadata preservation", Colors.OKGREEN),
                ("✓", "Exposure, contrast, saturation adjustments", Colors.OKGREEN),
            ]
        )

    if capabilities["image_packages"]["opencv"]:
        operations.append(("✓", "Advanced image filters and effects", Colors.OKGREEN))

    if capabilities["full_ready"]:
        operations.extend(
            [
                ("✓", "AI-powered depth estimation", Colors.OKGREEN),
                ("✓", "Stable Diffusion enhancement", Colors.OKGREEN),
                ("✓", "Governed AI upscaling backends and Pillow fallback", Colors.OKGREEN),
                ("✓", "ControlNet refinement", Colors.OKGREEN),
                ("✓", "Material Response processing", Colors.OKGREEN),
            ]
        )
    else:
        operations.extend(
            [
                ("○", "AI-powered depth estimation (requires torch)", Colors.WARNING),
                ("○", "Stable Diffusion enhancement (requires ML packages)", Colors.WARNING),
                ("○", "Governed AI upscaling (external Real-ESRGAN package unsupported)", Colors.WARNING),
            ]
        )

    for symbol, operation, color in operations:
        print(colored(f"{symbol} {operation}", color))


def print_quick_start_guide(capabilities: Dict, images: Dict) -> None:
    """Print quick start guide based on current capabilities."""
    print("\n" + "=" * 70)
    print(colored("🚀 QUICK START GUIDE", Colors.BOLD))
    print("=" * 70)

    if not capabilities["minimal_ready"]:
        print("\n" + colored("⚠ Install core packages first:", Colors.WARNING))
        print("   make install-core")
        print("   make check-environment")
        print("\n   Then run this script again to see available operations.")
        return

    # Check for images
    if not images["has_images"]:
        print("\n" + colored("📥 Step 1: Get Sample Images", Colors.OKBLUE))
        print("   No images found. Choose one:")
        print("   ")
        print("   a) Download samples:")
        print("      .venv/bin/python scripts/download_samples.py")
        print("   ")
        print("   b) Use your own images:")
        print("      cp ~/Downloads/my_image.jpg input_images/")
    else:
        print("\n" + colored("✓ Images found:", Colors.OKGREEN))
        if images["sample_count"] > 0:
            print(f"   {images['sample_count']} sample images in data/sample_images/")
        if images["input_count"] > 0:
            print(f"   {images['input_count']} images in input_images/")

    # Show appropriate processing examples
    print("\n" + colored("🎨 Step 2: Process Images", Colors.OKBLUE))

    if capabilities["minimal_ready"] and not capabilities["standard_ready"]:
        print("   With current setup (Minimal), you can:")
        print("   ")
        print("   # Convert and resize images")
        print(
            "   python -c \"from PIL import Image; img = Image.open('input_images/my_image.jpg'); img.resize((1920, 1080)).save('output.jpg')\""
        )
        print("   ")
        print("   Upgrade to Standard tier for professional workflows:")
        print("   make install-core")
        print("   # If packages are still missing, inspect requirements/README.md before installing ad hoc.")

    elif capabilities["standard_ready"] and not capabilities["full_ready"]:
        print("   With current setup (Standard), try:")
        print("   ")
        print("   # Basic batch processing (without AI)")
        print("   # Create a simple processing script using Pillow + scipy")
        print("   ")
        print("   Upgrade to Full tier for AI-powered processing:")
        print("   make install-ml-core")
        print("   # Advanced Apple Silicon bootstrap: ./scripts/bootstrap/install_ml_stack.sh --profile core-cpu")

    elif capabilities["full_ready"]:
        print("   With current setup (Full), you can use all pipelines:")
        print("   ")
        print("   # AI-powered render enhancement")
        print(
            "   .venv/bin/lux_render --input-glob \"input_images/*.tiff\" "
            "--out output/lux_render --prompt \"luxury interior\""
        )
        print("   ")
        print("   # Lux Depth V3 APEX processing")
        print(
            "   .venv/bin/lux-depth-v3 --input-dir input_images --output-dir output/lux_depth_v3 "
            "--quality-tier apex --model-key da3-metric"
        )
        print("   ")
        print("   # Batch TIFF processing")
        print("   .venv/bin/luxury-tiff-batch input_images/ output/tiff_lux --preset signature")

    print("\n" + colored("📖 Step 3: Explore Documentation", Colors.OKBLUE))
    print("   README.md - Feature overview and examples")
    print("   docs/guides/IMAGE_PROCESSING_READINESS.md - Capability tiers and install paths")
    print("   docs/cli/LUX_DEPTH_V3_CLI_GUIDE.md - Lux Depth V3 CLI examples")
    print("   docs/governance/DOCUMENTATION_MAP.md - Current documentation navigation")


def print_recommendations(disk: Dict, capabilities: Dict) -> None:
    """Print actionable recommendations."""
    print("\n" + "=" * 70)
    print(colored("💡 RECOMMENDATIONS", Colors.BOLD))
    print("=" * 70)

    recommendations = []

    # Disk space
    if "free_gb" in disk:
        if disk["free_gb"] < 5.0:
            recommendations.append(
                ("⚠", f"Low disk space: {disk['free_gb']:.1f}GB free. Need 5GB+ for ML packages.", Colors.WARNING)
            )
            recommendations.append(("→", "Clear pip cache: rm -rf ~/.cache/pip", Colors.OKCYAN))
        else:
            recommendations.append(("✓", f"Sufficient disk space: {disk['free_gb']:.1f}GB free", Colors.OKGREEN))

    # Package recommendations
    if not capabilities["minimal_ready"]:
        recommendations.append(("🔧", "Install core packages: make install-core", Colors.WARNING))
    elif not capabilities["standard_ready"]:
        recommendations.append(("🔧", "Upgrade to Standard: make install-core", Colors.OKCYAN))
    elif not capabilities["full_ready"]:
        if disk.get("sufficient", False):
            recommendations.append(("🔧", "Upgrade to Full: make install-ml-core", Colors.OKCYAN))
            recommendations.append(
                (
                    "→",
                    "Advanced Apple Silicon ML bootstrap: ./scripts/bootstrap/install_ml_stack.sh --profile core-cpu",
                    Colors.OKCYAN,
                )
            )
        else:
            recommendations.append(("⚠", "Full tier requires more disk space. Free up space first.", Colors.WARNING))

    # Model downloads
    repo_root = Path(__file__).resolve().parents[2]
    if (repo_root / "scripts" / "setup" / "download_depth_models.py").exists():
        recommendations.append(
            ("📦", "Download ML models: .venv/bin/python scripts/setup/download_depth_models.py", Colors.OKCYAN)
        )

    # Print all recommendations
    for symbol, message, color in recommendations:
        print(colored(f"{symbol} {message}", color))


def main():
    """Main readiness check routine."""
    parser = argparse.ArgumentParser(
        description="Check image processing readiness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                  # Full readiness check
  %(prog)s --verbose       # Show detailed information
  %(prog)s --quick-start   # Show only quick start guide
        """,
    )
    parser.add_argument("--verbose", action="store_true", help="Show detailed information")
    parser.add_argument("--quick-start", action="store_true", help="Show only quick start guide")

    args = parser.parse_args()

    # Run checks
    disk = check_disk_space()
    capabilities = assess_capabilities()
    images = check_sample_images()
    ffmpeg_installed, ffmpeg_version = check_ffmpeg()

    # Print results
    print("\n" + "=" * 70)
    print(colored("TRANSFORMATION PORTAL - IMAGE PROCESSING READINESS", Colors.BOLD))
    print("=" * 70)

    if not args.quick_start:
        # System info
        print("\n" + colored("💻 SYSTEM INFORMATION", Colors.OKBLUE))
        print("-" * 70)
        print(f"Python: {sys.version.split()[0]}")
        if "free_gb" in disk:
            print(
                f"Disk Space: {disk['free_gb']:.1f}GB free / {disk['total_gb']:.1f}GB total ({disk['used_percent']:.1f}% used)"
            )
        if ffmpeg_installed:
            print(f"FFmpeg: {ffmpeg_version.split()[2] if 'version' in ffmpeg_version else 'Installed'}")
        else:
            print("FFmpeg: Not installed (required for video processing)")

        # Tier status
        print_tier_status(capabilities)

        # Available operations
        print_available_operations(capabilities)

        # Recommendations
        print_recommendations(disk, capabilities)

    # Quick start guide
    print_quick_start_guide(capabilities, images)

    # Summary
    print("\n" + "=" * 70)
    print(colored("SUMMARY", Colors.BOLD))
    print("=" * 70)

    if capabilities["full_ready"]:
        print(colored("✓ FULLY READY", Colors.OKGREEN))
        print("  You can use all image processing pipelines!")
        return 0
    elif capabilities["standard_ready"]:
        print(colored("✓ STANDARD READY", Colors.OKGREEN))
        print("  You can process images professionally (no AI features yet)")
        return 0
    elif capabilities["minimal_ready"]:
        print(colored("○ MINIMAL READY", Colors.WARNING))
        print("  Basic operations available. Install more packages for full features.")
        return 0
    else:
        print(colored("✗ NOT READY", Colors.FAIL))
        print("  Install core packages to get started:")
        print("  make install-core")
        print("  make check-environment")
        return 1


if __name__ == "__main__":
    sys.exit(main())
