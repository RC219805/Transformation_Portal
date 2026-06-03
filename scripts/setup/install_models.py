#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Install Missing Pipeline Models
Transformation Portal - Model Setup Script

Downloads and configures machine learning models and governed model weights for:
1. Depth Anything V2 (HuggingFace) - Monocular depth estimation
2. Real-ESRGAN-compatible weights - legacy/local upscaling research artifacts
3. ControlNet models - Image conditioning for Stable Diffusion
4. Stable Diffusion XL - Base generation model

Features:
- SHA256 checksum verification
- Automatic retry on failure
- Disk space checking
- Model size estimation
- Progress bars with tqdm
- Offline mode detection
- Dry-run mode

Usage:
    python scripts/setup/install_models.py [--all] [--dry-run] [--force]

Examples:
    # Install essential models only (Depth, Real-ESRGAN-compatible weights)
    python scripts/setup/install_models.py

    # Install all models including Stable Diffusion
    python scripts/setup/install_models.py --all

    # Check what would be downloaded without downloading
    python scripts/setup/install_models.py --all --dry-run

    # Force re-download even if files exist
    python scripts/setup/install_models.py --force

Author: Transformation Portal Team
License: Attribution (see LICENSE)
"""

import argparse
import hashlib
import os
import re
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("⚠️  tqdm not installed. Install with: pip install tqdm")

# ============================================================================
# Configuration
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = REPO_ROOT / "weights"

# Model registry with URLs and checksums
REALESRGAN_MODELS = {
    "RealESRGAN_x4plus.pth": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "size_mb": 64,
        "sha256": "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
        "description": "Real-ESRGAN-compatible 4x upscaling weights (general images)",
        "required": True,
    },
    "RealESRGAN_x4plus_anime_6B.pth": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "size_mb": 17,
        "sha256": None,
        "description": "Real-ESRGAN-compatible 4x upscaling weights (anime/illustration)",
        "required": False,
    },
}

_SHA256_HEX_RE = re.compile(r"^[a-fA-F0-9]{64}$")

DEPTH_MODELS = {
    "depth-anything-small": {
        "model_id": "LiheYoung/depth-anything-small-h",
        "size_mb": 100,
        "description": "Depth Anything V2 Small (fastest)",
        "required": True,
    },
    "depth-anything-base": {
        "model_id": "LiheYoung/depth-anything-base-h",
        "size_mb": 350,
        "description": "Depth Anything V2 Base (balanced)",
        "required": False,
    },
}

CONTROLNET_MODELS = [
    {
        "model_id": "lllyasviel/sd-controlnet-canny",
        "size_mb": 1500,
        "description": "ControlNet Canny edge detection",
        "required": False,
    },
    {
        "model_id": "lllyasviel/sd-controlnet-depth",
        "size_mb": 1500,
        "description": "ControlNet depth conditioning",
        "required": False,
    },
]

SD_MODELS = [
    {
        "model_id": "runwayml/stable-diffusion-v1-5",
        "size_mb": 4000,
        "description": "Stable Diffusion 1.5",
        "required": False,
    },
    {
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "size_mb": 7000,
        "description": "Stable Diffusion XL Base",
        "required": False,
    },
]

# ============================================================================
# Utilities
# ============================================================================


class DownloadProgressBar:
    """Progress bar for downloads with fallback."""

    def __init__(self, desc: str, total: Optional[int] = None):
        self.desc = desc
        self.total = total
        self.pbar = None

        if HAS_TQDM:
            self.pbar = tqdm(total=total, unit="B", unit_scale=True, miniters=1, desc=desc)
        else:
            print(f"Downloading {desc}...", end="", flush=True)

    def update(self, n: int):
        if self.pbar:
            self.pbar.update(n)
        elif n >= self.total:
            print(" Done!")

    def close(self):
        if self.pbar:
            self.pbar.close()

    def __call__(self, block_num: int, block_size: int, total_size: int):
        """Callback for urllib.request.urlretrieve"""
        if self.total is None and total_size > 0:
            self.total = total_size
            if self.pbar:
                self.pbar.total = total_size

        downloaded = block_num * block_size
        if self.pbar:
            self.pbar.n = min(downloaded, total_size)
            self.pbar.refresh()
        elif downloaded >= total_size:
            print(" Done!")


def check_disk_space(required_mb: int) -> bool:
    """Check if sufficient disk space is available."""
    try:
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        stat = shutil.disk_usage(WEIGHTS_DIR)
        free_mb = stat.free / (1024 * 1024)

        if free_mb < required_mb * 1.1:  # 10% buffer
            print(f"⚠️  Low disk space: {free_mb:.0f} MB free, need {required_mb} MB")
            return False
        return True
    except Exception as e:
        print(f"⚠️  Could not check disk space: {e}")
        return True  # Proceed anyway


def verify_checksum(file_path: Path, expected_sha256: Optional[str]) -> bool:
    """Verify file SHA256 checksum."""
    if not expected_sha256:
        print("  ✗ Missing expected SHA256 checksum; refusing unverified artifact")
        return False
    if not _SHA256_HEX_RE.fullmatch(expected_sha256.strip()):
        print(f"  ✗ Invalid expected SHA256 format: {expected_sha256!r}")
        return False

    print("  Verifying checksum...")
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)

    actual = sha256.hexdigest()
    expected = expected_sha256.lower()

    if actual != expected:
        print("  ✗ Checksum mismatch!")
        print(f"    Expected: {expected}")
        print(f"    Got:      {actual}")
        return False

    print("  ✓ Checksum verified")
    return True


def download_file_with_retry(
    url: str, output_path: Path, description: str, expected_sha256: Optional[str] = None, max_retries: int = 3
) -> bool:
    """Download file with retry logic and checksum verification."""
    if not expected_sha256 or not _SHA256_HEX_RE.fullmatch(expected_sha256.strip()):
        print(f"  ✗ Missing or invalid SHA256 for {description}; refusing download")
        return False

    for attempt in range(max_retries):
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if attempt > 0:
                print(f"  Retry {attempt}/{max_retries}...")

            # Create progress bar
            progress = DownloadProgressBar(description)

            # Download
            urllib.request.urlretrieve(url, output_path, reporthook=progress)
            progress.close()

            # Verify checksum
            if not verify_checksum(output_path, expected_sha256):
                if attempt < max_retries - 1:
                    output_path.unlink()  # Remove corrupted file
                    continue
                return False

            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ Downloaded: {output_path.name} ({size_mb:.1f} MB)")
            return True

        except KeyboardInterrupt:
            print("\n\n⚠️  Download cancelled by user")
            if output_path.exists():
                output_path.unlink()
            raise
        except Exception as e:
            print(f"  ✗ Download failed: {e}")
            if attempt < max_retries - 1:
                if output_path.exists():
                    output_path.unlink()
            else:
                print(f"  Failed after {max_retries} attempts")
                return False

    return False


def check_huggingface_model(model_id: str) -> Tuple[bool, Optional[str]]:
    """Check if HuggingFace model is cached."""
    try:
        from huggingface_hub import snapshot_download

        # Try to find cached model
        cache_dir = snapshot_download(repo_id=model_id, allow_patterns=["*.json"], local_files_only=True)
        return True, cache_dir
    except Exception:
        return False, None


def estimate_download_time(size_mb: int, speed_mbps: float = 10.0) -> str:
    """Estimate download time based on size and connection speed."""
    seconds = (size_mb * 8) / speed_mbps

    if seconds < 60:
        return f"~{int(seconds)}s"
    elif seconds < 3600:
        return f"~{int(seconds / 60)}min"
    else:
        return f"~{seconds / 3600:.1f}hr"


# ============================================================================
# Installation Functions
# ============================================================================


def install_depth_models(install_all: bool = False, dry_run: bool = False) -> int:
    """Install Depth Anything V2 models."""
    print("\n" + "=" * 70)
    print("1. DEPTH ANYTHING V2 MODELS")
    print("=" * 70)

    installed = 0

    try:
        from transformers import AutoImageProcessor

        for name, config in DEPTH_MODELS.items():
            if not config["required"] and not install_all:
                continue

            model_id = config["model_id"]
            print(f"\nChecking: {config['description']}")
            print(f"  Model ID: {model_id}")

            is_cached, cache_dir = check_huggingface_model(model_id)

            if is_cached:
                print(f"  ✓ Already cached: {cache_dir}")
                installed += 1
            else:
                print(f"  ⚠️  Not cached (~{config['size_mb']} MB)")
                print(f"  Download time: {estimate_download_time(config['size_mb'])}")

                if dry_run:
                    print("  [DRY RUN] Would download on first use")
                else:
                    print("  Will download automatically on first pipeline run")

    except ImportError:
        print("\n✗ transformers not installed")
        print("  Install with: pip install transformers torch")
        return 0

    return installed


def install_realesrgan_weights(force: bool = False, dry_run: bool = False) -> int:
    """Install Real-ESRGAN-compatible model weights."""
    print("\n" + "=" * 70)
    print("2. REAL-ESRGAN-COMPATIBLE WEIGHTS")
    print("=" * 70)

    print(f"\nWeights directory: {WEIGHTS_DIR}")
    if not dry_run:
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    installed = 0

    for model_name, config in REALESRGAN_MODELS.items():
        model_path = WEIGHTS_DIR / model_name
        checksum = config.get("sha256")

        print(f"\nModel: {config['description']}")
        print(f"  File: {model_name}")
        print(f"  Size: ~{config['size_mb']} MB")

        if config["required"] and (not checksum or not _SHA256_HEX_RE.fullmatch(checksum.strip())):
            print("  ✗ Required model has missing/invalid SHA256 metadata; refusing insecure download")
            continue

        if model_path.exists() and not force:
            if checksum:
                if verify_checksum(model_path, checksum):
                    size_mb = model_path.stat().st_size / (1024 * 1024)
                    print(f"  ✓ Already installed and verified ({size_mb:.1f} MB)")
                    installed += 1
                    continue
                print("  ⚠️  Existing file failed verification; will re-download")
                model_path.unlink(missing_ok=True)
            elif config["required"]:
                print("  ✗ Required model is missing trusted SHA256 metadata; refusing to use existing file")
                model_path.unlink(missing_ok=True)
            else:
                size_mb = model_path.stat().st_size / (1024 * 1024)
                print(f"  ✓ Already installed ({size_mb:.1f} MB, checksum unavailable)")
                installed += 1
                continue

        if not config["required"]:
            print("  ⚠️  Optional model, skipping")
            print(f"  Download manually if needed: {config['url']}")
            continue

        print("  ⚠️  Not found")
        print(f"  Download time: {estimate_download_time(config['size_mb'])}")

        if dry_run:
            print("  [DRY RUN] Would prompt for download")
            continue

        # Check disk space
        if not check_disk_space(config["size_mb"]):
            print("  ✗ Insufficient disk space, skipping")
            continue

        # Prompt for download
        response = input(f"  Download {model_name}? [y/N]: ").lower().strip()

        if response == "y":
            success = download_file_with_retry(config["url"], model_path, model_name, config["sha256"])

            if success:
                installed += 1
            else:
                print("  ✗ Download failed")
                print(f"  Manual download: {config['url']}")
        else:
            print("  Skipped")

    return installed


def install_controlnet_models(dry_run: bool = False) -> int:
    """Install ControlNet models."""
    print("\n" + "=" * 70)
    print("3. CONTROLNET MODELS")
    print("=" * 70)

    installed = 0

    try:
        from diffusers import ControlNetModel

        for config in CONTROLNET_MODELS:
            model_id = config["model_id"]
            print(f"\nModel: {config['description']}")
            print(f"  ID: {model_id}")
            print(f"  Size: ~{config['size_mb']} MB")

            is_cached, cache_dir = check_huggingface_model(model_id)

            if is_cached:
                print("  ✓ Already cached")
                installed += 1
            else:
                print("  ⚠️  Not cached")
                print(f"  Download time: {estimate_download_time(config['size_mb'])}")

                if dry_run:
                    print("  [DRY RUN] Would download on first use")
                else:
                    print("  Will download automatically on first use")

    except ImportError:
        print("\n✗ diffusers not installed")
        print("  Install with: pip install diffusers torch")
        return 0

    return installed


def install_stable_diffusion_models(dry_run: bool = False) -> int:
    """Install Stable Diffusion models."""
    print("\n" + "=" * 70)
    print("4. STABLE DIFFUSION MODELS")
    print("=" * 70)

    installed = 0

    for config in SD_MODELS:
        model_id = config["model_id"]
        print(f"\nModel: {config['description']}")
        print(f"  ID: {model_id}")
        print(f"  Size: ~{config['size_mb']} MB")

        is_cached, cache_dir = check_huggingface_model(model_id)

        if is_cached:
            print("  ✓ Already cached")
            installed += 1
        else:
            print("  ⚠️  Not cached")
            print(f"  Download time: {estimate_download_time(config['size_mb'])}")

            if dry_run:
                print("  [DRY RUN] Would download on first use")
            else:
                print("  Will download automatically on first use")

    return installed


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Install machine learning models for Transformation Portal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Install essential models (Depth, Real-ESRGAN-compatible weights)
  python scripts/setup/install_models.py

  # Install all models including Stable Diffusion
  python scripts/setup/install_models.py --all

  # Preview what would be downloaded
  python scripts/setup/install_models.py --all --dry-run

  # Force re-download existing models
  python scripts/setup/install_models.py --force
        """,
    )

    parser.add_argument("--all", action="store_true", help="Install all models including optional ones")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded without downloading")
    parser.add_argument("--force", action="store_true", help="Force re-download even if files exist")

    args = parser.parse_args()

    # Header
    print("=" * 70)
    print("TRANSFORMATION PORTAL - MODEL INSTALLATION")
    print("=" * 70)

    if args.dry_run:
        print("\n[DRY RUN MODE] - No files will be downloaded")

    # Install models
    depth_installed = install_depth_models(args.all, args.dry_run)
    realesrgan_installed = install_realesrgan_weights(args.force, args.dry_run)

    if args.all:
        controlnet_installed = install_controlnet_models(args.dry_run)
        sd_installed = install_stable_diffusion_models(args.dry_run)
    else:
        controlnet_installed = 0
        sd_installed = 0
        print("\n💡 Tip: Use --all to check/install ControlNet and Stable Diffusion models")

    # Summary
    print("\n" + "=" * 70)
    print("INSTALLATION SUMMARY")
    print("=" * 70)

    print("\n✓ Models Ready:")
    print(f"  • Depth Anything V2:  {depth_installed} models")
    print(f"  • Real-ESRGAN-compatible weights: {realesrgan_installed} models")
    if args.all:
        print(f"  • ControlNet:         {controlnet_installed} models")
        print(f"  • Stable Diffusion:   {sd_installed} models")

    print("\n📦 Model Locations:")
    print("  • HuggingFace cache: ~/.cache/huggingface/")
    print(f"  • Real-ESRGAN-compatible weights: {WEIGHTS_DIR}")

    print("\n💡 Notes:")
    print("  • HuggingFace models auto-download on first use")
    print("  • First pipeline run will be slower (model loading)")
    print("  • Models are cached and reused across runs")

    print("\n🔧 Optional Dependencies:")
    print("  • pip install accelerate      (faster loading)")
    print("  • External `realesrgan` package is unsupported by dependency policy")
    print("  • pip install torch            (GPU acceleration)")

    if not args.dry_run:
        print("\n✅ Setup complete!")
    else:
        print("\n[DRY RUN] Run without --dry-run to actually download")

    print("=" * 70)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
