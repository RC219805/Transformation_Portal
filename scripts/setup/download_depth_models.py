#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download depth estimation models for the Transformation Portal.

This script downloads required models for depth-aware processing:
- Depth Anything V2 (CoreML format for Apple Silicon)
- Alternative depth models for non-Apple platforms

Usage:
    .venv/bin/python scripts/setup/download_depth_models.py [--model depth] [--output-dir DIR] [--verify-only]

Examples:
    # Download default Depth Anything V2 Small model
    .venv/bin/python scripts/setup/download_depth_models.py

    # Download to custom location
    .venv/bin/python scripts/setup/download_depth_models.py --output-dir ./models/depth
"""

import argparse
import hashlib
import re
import sys
import urllib.request
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# Model configuration constants
# Real-ESRGAN 4x upscaling model
# NOTE: Update URL and version when new releases are available
# Latest releases: https://github.com/xinntao/Real-ESRGAN/releases
REALESRGAN_MODEL_VERSION = "v0.2.5.0"
REALESRGAN_MODEL_URL = (
    f"https://github.com/xinntao/Real-ESRGAN/releases/download/{REALESRGAN_MODEL_VERSION}/RealESRGAN_x4plus.pth"
)
REALESRGAN_MODEL_FILENAME = "RealESRGAN_x4plus.pth"
# Verified SHA-256 for RealESRGAN_x4plus weights.
REALESRGAN_MODEL_SHA256 = "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1"

# Depth Anything V2 CoreML model (not yet publicly hosted)
# Will be available once official CoreML models are released
DEPTH_ANYTHING_V2_COREML_FILENAME = "DepthAnythingV2SmallF16.mlpackage"

_SHA256_HEX_RE = re.compile(r"^[a-fA-F0-9]{64}$")


class DownloadProgressBar:
    """Progress bar for downloads using tqdm if available, otherwise basic progress."""

    def __init__(self, desc: str):
        self.desc = desc
        self.pbar = None

    def __call__(self, block_num: int, block_size: int, total_size: int):
        if total_size > 0:
            if self.pbar is None:
                if tqdm:
                    self.pbar = tqdm(total=total_size, unit="B", unit_scale=True, unit_divisor=1024, desc=self.desc)
                else:
                    print(f"Downloading {self.desc}... 0%", end="", flush=True)

            downloaded = block_num * block_size
            if tqdm and self.pbar:
                self.pbar.update(block_size)
            elif not tqdm:
                percent = min(100, int(downloaded * 100 / total_size))
                print(f"\rDownloading {self.desc}... {percent}%", end="", flush=True)

            if downloaded >= total_size:
                if tqdm and self.pbar:
                    self.pbar.close()
                elif not tqdm:
                    print()  # New line after completion


def compute_file_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(file_path: Path, expected_sha256: str) -> bool:
    """Verify SHA-256 checksum for a downloaded file."""
    normalized = expected_sha256.strip().lower()
    if not _SHA256_HEX_RE.fullmatch(normalized):
        raise ValueError(f"Invalid SHA256 digest format: {expected_sha256!r}")
    actual = compute_file_sha256(file_path)
    return actual == normalized


def download_file(url: str, output_path: Path, expected_sha256: str, desc: str = "file") -> bool:
    """Download a file with progress bar.

    Args:
        url: URL to download from
        output_path: Local path to save the file
        expected_sha256: Trusted SHA-256 digest (required)
        desc: Description for progress bar

    Returns:
        True if successful, False otherwise
    """
    try:
        if not _SHA256_HEX_RE.fullmatch(expected_sha256.strip()):
            raise ValueError(f"Missing or invalid SHA256 for {desc}. Refusing to download unverified artifact.")

        print(f"Downloading {desc} from {url}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")

        urllib.request.urlretrieve(url, temp_path, reporthook=DownloadProgressBar(desc))
        if not verify_checksum(temp_path, expected_sha256):
            actual = compute_file_sha256(temp_path)
            temp_path.unlink(missing_ok=True)
            print(
                f"✗ Failed checksum verification for {desc}. " f"Expected {expected_sha256.lower()}, got {actual}",
                file=sys.stderr,
            )
            return False

        temp_path.replace(output_path)

        print(f"✓ Successfully downloaded to {output_path}")
        return True

    except Exception as e:
        temp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
        temp_path.unlink(missing_ok=True)
        print(f"✗ Failed to download {desc}: {e}", file=sys.stderr)
        return False


def download_depth_anything_v2_coreml(output_dir: Path) -> bool:
    """Download Depth Anything V2 Small CoreML model for Apple Silicon.

    Note: This is a placeholder. The actual CoreML model needs to be:
    1. Converted from the PyTorch model using coremltools
    2. Hosted on a public URL (Hugging Face, GitHub releases, etc.)

    For now, we provide instructions for manual download.

    Args:
        output_dir: Directory to save the model

    Returns:
        True if successful, False otherwise
    """
    model_path = output_dir / DEPTH_ANYTHING_V2_COREML_FILENAME

    print("\n" + "=" * 70)
    print("DEPTH ANYTHING V2 COREML MODEL")
    print("=" * 70)
    print("\nThe CoreML model for Depth Anything V2 is not yet publicly hosted.")
    print("You have two options:\n")

    print("Option 1: Convert from PyTorch yoursel")
    print("-" * 70)
    print("1. Install coremltools: pip install coremltools")
    print("2. Download the PyTorch model from Hugging Face:")
    print("   https://huggingface.co/depth-anything/Depth-Anything-V2-Small")
    print("3. Convert to CoreML format using the conversion script")
    print("4. Save as:", model_path)

    print("\nOption 2: Use PyTorch model directly")
    print("-" * 70)
    print("The repository can use Depth Anything V2 through the governed ML baseline:")
    print("  make install-ml-core")
    print("The model will download automatically on first use.")
    print("Note: PyTorch is slower than CoreML on Apple Silicon but works cross-platform.")

    print("\nOption 3: Use alternative depth model")
    print("-" * 70)
    print("You can also use ZoeDepth or MiDaS via controlnet-aux:")
    print("  pip install controlnet-aux")
    print("Models download automatically but are large (1.4GB+).")

    print("\n" + "=" * 70)

    return False  # Manual action required


def download_realesrgan_weights(output_dir: Path, expected_sha256: str = REALESRGAN_MODEL_SHA256) -> bool:
    """Download Real-ESRGAN 4x upscaling model weights.

    Args:
        output_dir: Directory to save the model

    Returns:
        True if successful, False otherwise
    """
    model_path = output_dir / REALESRGAN_MODEL_FILENAME

    print("\n" + "=" * 70)
    print("REAL-ESRGAN 4X UPSCALING MODEL")
    print("=" * 70)

    if model_path.exists():
        if verify_checksum(model_path, expected_sha256):
            print(f"✓ Model already exists and checksum verified: {model_path}")
            return True
        print(f"⚠️  Existing model failed checksum verification, re-downloading: {model_path}")
        model_path.unlink(missing_ok=True)

    return download_file(REALESRGAN_MODEL_URL, model_path, expected_sha256, "Real-ESRGAN 4x model")


def verify_models(model_dir: Path) -> dict:
    """Verify which models are installed.

    Args:
        model_dir: Directory containing models

    Returns:
        Dictionary of model status
    """
    status = {
        "depth_anything_coreml": (model_dir / DEPTH_ANYTHING_V2_COREML_FILENAME).exists(),
    }

    print("\n" + "=" * 70)
    print("MODEL STATUS")
    print("=" * 70)

    for model_name, installed in status.items():
        symbol = "✓" if installed else "✗"
        status_text = "Installed" if installed else "Not installed"
        print(f"{symbol} {model_name}: {status_text}")

    return status


def main():
    """Main entry point for model download script."""
    parser = argparse.ArgumentParser(description="Depth model setup and verification utility for Transformation Portal")
    parser.add_argument(
        "--model",
        type=str,
        choices=["depth"],
        default="depth",
        help="Which model to download (default: depth)",
    )
    parser.add_argument("--output-dir", type=str, default="./weights", help="Output directory for models (default: ./weights)")
    parser.add_argument("--verify-only", action="store_true", help="Only verify model status, don't download")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Transformation Portal - Model Download Utility")
    print("=" * 70)
    print(f"Output directory: {output_dir.absolute()}\n")

    if args.verify_only:
        verify_models(output_dir)
        return 0

    success = True

    if args.model == "depth":
        # Depth Anything V2 CoreML (manual instructions)
        download_depth_anything_v2_coreml(output_dir)

    # Verify final status
    print("\n")
    verify_models(output_dir)

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("1. Install pinned core runtime and dev tooling:")
    print("   make install-core")
    print("\n2. Install the supported ML baseline when you need live model execution:")
    print("   make install-ml-core")
    print("\n3. Test your installation:")
    print("   .venv/bin/python scripts/verification/verify_core.py")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
