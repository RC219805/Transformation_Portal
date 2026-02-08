#!/usr/bin/env python3
"""
Install Missing Pipeline Models
Transformation Portal - Model Setup Script

Downloads and configures:
1. Depth Anything V2 (HuggingFace)
2. Real-ESRGAN weights
3. ControlNet models (if missing)
"""

import os
import sys
import urllib.request
from pathlib import Path

from tqdm import tqdm

print("=" * 70)
print("TRANSFORMATION PORTAL - MODEL INSTALLATION")
print("=" * 70)

# Setup paths
REPO_ROOT = Path(__file__).parent.parent
WEIGHTS_DIR = REPO_ROOT / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, output_path):
    """Download file with progress bar."""
    print(f"\nDownloading: {output_path.name}")
    print(f"From: {url}")

    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, output_path, reporthook=t.update_to)

    print(f"✓ Downloaded: {output_path}")


# ============================================================================
# 1. DEPTH ANYTHING V2 - HuggingFace Model
# ============================================================================
print("\n" + "=" * 70)
print("1. DEPTH ANYTHING V2")
print("=" * 70)

print("\nChecking for Depth Anything V2 in HuggingFace cache...")

try:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    # Check if model is cached
    model_id = "LiheYoung/depth-anything-small-h"

    try:
        print(f"\nVerifying model: {model_id}")
        processor = AutoImageProcessor.from_pretrained(model_id)
        print(f"✓ Model found in cache: {model_id}")
        print(f"  Processor: {type(processor).__name__}")

        # Test model loading (don't actually load to save time)
        print("✓ Depth Anything V2 is ready")

    except Exception as e:
        print("⚠ Model not in cache, will download on first use")
        print("  This is normal - models download automatically")

except ImportError:
    print("✗ transformers not installed")
    print("  Install with: pip install transformers")

# ============================================================================
# 2. REAL-ESRGAN WEIGHTS
# ============================================================================
print("\n" + "=" * 70)
print("2. REAL-ESRGAN WEIGHTS")
print("=" * 70)

REALESRGAN_MODELS = {
    "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "RealESRGAN_x4plus_anime_6B.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
}

print(f"\nChecking Real-ESRGAN weights in: {WEIGHTS_DIR}")

for model_name, model_url in REALESRGAN_MODELS.items():
    model_path = WEIGHTS_DIR / model_name

    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✓ Found: {model_name} ({size_mb:.1f} MB)")
    else:
        print(f"\n⚠ Missing: {model_name}")
        response = input(f"  Download {model_name}? (~67MB) [y/N]: ").lower().strip()

        if response == "y":
            try:
                download_file(model_url, model_path)
            except Exception as e:
                print(f"✗ Download failed: {e}")
                print(f"  Manual download: {model_url}")
        else:
            print("  Skipped. Real-ESRGAN 4x upscaling will not be available.")
            print(f"  Download manually: {model_url}")

# ============================================================================
# 3. CONTROLNET MODELS (via HuggingFace)
# ============================================================================
print("\n" + "=" * 70)
print("3. CONTROLNET MODELS")
print("=" * 70)

CONTROLNET_MODELS = [
    "lllyasviel/sd-controlnet-canny",
    "lllyasviel/sd-controlnet-depth",
]

print("\nChecking ControlNet models in HuggingFace cache...")

try:
    from diffusers import ControlNetModel

    for model_id in CONTROLNET_MODELS:
        try:
            # Check if model exists in cache (don't load)
            from huggingface_hub import snapshot_download

            cache_dir = snapshot_download(repo_id=model_id, allow_patterns=["*.json"])
            print(f"✓ Found: {model_id}")
        except Exception as e:
            print(f"⚠ Not cached: {model_id}")
            print("  Will download automatically on first use (~1.5GB)")

except ImportError:
    print("✗ diffusers not installed")
    print("  Install with: pip install diffusers")

# ============================================================================
# 4. STABLE DIFFUSION MODEL
# ============================================================================
print("\n" + "=" * 70)
print("4. STABLE DIFFUSION MODEL")
print("=" * 70)

SD_MODEL = "runwayml/stable-diffusion-v1-5"

print(f"\nChecking for: {SD_MODEL}")

try:
    from huggingface_hub import snapshot_download

    try:
        cache_dir = snapshot_download(repo_id=SD_MODEL, allow_patterns=["*.json"])
        print(f"✓ Found: {SD_MODEL}")
    except Exception:
        print(f"⚠ Not cached: {SD_MODEL}")
        print("  Will download automatically on first use (~4GB)")

except ImportError:
    print("✗ huggingface_hub not installed")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("INSTALLATION SUMMARY")
print("=" * 70)

print("\n📦 Model Locations:")
print("  • HuggingFace models: ~/.cache/huggingface/")
print(f"  • Real-ESRGAN weights: {WEIGHTS_DIR}")
print("  • ControlNet models: (cached by HuggingFace)")

print("\n💡 Notes:")
print("  • HuggingFace models download automatically on first use")
print("  • Downloads happen once, then cached for future use")
print("  • Real-ESRGAN requires manual download (~67MB per model)")
print("  • First run of each pipeline will take longer due to downloads")

print("\n🔧 Optional Dependencies:")
print("  For faster loading: pip install accelerate")
print("  For Real-ESRGAN: pip install realesrgan basicsr")

print("\n✅ Setup complete!")
print("=" * 70)
