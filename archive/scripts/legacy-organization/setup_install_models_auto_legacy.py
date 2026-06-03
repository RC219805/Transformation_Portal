#!/usr/bin/env python3
"""Automated Model Installation - No prompts"""

import urllib.request
from pathlib import Path

print("=" * 70)
print("AUTOMATED MODEL INSTALLATION")
print("=" * 70)

REPO_ROOT = Path(__file__).parent.parent
WEIGHTS_DIR = REPO_ROOT / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)


def download_file(url, output_path):
    """Download file with progress."""
    print(f"\nDownloading: {output_path.name}")
    print(f"URL: {url}")

    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, (downloaded / total_size) * 100)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, output_path, reporthook=report_progress)
        print(f"\n✓ Downloaded: {output_path.name}")
        return True
    except Exception as e:
        print(f"\n✗ Failed: {e}")
        return False


# Check Depth Anything V2
print("\n[1/4] Checking Depth Anything V2...")
try:
    from transformers import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-h")
    print("✓ Depth Anything V2 ready")
except Exception as e:
    print(f"⚠ Will download on first use: {e}")

# Download Real-ESRGAN
print("\n[2/4] Installing Real-ESRGAN weights...")
model_path = WEIGHTS_DIR / "RealESRGAN_x4plus.pth"

if model_path.exists():
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"✓ Already exists: {model_path.name} ({size_mb:.1f} MB)")
else:
    url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    if download_file(url, model_path):
        print(f"✓ Real-ESRGAN installed: {model_path}")
    else:
        print(f"✗ Manual download required: {url}")

# Check ControlNet
print("\n[3/4] Checking ControlNet models...")
try:
    from huggingface_hub import snapshot_download

    for model_id in ["lllyasviel/sd-controlnet-canny", "lllyasviel/sd-controlnet-depth"]:
        try:
            snapshot_download(repo_id=model_id, allow_patterns=["*.json"])
            print(f"✓ {model_id}")
        except Exception:
            print(f"⚠ {model_id} - will download on first use")
except ImportError:
    print("⚠ huggingface_hub not installed")

# Check Stable Diffusion
print("\n[4/4] Checking Stable Diffusion...")
try:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id="runwayml/stable-diffusion-v1-5", allow_patterns=["*.json"])
    print("✓ Stable Diffusion v1.5 cached")
except Exception:
    print("⚠ Will download on first use (~4GB)")

print("\n" + "=" * 70)
print("✅ INSTALLATION COMPLETE")
print("=" * 70)
print(f"\nWeights directory: {WEIGHTS_DIR}")
print("HuggingFace cache: ~/.cache/huggingface/")
print("\nModels will auto-download on first pipeline run if not cached.")
print("=" * 70)
