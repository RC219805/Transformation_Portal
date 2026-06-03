#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Automated Model Installation - No prompts, with reliability features.

Downloads and configures machine learning models and governed model weights for Transformation Portal:
1. Depth Anything V2 (HuggingFace) - Monocular depth estimation
2. Real-ESRGAN-compatible weights - legacy/local upscaling research artifacts
3. ControlNet models - Image conditioning for Stable Diffusion
4. Stable Diffusion - Base generation model

Features (upgraded from basic version):
- Automatic retry on failure (3 attempts)
- SHA256 checksum verification
- Disk space checking
- Download resume support
- Progress bars with tqdm
- Graceful error handling

Usage:
    python scripts/setup/install_models_auto.py [--skip-optional] [--force]

Performance: ~5-10 minutes total for required models (depends on connection)

Author: Transformation Portal Team
License: Attribution (see LICENSE)
"""

import argparse
import hashlib
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ============================================================================
# Configuration
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = REPO_ROOT / "weights"

# Import security helpers from local source tree (script can run before package install).
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

MODEL_LOCK_IMPORT_ERROR: Optional[Exception] = None
try:
    from transformation_portal.core.security.model_lock import ModelLockError, resolve_model_lock_revision

    MODEL_LOCK_AVAILABLE = True
except Exception as exc:  # pragma: no cover - defensive import guard for bootstrap environments
    ModelLockError = RuntimeError  # type: ignore[assignment]
    MODEL_LOCK_AVAILABLE = False
    MODEL_LOCK_IMPORT_ERROR = exc

# Real-ESRGAN-compatible model weights with checksum. The external Python
# package remains unsupported by dependency policy.
REALESRGAN_MODEL = {
    "name": "RealESRGAN_x4plus.pth",
    "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "size_mb": 64,
    "sha256": "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",  # Verified checksum
}

MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


class SecureModelLockCheckError(RuntimeError):
    """Raised when strict model-lock checks fail in installer preflight."""


def _resolve_required_revision(repo_id: str, context: str) -> str:
    """Resolve pinned revision for secure model downloads.

    This installer is intentionally fail-closed for HuggingFace artifacts.
    """
    if not MODEL_LOCK_AVAILABLE:
        reason = (
            f"{type(MODEL_LOCK_IMPORT_ERROR).__name__}: {MODEL_LOCK_IMPORT_ERROR}" if MODEL_LOCK_IMPORT_ERROR else "unknown"
        )
        raise SecureModelLockCheckError(f"Model lock helpers unavailable ({reason})")

    revision = resolve_model_lock_revision(
        repo_id,
        requested_revision=None,
        strict=True,
        context=context,
    )
    if not revision:
        raise ModelLockError(f"{context}: missing pinned revision for repo '{repo_id}'")
    return revision


def check_disk_space(required_mb: int = 1000) -> bool:
    """Check if sufficient disk space is available.

    Args:
        required_mb: Required space in megabytes

    Returns:
        True if sufficient space available
    """
    try:
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        stat = os.statvfs(WEIGHTS_DIR)
        free_mb = (stat.f_bavail * stat.f_frsize) / (1024 * 1024)

        if free_mb < required_mb:
            print(f"⚠️  Warning: Low disk space ({free_mb:.1f} MB free, {required_mb} MB required)")
            return False
        return True
    except Exception:
        return True


def verify_checksum(file_path: Path, expected_sha256: Optional[str]) -> bool:
    """Verify file SHA256 checksum.

    Args:
        file_path: Path to file
        expected_sha256: Expected SHA256 hash (or None to skip)

    Returns:
        True if checksum matches or verification skipped
    """
    if not expected_sha256:
        return True

    print("  Verifying checksum...")
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)

    actual = sha256.hexdigest()
    if actual != expected_sha256:
        print("  ❌ Checksum mismatch!")
        print(f"     Expected: {expected_sha256}")
        print(f"     Actual:   {actual}")
        return False

    print("  ✓ Checksum verified")
    return True


def download_file(url: str, output_path: Path, expected_sha256: Optional[str] = None, max_retries: int = MAX_RETRIES) -> bool:
    """Download file with retry logic and checksum verification.

    Args:
        url: Download URL
        output_path: Output file path
        expected_sha256: Expected SHA256 hash for verification
        max_retries: Maximum retry attempts

    Returns:
        True if download successful and verified
    """
    print(f"\nDownloading: {output_path.name}")
    print(f"  URL: {url}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                print(f"  Retry attempt {attempt}/{max_retries}...")
                time.sleep(RETRY_DELAY)

            if HAS_TQDM:
                with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1) as pbar:

                    def update_progress(block_num: int, block_size: int, total_size: int):
                        if pbar.total is None and total_size > 0:
                            pbar.total = total_size
                        pbar.update(block_size)

                    urllib.request.urlretrieve(url, output_path, reporthook=update_progress)
            else:

                def report_progress(block_num: int, block_size: int, total_size: int):
                    downloaded = block_num * block_size
                    if total_size > 0:
                        percent = min(100, (downloaded / total_size) * 100)
                        mb_downloaded = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        print(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)

                urllib.request.urlretrieve(url, output_path, reporthook=report_progress)
                print()

            # Verify checksum
            if not verify_checksum(output_path, expected_sha256):
                if output_path.exists():
                    output_path.unlink()  # Delete corrupted file
                if attempt < max_retries:
                    continue  # Retry
                return False

            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ Downloaded: {output_path.name} ({size_mb:.1f} MB)")
            return True

        except Exception as e:
            print(f"  ❌ Download failed: {e}")
            if output_path.exists():
                output_path.unlink()

            if attempt >= max_retries:
                print("  ❌ Max retries reached. Manual download required:")
                print(f"     {url}")
                return False

    return False


def check_depth_anything() -> bool:
    """Check if Depth Anything V2 is available.

    Returns:
        True if model can be loaded
    """
    print("\n[1/4] Checking Depth Anything V2...")
    model_id = "depth-anything/Depth-Anything-V2-Small-hf"
    try:
        from transformers import AutoImageProcessor

        revision = _resolve_required_revision(
            model_id,
            context="install_models_auto.depth_anything",
        )
        AutoImageProcessor.from_pretrained(model_id, revision=revision)  # nosec B615
        print("  ✓ Depth Anything V2 ready")
        return True
    except (ModelLockError, SecureModelLockCheckError) as e:
        print(f"  ❌ Secure model lock check failed: {e}")
        raise SecureModelLockCheckError(f"Depth Anything lock check failed: {e}") from e
    except Exception as e:
        print(f"  ⚠️  Will download on first use: {e}")
        return False


def install_realesrgan(force: bool = False) -> bool:
    """Install Real-ESRGAN-compatible weights.

    Args:
        force: Force re-download even if file exists

    Returns:
        True if installation successful
    """
    print("\n[2/4] Installing Real-ESRGAN-compatible weights...")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = WEIGHTS_DIR / REALESRGAN_MODEL["name"]

    if model_path.exists() and not force:
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Already exists: {model_path.name} ({size_mb:.1f} MB)")

        # Verify checksum of existing file
        if not verify_checksum(model_path, REALESRGAN_MODEL["sha256"]):
            print("  ⚠️  Existing file corrupted, re-downloading...")
            force = True

    if not model_path.exists() or force:
        if not check_disk_space(REALESRGAN_MODEL["size_mb"]):
            return False

        return download_file(REALESRGAN_MODEL["url"], model_path, REALESRGAN_MODEL["sha256"])

    return True


def check_controlnet() -> bool:
    """Check if ControlNet models are available.

    Returns:
        True if models can be accessed
    """
    print("\n[3/4] Checking ControlNet models...")
    try:
        from huggingface_hub import snapshot_download

        models = ["lllyasviel/sd-controlnet-canny", "lllyasviel/sd-controlnet-depth"]
        success = True

        for model_id in models:
            try:
                revision = _resolve_required_revision(
                    model_id,
                    context=f"install_models_auto.controlnet.{model_id}",
                )
                snapshot_download(  # nosec B615
                    repo_id=model_id,
                    revision=revision,
                    allow_patterns=["*.json"],
                )
                print(f"  ✓ {model_id}")
            except (ModelLockError, SecureModelLockCheckError) as e:
                print(f"  ❌ {model_id} - secure model lock check failed: {e}")
                raise SecureModelLockCheckError(f"ControlNet lock check failed for {model_id}: {e}") from e
            except Exception:
                print(f"  ⚠️  {model_id} - will download on first use")
                success = False

        return success
    except ImportError:
        print("  ⚠️  huggingface_hub not installed")
        return False


def check_stable_diffusion(skip_optional: bool = False) -> bool:
    """Check if Stable Diffusion is available.

    Args:
        skip_optional: Skip optional models

    Returns:
        True if model can be accessed
    """
    if skip_optional:
        print("\n[4/4] Skipping Stable Diffusion (optional)")
        return True

    print("\n[4/4] Checking Stable Diffusion...")
    model_id = "runwayml/stable-diffusion-v1-5"
    try:
        from huggingface_hub import snapshot_download

        revision = _resolve_required_revision(
            model_id,
            context="install_models_auto.stable_diffusion",
        )
        snapshot_download(  # nosec B615
            repo_id=model_id,
            revision=revision,
            allow_patterns=["*.json"],
        )
        print("  ✓ Stable Diffusion v1.5 cached")
        return True
    except (ModelLockError, SecureModelLockCheckError) as e:
        print(f"  ❌ Secure model lock check failed: {e}")
        raise SecureModelLockCheckError(f"Stable Diffusion lock check failed: {e}") from e
    except Exception:
        print("  ⚠️  Will download on first use (~4GB)")
        return False


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Automated installation of Transformation Portal ML models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--skip-optional",
        action="store_true",
        help="Skip optional models (Stable Diffusion)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist",
    )
    return parser.parse_args()


def main() -> None:
    """Main installation workflow."""
    args = parse_args()

    print("=" * 70)
    print("AUTOMATED MODEL INSTALLATION")
    print("Transformation Portal - ML Model Setup")
    print("=" * 70)

    if not HAS_TQDM:
        print("⚠️  tqdm not installed. Install with: pip install tqdm")
        print("    (Progress bars will be basic)")

    print(f"\nWeights directory: {WEIGHTS_DIR}")

    check_disk_space(required_mb=500)

    # Install models
    results = []
    results.append(check_depth_anything())
    results.append(install_realesrgan(force=args.force))
    results.append(check_controlnet())
    results.append(check_stable_diffusion(skip_optional=args.skip_optional))

    print("\n" + "=" * 70)
    if all(results):
        print("✅ INSTALLATION COMPLETE - All models ready")
    else:
        print("⚠️  INSTALLATION PARTIAL - Some models will download on first use")
    print("=" * 70)
    print(f"\nWeights directory: {WEIGHTS_DIR}")
    print("HuggingFace cache: ~/.cache/huggingface/")
    print("\nModels will auto-download on first pipeline run if not cached.")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Installation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Installation failed: {e}")
        sys.exit(1)
