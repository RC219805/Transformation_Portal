#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download sample images for Transformation Portal development and testing.

This script downloads sample images from GitHub Releases or external storage
to avoid bloating the Git repository with large binary files.

Usage:
    python scripts/download_samples.py [--all] [--output-dir DIR] [--force]

Examples:
    # Download minimal test fixtures (required for tests)
    python scripts/download_samples.py

    # Download all sample images (for pipeline testing)
    python scripts/download_samples.py --all

    # Force re-download even if files exist
    python scripts/download_samples.py --all --force

    # Download to custom location
    python scripts/download_samples.py --output-dir ./my_samples

Sample Categories:
    - minimal: Tiny synthetic images for unit tests (< 50KB total)
    - demo: Small demo images for README examples (~10MB total)
    - full: Complete sample dataset for pipeline testing (~50MB total)

Author: Transformation Portal Team
License: Attribution (see LICENSE)
"""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ============================================================================
# Sample Image Registry
# ============================================================================

# NOTE: Sample URLs are pending GitHub Release upload (v2.4.0 roadmap item)
# See: docs/architecture/TODO_INVENTORY_QUICK_REF.md - Finding #5
# Format: https://github.com/RC219805/Transformation_Portal/releases/download/samples-v1.0.0/<filename>

SAMPLE_REGISTRY: Dict[str, Dict] = {
    # ========================================================================
    # MINIMAL: Test fixtures for unit tests (synthetic images)
    # Generated locally — no network or third-party hosting dependency.
    # ========================================================================
    "test_image_small": {
        "category": "minimal",
        "url": None,
        "synthetic": "rgb_gradient",
        "synthetic_kwargs": {"width": 100, "height": 100, "seed": 0},
        "size": "1KB",
        "path": "tests/fixtures/test_image_small.jpg",
        "sha256": None,
        "description": "Tiny test image for unit tests (100x100px)",
    },
    "test_depth_map": {
        "category": "minimal",
        "url": None,
        "synthetic": "depth_gradient",
        "synthetic_kwargs": {"width": 256, "height": 256, "seed": 1},
        "size": "5KB",
        "path": "tests/fixtures/test_depth.jpg",
        "sha256": None,
        "description": "Grayscale depth map for testing (256x256px)",
    },
    # ========================================================================
    # DEMO: Small examples for README and documentation
    # Status: Pending GitHub Release (v2.4.0 roadmap)
    # ========================================================================
    "demo_coastal_interior": {
        "category": "demo",
        "url": None,  # Pending GitHub Release upload (v2.4.0)
        "size": "5MB",
        "path": "data/sample_images/demo_coastal_interior.jpg",
        "sha256": None,
        "description": "Coastal interior render (downscaled to 2K for demo)",
    },
    "demo_pool_aerial": {
        "category": "demo",
        "url": None,  # Pending GitHub Release upload (v2.4.0)
        "size": "8MB",
        "path": "data/sample_images/demo_pool_aerial.jpg",
        "sha256": None,
        "description": "Pool aerial enhancement demo (downscaled to 2K)",
    },
    # ========================================================================
    # FULL: Complete sample dataset for pipeline testing
    # Status: Pending GitHub Release (v2.4.0 roadmap)
    # ========================================================================
    "sample_render_4k": {
        "category": "full",
        "url": None,  # Pending GitHub Release upload (v2.4.0)
        "size": "25MB",
        "path": "data/sample_images/sample_render_4k.tif",
        "sha256": None,
        "description": "4K architectural render (16-bit TIFF)",
    },
    "sample_depth_anything_v2": {
        "category": "full",
        "url": None,  # Pending GitHub Release upload (v2.4.0)
        "size": "2MB",
        "path": "data/sample_images/depth_maps/sample_depth.npy",
        "sha256": None,
        "description": "Pre-computed depth map (Depth Anything V2)",
    },
}


# ============================================================================
# Download Utilities
# ============================================================================


class DownloadProgressBar:
    """Progress bar for downloads using tqdm if available."""

    def __init__(self, desc: str):
        self.desc = desc
        self.pbar = None

    def __call__(self, block_num: int, block_size: int, total_size: int):
        if self.pbar is None:
            if tqdm:
                self.pbar = tqdm(total=total_size, unit="B", unit_scale=True, desc=self.desc)
            else:
                # Fallback to simple progress
                print(f"Downloading {self.desc}...", end="", flush=True)

        if self.pbar:
            downloaded = block_num * block_size
            if downloaded < total_size:
                self.pbar.update(block_size)
            else:
                self.pbar.close()
        elif block_num * block_size >= total_size:
            print(" Done!")


def verify_checksum(file_path: Path, expected_sha256: Optional[str]) -> bool:
    """Verify file SHA256 checksum."""
    if expected_sha256 is None:
        return True  # Skip verification if no checksum provided

    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)

    actual = sha256.hexdigest()
    if actual != expected_sha256:
        print(f"❌ Checksum mismatch for {file_path.name}")
        print(f"   Expected: {expected_sha256}")
        print(f"   Got:      {actual}")
        return False
    return True


def _render_synthetic_image_bytes(kind: str, **kwargs) -> Optional[bytes]:
    """Render a synthetic fixture to JPEG bytes in memory (no I/O).

    Recognized kinds:
        - "rgb_gradient": colorful gradient + low-amplitude noise (RGB JPEG)
        - "depth_gradient": grayscale radial gradient (single-channel JPEG)

    Output is fully deterministic given the same kwargs (seeded RNG) so
    committed fixtures stay reproducible without third-party hosting.
    Returns None on dependency or input error.
    """
    try:
        import io

        import numpy as np
        from PIL import Image
    except ImportError as exc:
        print(f"❌ Synthetic generation requires Pillow + numpy: {exc}")
        return None

    width = int(kwargs.get("width", 100))
    height = int(kwargs.get("height", 100))
    seed = int(kwargs.get("seed", 0))
    rng = np.random.default_rng(seed)

    if kind == "rgb_gradient":
        ys = np.linspace(0, 255, height, dtype=np.float32)[:, None]
        xs = np.linspace(0, 255, width, dtype=np.float32)[None, :]
        red = np.broadcast_to(xs, (height, width))
        green = np.broadcast_to(ys, (height, width))
        blue = (xs + ys) * 0.5
        rgb = np.stack([red, green, blue], axis=-1)
        rgb = rgb + rng.uniform(-8.0, 8.0, size=rgb.shape).astype(np.float32)
        image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
    elif kind == "depth_gradient":
        cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
        yy, xx = np.indices((height, width), dtype=np.float32)
        radius = np.hypot(yy - cy, xx - cx)
        radius /= max(radius.max(), 1.0)
        depth = (1.0 - radius) * 255.0
        depth = depth + rng.uniform(-2.0, 2.0, size=depth.shape).astype(np.float32)
        image = Image.fromarray(np.clip(depth, 0, 255).astype(np.uint8), mode="L")
    else:
        print(f"❌ Unknown synthetic kind: {kind!r}")
        return None

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _generate_synthetic_image(kind: str, output_path: Path, **kwargs) -> str:
    """Materialize a synthetic fixture at output_path.

    Returns one of:
        - "up_to_date": existing file already matches deterministic output
        - "generated":  bytes were (re)written (migration from stale placeholder
                        fixtures or first-time write)
        - "failed":     dependency, render, or filesystem error
    """
    expected = _render_synthetic_image_bytes(kind, **kwargs)
    if expected is None:
        return "failed"

    # Migration path: stale fixtures left behind by older versions of this
    # script (placeholder downloads from a third party) get overwritten when
    # their bytes don't match the current deterministic render. Matching
    # bytes are left untouched to preserve idempotency and mtime.
    if output_path.exists():
        try:
            if output_path.read_bytes() == expected:
                return "up_to_date"
        except OSError as exc:
            print(f"❌ Could not read existing {output_path}: {exc}")
            return "failed"

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(expected)
        tmp_path.replace(output_path)
        return "generated"
    except OSError as exc:
        print(f"❌ Failed to write {output_path}: {exc}")
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        return "failed"


def download_file(url: str, output_path: Path, description: str, sha256: Optional[str] = None) -> bool:
    """Download a file with progress bar and checksum verification."""
    try:
        # Create parent directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Download file
        urllib.request.urlretrieve(url, output_path, reporthook=DownloadProgressBar(description))

        # Verify checksum
        if not verify_checksum(output_path, sha256):
            output_path.unlink()  # Remove corrupted file
            return False

        return True

    except Exception as e:
        print(f"❌ Failed to download {description}: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


# ============================================================================
# Main Download Logic
# ============================================================================


def get_samples_by_category(category: str) -> List[Dict]:
    """Get all samples in a category."""
    return [{**sample, "name": name} for name, sample in SAMPLE_REGISTRY.items() if sample["category"] == category]


def download_samples(categories: List[str], output_dir: Optional[Path] = None, force: bool = False) -> int:
    """
    Download samples from specified categories.

    Args:
        categories: List of categories to download ("minimal", "demo", "full")
        output_dir: Base output directory (default: repository root)
        force: Force re-download even if files exist

    Returns:
        Number of files successfully downloaded
    """
    if output_dir is None:
        # Default to repository root
        output_dir = Path(__file__).parent.parent

    # Collect all samples to download
    samples_to_download = []
    for category in categories:
        samples_to_download.extend(get_samples_by_category(category))

    if not samples_to_download:
        print(f"No samples found for categories: {categories}")
        return 0

    print(f"\n📥 Downloading {len(samples_to_download)} sample files...")
    print(f"📂 Output directory: {output_dir}\n")

    downloaded = 0
    skipped = 0
    failed = 0

    for sample in samples_to_download:
        output_path = output_dir / sample["path"]

        # Locally synthesized fixtures bypass the skip-if-exists short-circuit
        # because regenerating costs ~1ms and we need to migrate stale
        # placeholder fixtures from older script versions. The helper itself
        # returns "up_to_date" when bytes already match, preserving idempotency.
        synthetic_kind = sample.get("synthetic")
        if synthetic_kind:
            status = _generate_synthetic_image(synthetic_kind, output_path, **sample.get("synthetic_kwargs", {}))
            if status == "up_to_date":
                print(f"⏭️  Skipped {sample['name']} (synthetic, up to date)")
                skipped += 1
            elif status == "generated":
                print(f"✅ Generated {sample['name']} ({sample['size']}, synthetic)")
                downloaded += 1
            else:
                failed += 1
            continue

        # Skip if file exists and not forcing re-download (downloads only)
        if output_path.exists() and not force:
            print(f"⏭️  Skipped {sample['name']} (already exists)")
            skipped += 1
            continue

        # Check if URL is available
        if sample["url"] is None:
            print(f"⚠️  {sample['name']}: URL pending GitHub Release upload (v2.4.0 roadmap)")
            failed += 1
            continue

        # Download file
        success = download_file(sample["url"], output_path, sample["name"], sample.get("sha256"))

        if success:
            print(f"✅ Downloaded {sample['name']} ({sample['size']})")
            downloaded += 1
        else:
            failed += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"✅ Downloaded: {downloaded}")
    print(f"⏭️  Skipped:    {skipped}")
    print(f"❌ Failed:     {failed}")
    print(f"{'='*60}\n")

    if failed > 0:
        print("⚠️  Some files failed to download. This is expected if samples haven't been uploaded to GitHub Releases yet.")
        print("   See BINARY_FILE_BEST_PRACTICES.md for instructions on hosting samples.\n")

    return downloaded


# ============================================================================
# CLI Interface
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Download sample images for Transformation Portal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download minimal test fixtures (< 50KB)
  python scripts/download_samples.py

  # Download all samples including demo and full datasets
  python scripts/download_samples.py --all

  # Force re-download
  python scripts/download_samples.py --all --force

  # Download to custom location
  python scripts/download_samples.py --output-dir ./my_samples

Categories:
  minimal - Tiny synthetic images for unit tests (< 50KB total)
  demo    - Small demo images for README examples (~10MB total)
  full    - Complete sample dataset for pipeline testing (~50MB total)
        """,
    )

    parser.add_argument("--all", action="store_true", help="Download all sample categories (minimal + demo + full)")
    parser.add_argument("--demo", action="store_true", help="Download demo samples in addition to minimal")
    parser.add_argument("--output-dir", type=Path, help="Output directory (default: repository root)")
    parser.add_argument("--force", action="store_true", help="Force re-download even if files exist")
    parser.add_argument("--list", action="store_true", help="List available samples without downloading")

    args = parser.parse_args()

    # List samples if requested
    if args.list:
        print("\n📋 Available Samples:\n")
        for category in ["minimal", "demo", "full"]:
            samples = get_samples_by_category(category)
            print(f"\n{category.upper()}:")
            for sample in samples:
                if sample.get("synthetic"):
                    status = "🛠  Synthetic (local)"
                elif sample["url"]:
                    status = "✅ Ready"
                else:
                    status = "⏳ Pending v2.4.0"
                print(f"  - {sample['name']:30} ({sample['size']:>6}) {status}")
                print(f"    {sample['description']}")
        print()
        return 0

    # Determine categories to download
    categories = ["minimal"]  # Always download minimal for tests
    if args.demo:
        categories.append("demo")
    if args.all:
        categories = ["minimal", "demo", "full"]

    # Download samples
    downloaded = download_samples(categories, output_dir=args.output_dir, force=args.force)

    if downloaded == 0 and not args.list:
        print("ℹ️  No new files downloaded. Use --force to re-download existing files.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
