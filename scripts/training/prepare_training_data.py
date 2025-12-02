#!/usr/bin/env python3
"""
Training Data Preparation Script

Prepares architectural depth estimation training data:
- Downloads sample datasets (optional)
- Converts depth maps to consistent format
- Creates train/val/test splits
- Validates data integrity
- Computes dataset statistics

Usage:
    python scripts/training/prepare_training_data.py \\
        --source-dir /path/to/raw/data \\
        --output-dir data/architectural \\
        --val-split 0.1 \\
        --test-split 0.1

Author: Transformation Portal Team
Version: 1.0.0
"""

import argparse
import hashlib
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Try to import dependencies
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import tifffile
    TIFF_AVAILABLE = True
except ImportError:
    TIFF_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Supported file formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}
DEPTH_EXTENSIONS = {".png", ".tiff", ".tif", ".npy", ".npz", ".exr"}


def validate_dependencies() -> bool:
    """Validate required dependencies are installed.

    Returns:
        True if all dependencies are available
    """
    if not PIL_AVAILABLE:
        logger.error("PIL (Pillow) is required. Install with: pip install Pillow")
        return False
    return True


def find_image_depth_pairs(
    source_dir: Path,
    images_subdir: str = "images",
    depth_subdir: str = "depth",
) -> List[Tuple[Path, Path]]:
    """Find matching image-depth pairs in source directory.

    Args:
        source_dir: Source directory
        images_subdir: Subdirectory for images
        depth_subdir: Subdirectory for depth maps

    Returns:
        List of (image_path, depth_path) tuples
    """
    images_dir = source_dir / images_subdir
    depth_dir = source_dir / depth_subdir

    if not images_dir.exists():
        # Try flat structure
        images_dir = source_dir
        logger.info(f"Using flat structure, looking for images in {source_dir}")

    if not depth_dir.exists():
        logger.error(f"Depth directory not found: {depth_dir}")
        return []

    pairs = []

    # Find all image files
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(images_dir.glob(f"*{ext}"))
        image_files.extend(images_dir.glob(f"*{ext.upper()}"))

    logger.info(f"Found {len(image_files)} image files")

    # Find matching depth files
    for image_path in image_files:
        stem = image_path.stem

        # Try each depth extension
        depth_path = None
        for ext in DEPTH_EXTENSIONS:
            candidate = depth_dir / f"{stem}{ext}"
            if candidate.exists():
                depth_path = candidate
                break

            # Try with _depth suffix
            candidate = depth_dir / f"{stem}_depth{ext}"
            if candidate.exists():
                depth_path = candidate
                break

        if depth_path is not None:
            pairs.append((image_path, depth_path))
        else:
            logger.debug(f"No depth file found for {image_path.name}")

    logger.info(f"Found {len(pairs)} valid image-depth pairs")
    return pairs


def load_depth(path: Path) -> np.ndarray:
    """Load depth map from various formats.

    Args:
        path: Path to depth file

    Returns:
        Depth map as float32 numpy array
    """
    suffix = path.suffix.lower()

    if suffix == ".npy":
        depth = np.load(path)
    elif suffix == ".npz":
        data = np.load(path)
        for key in ["depth", "arr_0", "data"]:
            if key in data:
                depth = data[key]
                break
        else:
            depth = data[list(data.keys())[0]]
    elif suffix in [".tiff", ".tif"]:
        if TIFF_AVAILABLE:
            depth = tifffile.imread(path)
        else:
            depth = np.array(Image.open(path))
    elif suffix == ".png":
        depth = np.array(Image.open(path))
    elif suffix == ".exr":
        try:
            import cv2
            depth = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
        except ImportError:
            raise ImportError("OpenCV required for EXR files. Install with: pip install opencv-python")
    else:
        raise ValueError(f"Unsupported depth format: {suffix}")

    # Convert to float32
    depth = depth.astype(np.float32)

    # Handle multi-channel depth
    if depth.ndim == 3:
        depth = depth[:, :, 0]

    return depth


def save_depth(depth: np.ndarray, path: Path, format: str = "npy") -> None:
    """Save depth map in specified format.

    Args:
        depth: Depth map array
        path: Output path
        format: Output format ('npy', 'png', 'tiff')
    """
    path = path.with_suffix(f".{format}")

    if format == "npy":
        np.save(path, depth.astype(np.float32))
    elif format == "png":
        # Normalize to 16-bit for PNG
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 0:
            depth_norm = (depth - d_min) / (d_max - d_min)
        else:
            depth_norm = np.zeros_like(depth)
        depth_16bit = (depth_norm * 65535).astype(np.uint16)
        Image.fromarray(depth_16bit).save(path)
    elif format in ["tiff", "tif"]:
        if TIFF_AVAILABLE:
            tifffile.imwrite(path, depth.astype(np.float32))
        else:
            # Fallback to PIL (less precision)
            depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            depth_16bit = (depth_norm * 65535).astype(np.uint16)
            Image.fromarray(depth_16bit).save(path)
    else:
        raise ValueError(f"Unsupported output format: {format}")


def compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of file.

    Args:
        path: Path to file

    Returns:
        Hex digest of hash
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def split_data(
    pairs: List[Tuple[Path, Path]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[Tuple[Path, Path]]]:
    """Split data into train/val/test sets.

    Args:
        pairs: List of (image, depth) path tuples
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed

    Returns:
        Dictionary with 'train', 'val', 'test' lists
    """
    # Validate ratios
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total}")

    # Shuffle with seed
    np.random.seed(seed)
    indices = np.random.permutation(len(pairs))

    # Calculate split points
    n = len(pairs)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    # Split
    splits = {
        "train": [pairs[i] for i in indices[:train_end]],
        "val": [pairs[i] for i in indices[train_end:val_end]],
        "test": [pairs[i] for i in indices[val_end:]],
    }

    logger.info(
        f"Split data: train={len(splits['train'])}, "
        f"val={len(splits['val'])}, test={len(splits['test'])}"
    )

    return splits


def copy_pairs(
    pairs: List[Tuple[Path, Path]],
    output_dir: Path,
    depth_format: str = "npy",
    resize: Optional[Tuple[int, int]] = None,
) -> int:
    """Copy image-depth pairs to output directory.

    Args:
        pairs: List of (image, depth) path tuples
        output_dir: Output directory
        depth_format: Output depth format
        resize: Optional (height, width) to resize to

    Returns:
        Number of pairs copied
    """
    images_dir = output_dir / "images"
    depth_dir = output_dir / "depth"

    images_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    iterator = tqdm(pairs, desc="Copying") if TQDM_AVAILABLE else pairs
    copied = 0

    for image_path, depth_path in iterator:
        try:
            stem = image_path.stem

            # Copy/process image
            image = Image.open(image_path).convert("RGB")
            if resize:
                image = image.resize((resize[1], resize[0]), Image.Resampling.BILINEAR)
            output_image_path = images_dir / f"{stem}.png"
            image.save(output_image_path)

            # Copy/process depth
            depth = load_depth(depth_path)
            if resize:
                # Resize depth with nearest neighbor
                depth_pil = Image.fromarray(depth)
                depth_pil = depth_pil.resize((resize[1], resize[0]), Image.Resampling.NEAREST)
                depth = np.array(depth_pil)

            output_depth_path = depth_dir / stem
            save_depth(depth, output_depth_path, format=depth_format)

            copied += 1

        except Exception as e:
            logger.error(f"Error processing {image_path.name}: {e}")

    return copied


def compute_statistics(pairs: List[Tuple[Path, Path]]) -> Dict:
    """Compute dataset statistics.

    Args:
        pairs: List of (image, depth) path tuples

    Returns:
        Dictionary of statistics
    """
    depth_mins = []
    depth_maxs = []
    depth_means = []
    image_sizes = []

    iterator = tqdm(pairs, desc="Computing stats") if TQDM_AVAILABLE else pairs

    for image_path, depth_path in iterator:
        try:
            # Image stats
            image = Image.open(image_path)
            image_sizes.append(image.size)

            # Depth stats
            depth = load_depth(depth_path)
            depth_mins.append(float(depth.min()))
            depth_maxs.append(float(depth.max()))
            depth_means.append(float(depth.mean()))

        except Exception as e:
            logger.warning(f"Error computing stats for {image_path.name}: {e}")

    stats = {
        "num_samples": len(pairs),
        "depth_min": float(np.mean(depth_mins)),
        "depth_max": float(np.mean(depth_maxs)),
        "depth_mean": float(np.mean(depth_means)),
        "depth_min_overall": float(min(depth_mins)) if depth_mins else 0,
        "depth_max_overall": float(max(depth_maxs)) if depth_maxs else 0,
    }

    if image_sizes:
        widths = [s[0] for s in image_sizes]
        heights = [s[1] for s in image_sizes]
        stats["image_width_mean"] = float(np.mean(widths))
        stats["image_height_mean"] = float(np.mean(heights))

    return stats


def create_sample_data(output_dir: Path, num_samples: int = 100) -> None:
    """Create synthetic sample data for testing.

    Args:
        output_dir: Output directory
        num_samples: Number of samples to create
    """
    logger.info(f"Creating {num_samples} synthetic samples...")

    images_dir = output_dir / "images"
    depth_dir = output_dir / "depth"

    images_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    iterator = range(num_samples)
    if TQDM_AVAILABLE:
        iterator = tqdm(iterator, desc="Generating")

    for i in iterator:
        # Create synthetic image (simple architectural pattern)
        h, w = 518, 518
        image = np.zeros((h, w, 3), dtype=np.uint8)

        # Sky gradient
        sky_h = h // 3
        for y in range(sky_h):
            intensity = int(200 - y * 0.5)
            image[y, :, 2] = intensity  # Blue
            image[y, :, 1] = int(intensity * 0.7)
            image[y, :, 0] = int(intensity * 0.4)

        # Building
        building_color = [220, 215, 200]
        image[sky_h:, :, :] = building_color

        # Windows
        for wx in range(50, w - 50, 100):
            for wy in range(sky_h + 30, h - 30, 80):
                image[wy:wy + 50, wx:wx + 60, :] = [80, 100, 120]

        # Create corresponding depth map
        depth = np.zeros((h, w), dtype=np.float32)

        # Sky is far
        depth[:sky_h, :] = 100.0

        # Building is closer
        depth[sky_h:, :] = 20.0

        # Windows are slightly recessed
        for wx in range(50, w - 50, 100):
            for wy in range(sky_h + 30, h - 30, 80):
                depth[wy:wy + 50, wx:wx + 60] = 25.0

        # Add noise for realism
        depth += np.random.randn(h, w).astype(np.float32) * 0.5

        # Save
        Image.fromarray(image).save(images_dir / f"sample_{i:04d}.png")
        np.save(depth_dir / f"sample_{i:04d}.npy", depth)

    logger.info(f"Created {num_samples} synthetic samples in {output_dir}")


def main(args: argparse.Namespace) -> int:
    """Main function.

    Args:
        args: Command line arguments

    Returns:
        Exit code
    """
    if not validate_dependencies():
        return 1

    output_dir = Path(args.output_dir)

    # Create sample data if requested
    if args.create_sample:
        create_sample_data(output_dir / "train", num_samples=args.num_samples)
        create_sample_data(output_dir / "val", num_samples=max(10, args.num_samples // 10))
        logger.info("Sample data creation complete!")
        return 0

    # Validate source directory
    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        logger.error(f"Source directory not found: {source_dir}")
        return 1

    # Find pairs
    pairs = find_image_depth_pairs(source_dir)
    if not pairs:
        logger.error("No valid image-depth pairs found")
        return 1

    # Compute statistics
    if args.stats_only:
        stats = compute_statistics(pairs)
        print("\nDataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        return 0

    # Split data
    splits = split_data(
        pairs,
        train_ratio=1.0 - args.val_split - args.test_split,
        val_ratio=args.val_split,
        test_ratio=args.test_split,
        seed=args.seed,
    )

    # Copy data to output directory
    resize = tuple(args.resize) if args.resize else None

    for split_name, split_pairs in splits.items():
        if split_pairs:
            split_dir = output_dir / split_name
            logger.info(f"Processing {split_name} split...")
            copied = copy_pairs(
                split_pairs,
                split_dir,
                depth_format=args.depth_format,
                resize=resize,
            )
            logger.info(f"Copied {copied} pairs to {split_dir}")

    # Compute and save statistics
    all_pairs = splits["train"] + splits["val"] + splits["test"]
    stats = compute_statistics(all_pairs)

    stats_path = output_dir / "dataset_stats.txt"
    with open(stats_path, "w") as f:
        f.write("Dataset Statistics\n")
        f.write("=" * 40 + "\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")

    logger.info(f"Dataset statistics saved to {stats_path}")
    logger.info("Data preparation complete!")

    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Prepare training data for depth estimation"
    )

    parser.add_argument(
        "--source-dir",
        type=str,
        default="data/raw",
        help="Source directory containing raw data"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/architectural",
        help="Output directory for prepared data"
    )

    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of data for validation"
    )

    parser.add_argument(
        "--test-split",
        type=float,
        default=0.1,
        help="Fraction of data for testing"
    )

    parser.add_argument(
        "--depth-format",
        type=str,
        default="npy",
        choices=["npy", "png", "tiff"],
        help="Output depth format"
    )

    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=None,
        metavar=("HEIGHT", "WIDTH"),
        help="Resize images to specified size"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting"
    )

    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only compute statistics, don't copy data"
    )

    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create synthetic sample data for testing"
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of synthetic samples to create"
    )

    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = parse_args()
        sys.exit(main(args))
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
