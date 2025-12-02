#!/usr/bin/env python3
"""
Flexible Dataset Class for Depth Estimation Training

Supports paired RGB-Depth data with multiple formats including PNG, TIFF, and NPY.
Provides data validation, filtering, and train/val/test split utilities.

Author: Transformation Portal Team
Version: 1.0.0
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# Try to import optional dependencies
try:
    import torch
    from torch.utils.data import Dataset, DataLoader, random_split
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object  # type: ignore

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
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class DepthDataConfig:
    """Configuration for depth dataset.

    Attributes:
        train_dir: Path to training data directory
        val_dir: Path to validation data directory
        test_dir: Optional path to test data directory
        image_size: Target image size (height, width)
        augmentation: Whether to apply augmentations
        normalize: Whether to normalize inputs
        cache_images: Whether to cache loaded images in memory
        min_depth: Minimum valid depth value
        max_depth: Maximum valid depth value
        depth_format: Format of depth files ('png', 'tiff', 'npy')
    """
    train_dir: str = "data/architectural/train"
    val_dir: str = "data/architectural/val"
    test_dir: Optional[str] = None
    image_size: Tuple[int, int] = (518, 518)
    augmentation: bool = True
    normalize: bool = True
    cache_images: bool = False
    min_depth: float = 0.0
    max_depth: float = 1000.0
    depth_format: str = "auto"
    valid_image_extensions: List[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".tiff", ".tif"]
    )
    valid_depth_extensions: List[str] = field(
        default_factory=lambda: [".png", ".tiff", ".tif", ".npy", ".npz"]
    )


class ArchitecturalDepthDataset(Dataset):
    """Dataset for paired RGB-Depth architectural images.

    Supports the following directory structure:
        data/
          train/
            images/
              image_001.jpg
              image_002.png
            depth/
              image_001.png
              image_002.npy
          val/
            images/
            depth/

    Example:
        >>> config = DepthDataConfig(
        ...     train_dir="data/architectural/train",
        ...     image_size=(518, 518)
        ... )
        >>> dataset = ArchitecturalDepthDataset(config, split="train")
        >>> image, depth = dataset[0]
    """

    def __init__(
        self,
        config: DepthDataConfig,
        split: str = "train",
        transform: Optional[Callable] = None,
    ):
        """Initialize dataset.

        Args:
            config: Dataset configuration
            split: Data split ('train', 'val', or 'test')
            transform: Optional transform to apply
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch required for ArchitecturalDepthDataset. "
                "Install with: pip install torch"
            )

        self.config = config
        self.split = split
        self.transform = transform

        # Get data directory for this split
        if split == "train":
            self.data_dir = Path(config.train_dir)
        elif split == "val":
            self.data_dir = Path(config.val_dir)
        elif split == "test":
            if config.test_dir is None:
                raise ValueError("test_dir not specified in config")
            self.data_dir = Path(config.test_dir)
        else:
            raise ValueError(f"Invalid split: {split}")

        # Set up directories
        self.images_dir = self.data_dir / "images"
        self.depth_dir = self.data_dir / "depth"

        # Validate directories
        self._validate_directories()

        # Find all image-depth pairs
        self.pairs = self._find_pairs()

        if len(self.pairs) == 0:
            logger.warning(
                f"No valid image-depth pairs found in {self.data_dir}"
            )

        # Cache for loaded images
        self._cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        logger.info(
            f"Initialized {split} dataset with {len(self.pairs)} pairs"
        )

    def _validate_directories(self) -> None:
        """Validate that data directories exist."""
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}"
            )

        if not self.images_dir.exists():
            # Try alternative structure (images at root)
            self.images_dir = self.data_dir
            logger.warning(
                f"images/ subdirectory not found, using {self.data_dir}"
            )

        if not self.depth_dir.exists():
            raise FileNotFoundError(
                f"Depth directory not found: {self.depth_dir}"
            )

    def _find_pairs(self) -> List[Tuple[Path, Path]]:
        """Find all valid image-depth pairs.

        Returns:
            List of (image_path, depth_path) tuples
        """
        pairs = []

        # Get all image files
        image_files = []
        for ext in self.config.valid_image_extensions:
            image_files.extend(self.images_dir.glob(f"*{ext}"))
            image_files.extend(self.images_dir.glob(f"*{ext.upper()}"))

        # Find corresponding depth files
        for image_path in sorted(image_files):
            depth_path = self._find_depth_file(image_path)
            if depth_path is not None:
                pairs.append((image_path, depth_path))
            else:
                logger.debug(
                    f"No depth file found for {image_path.name}"
                )

        return pairs

    def _find_depth_file(self, image_path: Path) -> Optional[Path]:
        """Find depth file corresponding to an image.

        Args:
            image_path: Path to image file

        Returns:
            Path to depth file, or None if not found
        """
        stem = image_path.stem

        # Try each valid depth extension
        for ext in self.config.valid_depth_extensions:
            depth_path = self.depth_dir / f"{stem}{ext}"
            if depth_path.exists():
                return depth_path

        # Try with _depth suffix
        for ext in self.config.valid_depth_extensions:
            depth_path = self.depth_dir / f"{stem}_depth{ext}"
            if depth_path.exists():
                return depth_path

        return None

    def _load_image(self, path: Path) -> np.ndarray:
        """Load RGB image from file.

        Args:
            path: Path to image file

        Returns:
            RGB image as numpy array (H, W, 3)
        """
        if PIL_AVAILABLE:
            image = Image.open(path).convert("RGB")
            return np.array(image)
        elif CV2_AVAILABLE:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ImportError(
                "PIL or OpenCV required for image loading. "
                "Install with: pip install Pillow or pip install opencv-python"
            )

    def _load_depth(self, path: Path) -> np.ndarray:
        """Load depth map from file.

        Args:
            path: Path to depth file

        Returns:
            Depth map as numpy array (H, W)
        """
        suffix = path.suffix.lower()

        if suffix == ".npy":
            depth = np.load(path)

        elif suffix == ".npz":
            data = np.load(path)
            # Try common keys
            for key in ["depth", "arr_0", "data"]:
                if key in data:
                    depth = data[key]
                    break
            else:
                # Use first array
                depth = data[list(data.keys())[0]]

        elif suffix in [".tiff", ".tif"]:
            if TIFF_AVAILABLE:
                depth = tifffile.imread(path)
            elif PIL_AVAILABLE:
                depth = np.array(Image.open(path))
            else:
                raise ImportError(
                    "tifffile or PIL required for TIFF loading. "
                    "Install with: pip install tifffile"
                )

        elif suffix == ".png":
            if PIL_AVAILABLE:
                depth = np.array(Image.open(path))
            elif CV2_AVAILABLE:
                depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            else:
                raise ImportError(
                    "PIL or OpenCV required for PNG loading."
                )

        else:
            raise ValueError(f"Unsupported depth format: {suffix}")

        # Convert to float32
        depth = depth.astype(np.float32)

        # Handle multi-channel depth (take first channel)
        if depth.ndim == 3:
            depth = depth[:, :, 0]

        return depth

    def _resize(
        self,
        array: np.ndarray,
        target_size: Tuple[int, int],
        is_depth: bool = False,
    ) -> np.ndarray:
        """Resize array to target size.

        Args:
            array: Input array
            target_size: Target (height, width)
            is_depth: Whether this is a depth map

        Returns:
            Resized array
        """
        target_h, target_w = target_size
        current_h, current_w = array.shape[:2]

        if (current_h, current_w) == target_size:
            return array

        if PIL_AVAILABLE:
            if array.ndim == 3:
                pil_img = Image.fromarray(array)
            else:
                # Depth map
                if array.dtype == np.float32:
                    # Scale for uint16 precision
                    scaled = ((array - array.min()) /
                              (array.max() - array.min() + 1e-8) * 65535)
                    pil_img = Image.fromarray(scaled.astype(np.uint16))
                else:
                    pil_img = Image.fromarray(array)

            resample = Image.Resampling.NEAREST if is_depth else Image.Resampling.BILINEAR
            resized = pil_img.resize((target_w, target_h), resample=resample)
            result = np.array(resized)

            if is_depth and array.dtype == np.float32:
                # Rescale back
                result = result.astype(np.float32) / 65535.0
                result = result * (array.max() - array.min()) + array.min()

            return result

        elif CV2_AVAILABLE:
            interp = cv2.INTER_NEAREST if is_depth else cv2.INTER_LINEAR
            return cv2.resize(array, (target_w, target_h), interpolation=interp)

        else:
            raise ImportError("PIL or OpenCV required for resizing")

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.pairs)

    def __getitem__(
        self, idx: int
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image_tensor, depth_tensor)
        """
        # Check cache
        if self.config.cache_images and idx in self._cache:
            image, depth = self._cache[idx]
        else:
            image_path, depth_path = self.pairs[idx]

            # Load image and depth
            image = self._load_image(image_path)
            depth = self._load_depth(depth_path)

            # Resize to target size
            image = self._resize(image, self.config.image_size, is_depth=False)
            depth = self._resize(depth, self.config.image_size, is_depth=True)

            # Cache if enabled
            if self.config.cache_images:
                self._cache[idx] = (image, depth)

        # Apply transform
        if self.transform is not None:
            image, depth = self.transform(image, depth)

        # Convert to tensors
        image = self._to_tensor(image, is_depth=False)
        depth = self._to_tensor(depth, is_depth=True)

        return image, depth

    def _to_tensor(
        self,
        array: np.ndarray,
        is_depth: bool = False,
    ) -> "torch.Tensor":
        """Convert numpy array to tensor.

        Args:
            array: Input array
            is_depth: Whether this is a depth map

        Returns:
            PyTorch tensor
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for tensor conversion")

        # Ensure float32
        array = array.astype(np.float32)

        if is_depth:
            # Depth: (H, W) -> (1, H, W)
            if array.ndim == 2:
                array = array[np.newaxis, ...]
            return torch.from_numpy(array.copy())
        else:
            # Image: normalize to [0, 1] if needed
            if array.max() > 1.0:
                array = array / 255.0

            # (H, W, 3) -> (3, H, W)
            array = array.transpose(2, 0, 1)
            return torch.from_numpy(array.copy())

    def get_stats(self) -> Dict:
        """Compute dataset statistics.

        Returns:
            Dictionary with dataset statistics
        """
        depth_mins = []
        depth_maxs = []
        depth_means = []

        for idx in range(min(len(self), 100)):  # Sample first 100
            _, depth_path = self.pairs[idx]
            depth = self._load_depth(depth_path)
            depth_mins.append(depth.min())
            depth_maxs.append(depth.max())
            depth_means.append(depth.mean())

        return {
            "num_samples": len(self),
            "split": self.split,
            "image_size": self.config.image_size,
            "depth_min": float(np.mean(depth_mins)),
            "depth_max": float(np.mean(depth_maxs)),
            "depth_mean": float(np.mean(depth_means)),
        }


def create_data_loaders(
    config: DepthDataConfig,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    val_split: Optional[float] = None,
) -> Dict[str, "DataLoader"]:
    """Create data loaders for training and validation.

    Args:
        config: Dataset configuration
        train_transform: Transform for training data
        val_transform: Transform for validation data
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for GPU transfer
        val_split: If provided, split training data (0.0-1.0 fraction for val)

    Returns:
        Dictionary with 'train' and 'val' DataLoader objects
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch required for data loaders. "
            "Install with: pip install torch"
        )

    loaders = {}

    # Create training dataset
    train_dataset = ArchitecturalDepthDataset(
        config, split="train", transform=train_transform
    )

    # Handle validation split from training data
    if val_split is not None and val_split > 0:
        val_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - val_size

        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size]
        )

        # Wrap val_dataset to use val_transform
        # Note: This is a simplified approach
        logger.info(
            f"Split training data: {train_size} train, {val_size} val"
        )
    else:
        # Create separate validation dataset
        try:
            val_dataset = ArchitecturalDepthDataset(
                config, split="val", transform=val_transform
            )
        except FileNotFoundError:
            logger.warning(
                "Validation directory not found, using 10% of training data"
            )
            val_size = int(len(train_dataset) * 0.1)
            train_size = len(train_dataset) - val_size
            train_dataset, val_dataset = random_split(
                train_dataset, [train_size, val_size]
            )

    # Create data loaders
    loaders["train"] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    loaders["val"] = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    logger.info(
        f"Created data loaders: train={len(loaders['train'])} batches, "
        f"val={len(loaders['val'])} batches"
    )

    return loaders


def split_dataset(
    dataset: "ArchitecturalDepthDataset",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple["Dataset", "Dataset", "Dataset"]:
    """Split dataset into train/val/test sets.

    Args:
        dataset: Input dataset
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for dataset splitting")

    # Validate ratios
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0, got {total}"
        )

    # Set seed for reproducibility
    generator = torch.Generator().manual_seed(seed)

    # Calculate sizes
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    # Split
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )

    logger.info(
        f"Split dataset: train={train_size}, val={val_size}, test={test_size}"
    )

    return train_dataset, val_dataset, test_dataset
