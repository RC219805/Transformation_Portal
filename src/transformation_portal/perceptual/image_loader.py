"""
Image Loader for Perceptual Baseline Calibration

Handles loading, preprocessing, and metadata extraction for source images.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from enum import Enum
import logging

import torch
from torch import Tensor
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class ImageType(Enum):
    """Types of architectural images."""
    POOL = "pool"
    BEDROOMS = "bedrooms"
    BATHROOM = "bathroom"
    AERIAL = "aerial"
    KITCHEN = "kitchen"
    GREAT_ROOM = "great_room"


@dataclass
class ImageMetadata:
    """Metadata for loaded image."""
    path: Path
    image_type: Optional[ImageType]
    width: int
    height: int
    channels: int
    format: str
    size_bytes: int
    bit_depth: int
    color_space: str

    # Image statistics
    mean_intensity: float
    std_intensity: float
    dynamic_range: float

    # Additional metadata
    tags: Dict[str, Any]


class ImageLoader:
    """
    Image loader optimized for perceptual baseline calibration.

    Loads source images with proper preprocessing, normalization,
    and metadata extraction.
    """

    def __init__(
        self,
        substrate,
        target_size: Optional[tuple] = None,
        normalize: bool = True,
        preserve_aspect: bool = True
    ):
        """
        Initialize image loader.

        Args:
            substrate: Computational substrate from Phase 1
            target_size: Target size (H, W) or None to preserve original
            normalize: Whether to normalize to [0, 1]
            preserve_aspect: Preserve aspect ratio when resizing
        """
        self.substrate = substrate
        self.target_size = target_size
        self.normalize = normalize
        self.preserve_aspect = preserve_aspect

        # Image type mapping from filename patterns
        self.type_patterns = {
            ImageType.POOL: ["pool", "swimming"],
            ImageType.BEDROOMS: ["bedroom", "bed"],
            ImageType.BATHROOM: ["bathroom", "bath"],
            ImageType.AERIAL: ["aerial", "drone", "overhead", "top"],
            ImageType.KITCHEN: ["kitchen"],
            ImageType.GREAT_ROOM: ["great_room", "living", "greatroom"],
        }

        logger.info(f"Initialized ImageLoader with target_size={target_size}")

    def load(
        self,
        image_path: Union[str, Path],
        image_type: Optional[ImageType] = None
    ) -> tuple[Tensor, ImageMetadata]:
        """
        Load image with metadata extraction.

        Args:
            image_path: Path to image file
            image_type: Type of image (auto-detected if None)

        Returns:
            Tuple of (tensor, metadata)
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        logger.info(f"Loading image: {image_path.name}")

        # Load with PIL
        pil_image = Image.open(image_path)

        # Convert to RGB if needed
        if pil_image.mode != "RGB":
            logger.debug(f"Converting from {pil_image.mode} to RGB")
            pil_image = pil_image.convert("RGB")

        # Extract metadata before processing
        metadata = self._extract_metadata(pil_image, image_path, image_type)

        # Resize if needed
        if self.target_size is not None:
            pil_image = self._resize_image(pil_image, self.target_size)

        # Convert to tensor
        tensor = self._pil_to_tensor(pil_image)

        # Move to device
        tensor = self.substrate.to_device(tensor)

        # Normalize if requested
        if self.normalize:
            tensor = tensor.float() / 255.0

        logger.info(
            f"Loaded {metadata.image_type.value if metadata.image_type else 'unknown'} "
            f"image: {metadata.width}x{metadata.height}, "
            f"mean={metadata.mean_intensity:.3f}, std={metadata.std_intensity:.3f}"
        )

        return tensor, metadata

    def load_batch(
        self,
        image_paths: List[Union[str, Path]],
        image_types: Optional[List[ImageType]] = None
    ) -> tuple[List[Tensor], List[ImageMetadata]]:
        """
        Load batch of images.

        Args:
            image_paths: List of image paths
            image_types: List of image types (auto-detected if None)

        Returns:
            Tuple of (tensors, metadatas)
        """
        if image_types is None:
            image_types = [None] * len(image_paths)

        tensors = []
        metadatas = []

        for path, img_type in zip(image_paths, image_types):
            tensor, metadata = self.load(path, img_type)
            tensors.append(tensor)
            metadatas.append(metadata)

        return tensors, metadatas

    def _extract_metadata(
        self,
        pil_image: Image.Image,
        path: Path,
        image_type: Optional[ImageType]
    ) -> ImageMetadata:
        """Extract metadata from PIL image."""
        # Convert to numpy for statistics
        np_image = np.array(pil_image).astype(np.float32)

        # Auto-detect image type if not provided
        if image_type is None:
            image_type = self._detect_image_type(path.name)

        # Calculate statistics
        mean_intensity = np_image.mean() / 255.0
        std_intensity = np_image.std() / 255.0
        dynamic_range = (np_image.max() - np_image.min()) / 255.0

        # Get file info
        file_stat = path.stat()

        # Extract EXIF/metadata tags
        tags = {}
        if hasattr(pil_image, 'info'):
            tags.update(pil_image.info)

        metadata = ImageMetadata(
            path=path,
            image_type=image_type,
            width=pil_image.width,
            height=pil_image.height,
            channels=len(pil_image.getbands()),
            format=pil_image.format or "unknown",
            size_bytes=file_stat.st_size,
            bit_depth=8 if pil_image.mode == "RGB" else 16,
            color_space=pil_image.mode,
            mean_intensity=mean_intensity,
            std_intensity=std_intensity,
            dynamic_range=dynamic_range,
            tags=tags,
        )

        return metadata

    def _detect_image_type(self, filename: str) -> Optional[ImageType]:
        """Auto-detect image type from filename."""
        filename_lower = filename.lower()

        for img_type, patterns in self.type_patterns.items():
            for pattern in patterns:
                if pattern in filename_lower:
                    return img_type

        logger.warning(f"Could not detect image type from filename: {filename}")
        return None

    def _resize_image(
        self,
        pil_image: Image.Image,
        target_size: tuple
    ) -> Image.Image:
        """Resize image with aspect ratio preservation."""
        target_h, target_w = target_size

        if self.preserve_aspect:
            # Calculate scale to fit within target size
            scale = min(
                target_w / pil_image.width,
                target_h / pil_image.height
            )
            new_w = int(pil_image.width * scale)
            new_h = int(pil_image.height * scale)
        else:
            new_w = target_w
            new_h = target_h

        # Use high-quality resampling
        resized = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        return resized

    def _pil_to_tensor(self, pil_image: Image.Image) -> Tensor:
        """Convert PIL image to tensor (C, H, W)."""
        np_image = np.array(pil_image)

        # Convert to (C, H, W)
        if np_image.ndim == 2:
            # Grayscale
            tensor = torch.from_numpy(np_image).unsqueeze(0)
        else:
            # RGB
            tensor = torch.from_numpy(np_image).permute(2, 0, 1)

        return tensor

    def create_thumbnail(
        self,
        tensor: Tensor,
        size: tuple = (256, 256)
    ) -> Tensor:
        """
        Create thumbnail from tensor.

        Args:
            tensor: Input tensor (C, H, W)
            size: Thumbnail size (H, W)

        Returns:
            Thumbnail tensor
        """
        # Use tensor processor for resizing
        thumbnail = self.substrate.tensor_processor.resize(
            tensor.unsqueeze(0) if tensor.ndim == 3 else tensor,
            size=size,
            mode="bilinear"
        )

        if thumbnail.shape[0] == 1:
            thumbnail = thumbnail.squeeze(0)

        return thumbnail

    def save_tensor(
        self,
        tensor: Tensor,
        output_path: Union[str, Path],
        denormalize: bool = True
    ):
        """
        Save tensor as image.

        Args:
            tensor: Tensor to save (C, H, W)
            output_path: Output file path
            denormalize: Whether to denormalize from [0, 1] to [0, 255]
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Move to CPU
        tensor_cpu = tensor.cpu()

        # Denormalize if needed
        if denormalize:
            tensor_cpu = (tensor_cpu * 255).clamp(0, 255).byte()
        else:
            tensor_cpu = tensor_cpu.clamp(0, 255).byte()

        # Convert to numpy (H, W, C)
        if tensor_cpu.ndim == 3:
            np_image = tensor_cpu.permute(1, 2, 0).numpy()
        else:
            np_image = tensor_cpu.numpy()

        # Save with PIL
        pil_image = Image.fromarray(np_image)
        pil_image.save(output_path)

        logger.info(f"Saved image to {output_path}")

    def get_image_stats(self, tensor: Tensor) -> Dict[str, float]:
        """
        Get statistics for tensor.

        Args:
            tensor: Input tensor

        Returns:
            Dictionary of statistics
        """
        tensor_cpu = tensor.cpu().float()

        if not self.normalize:
            tensor_cpu = tensor_cpu / 255.0

        return {
            "mean": tensor_cpu.mean().item(),
            "std": tensor_cpu.std().item(),
            "min": tensor_cpu.min().item(),
            "max": tensor_cpu.max().item(),
            "dynamic_range": (tensor_cpu.max() - tensor_cpu.min()).item(),
        }
