"""
Image Loader for Perceptual Baseline Calibration

Handles loading, preprocessing, and metadata extraction for source images.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from torch import Tensor

logger = logging.getLogger(__name__)

# PIL modes whose samples are 16-bit; "I" (32-bit signed) is how Pillow commonly
# opens 16-bit PNG/TIFF data, so it is read on the same 0..65535 scale.
_SIXTEEN_BIT_MODES = frozenset({"I;16", "I;16B", "I;16L", "I;16N"})
_WIDE_INTEGER_MODES = _SIXTEEN_BIT_MODES | {"I"}
_MODE_BIT_DEPTH = {"1": 1, "I": 32, "F": 32, **{mode: 16 for mode in _SIXTEEN_BIT_MODES}}


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
        preserve_aspect: bool = True,
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

    def load(self, image_path: Union[str, Path], image_type: Optional[ImageType] = None) -> tuple[Tensor, ImageMetadata]:
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

        with Image.open(image_path) as source:
            source.load()
            # Source facts are read before any conversion: convert("RGB") drops the
            # format, the mode (bit depth / colour space) and the band count.
            source_mode = source.mode
            source_format = source.format or "unknown"
            source_channels = len(source.getbands())
            source_info = dict(source.info)

            if source_mode in _WIDE_INTEGER_MODES:
                # 16-bit data would saturate in convert("RGB"): scale it onto the
                # 8-bit float range instead and resize it in float.
                rgb_float = self._wide_integer_to_rgb_float(source)
                pil_image = None
            else:
                pil_image = source if source_mode == "RGB" else source.convert("RGB")
                if source_mode != "RGB":
                    logger.debug(f"Converting from {source_mode} to RGB")
                rgb_float = np.asarray(pil_image, dtype=np.float32)

            metadata = self._extract_metadata(
                rgb_float,
                image_path,
                image_type,
                mode=source_mode,
                image_format=source_format,
                channels=source_channels,
                info=source_info,
            )

            # Resize if needed
            if self.target_size is not None:
                if pil_image is not None:
                    pil_image = self._resize_image(pil_image, self.target_size)
                else:
                    rgb_float = self._resize_float_rgb(rgb_float, self.target_size)

            # Convert to a float tensor (C, H, W) on the 0..255 scale
            if pil_image is not None:
                tensor = self._pil_to_tensor(pil_image)
            else:
                tensor = torch.from_numpy(np.ascontiguousarray(rgb_float)).permute(2, 0, 1)

        # Move to device
        tensor = self.substrate.to_device(tensor)

        # Always hand float data downstream; normalize=False keeps the 0..255 scale.
        tensor = tensor.float()
        if self.normalize:
            tensor = tensor / 255.0

        logger.info(
            f"Loaded {metadata.image_type.value if metadata.image_type else 'unknown'} "
            f"image: {metadata.width}x{metadata.height}, "
            f"mean={metadata.mean_intensity:.3f}, std={metadata.std_intensity:.3f}"
        )

        return tensor, metadata

    def load_batch(
        self,
        image_paths: List[Union[str, Path]],
        image_types: Optional[List[ImageType]] = None,
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

    @staticmethod
    def _wide_integer_to_rgb_float(source: Image.Image) -> np.ndarray:
        """Read 16-bit (or ``I``-mode) data as float32 RGB (H, W, 3) on the 0..255 scale."""
        samples = np.asarray(source, dtype=np.float32)
        samples = np.clip(samples, 0.0, 65535.0) * np.float32(255.0 / 65535.0)
        if samples.ndim == 2:
            samples = np.repeat(samples[:, :, None], 3, axis=2)
        return samples

    def _resize_float_rgb(self, rgb_float: np.ndarray, target_size: tuple) -> np.ndarray:
        """Resize a float32 (H, W, 3) array channel-wise in PIL ``F`` mode."""
        channels = [
            self._resize_image(Image.fromarray(np.ascontiguousarray(rgb_float[:, :, c])), target_size) for c in range(3)
        ]
        return np.stack([np.asarray(ch, dtype=np.float32) for ch in channels], axis=2)

    def _extract_metadata(
        self,
        np_image: np.ndarray,
        path: Path,
        image_type: Optional[ImageType],
        *,
        mode: str,
        image_format: str,
        channels: int,
        info: Dict[str, Any],
    ) -> ImageMetadata:
        """Extract metadata from the source image facts and its RGB samples (0..255 scale)."""
        np_image = np.asarray(np_image, dtype=np.float32)

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
        tags: Dict[str, Any] = dict(info)

        metadata = ImageMetadata(
            path=path,
            image_type=image_type,
            width=int(np_image.shape[1]),
            height=int(np_image.shape[0]),
            channels=channels,
            format=image_format,
            size_bytes=file_stat.st_size,
            bit_depth=_MODE_BIT_DEPTH.get(mode, 8),
            color_space=mode,
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

    def _resize_image(self, pil_image: Image.Image, target_size: tuple) -> Image.Image:
        """Resize image with aspect ratio preservation."""
        target_h, target_w = target_size

        if self.preserve_aspect:
            # Calculate scale to fit within target size
            scale = min(target_w / pil_image.width, target_h / pil_image.height)
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

    def create_thumbnail(self, tensor: Tensor, size: tuple = (256, 256)) -> Tensor:
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
            mode="bilinear",
        )

        if thumbnail.shape[0] == 1:
            thumbnail = thumbnail.squeeze(0)

        return thumbnail

    def save_tensor(self, tensor: Tensor, output_path: Union[str, Path], denormalize: bool = True):
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
