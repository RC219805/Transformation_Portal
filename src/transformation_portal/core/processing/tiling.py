"""
Tiling utilities for ultra-high-resolution image processing.

Provides memory-efficient processing of large images by splitting into tiles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple, List
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    F = None


@dataclass
class TileConfig:
    """Configuration for tiled processing."""
    tile_size: int = 512
    overlap: int = 64
    blend_mode: str = "linear"  # "linear", "gaussian", "none"
    min_tile_size: int = 256

    def __post_init__(self):
        """Validate configuration."""
        if self.overlap >= self.tile_size:
            raise ValueError(f"Overlap ({self.overlap}) must be less than tile size ({self.tile_size})")

        if self.tile_size < self.min_tile_size:
            raise ValueError(f"Tile size ({self.tile_size}) must be at least {self.min_tile_size}")

        if self.blend_mode not in ("linear", "gaussian", "none"):
            raise ValueError(f"Invalid blend_mode: {self.blend_mode}")


class TiledProcessor:
    """
    Process ultra-high-resolution images with tiling and blending.

    Splits large images into overlapping tiles, processes each tile,
    and blends results to avoid seam artifacts.

    Example:
        >>> processor = TiledProcessor(tile_size=512, overlap=64)
        >>> result = processor.process(large_image, model_fn)
    """

    def __init__(
        self,
        tile_size: int = 512,
        overlap: int = 64,
        blend_mode: str = "linear"
    ):
        """
        Initialize tiled processor.

        Args:
            tile_size: Size of each tile (pixels)
            overlap: Overlap between tiles (pixels)
            blend_mode: Blending strategy ("linear", "gaussian", "none")
        """
        if not TORCH_AVAILABLE:
            raise ImportError("TiledProcessor requires torch")

        self.config = TileConfig(
            tile_size=tile_size,
            overlap=overlap,
            blend_mode=blend_mode
        )

    def process(
        self,
        image: torch.Tensor,
        processor_fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """
        Process image with tiling.

        Args:
            image: Input tensor [B, C, H, W] or [C, H, W]
            processor_fn: Function to process each tile

        Returns:
            Processed tensor with same shape as input
        """
        if not TORCH_AVAILABLE:
            raise ImportError("TiledProcessor requires torch")

        # Handle single image or batch
        if image.ndim == 3:
            image = image.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        _, _, h, w = image.shape

        # Check if tiling is needed
        if h <= self.config.tile_size and w <= self.config.tile_size:
            logger.debug(f"Image {h}x{w} fits in single tile, processing directly")
            result = processor_fn(image)
        else:
            logger.debug(
                f"Processing {h}x{w} image with tiling "
                f"(tile_size={self.config.tile_size}, overlap={self.config.overlap})"
            )
            result = self._process_tiled(image, processor_fn)

        if squeeze_output:
            result = result.squeeze(0)

        return result

    def _process_tiled(
        self,
        image: torch.Tensor,
        processor_fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """Process image in tiles with blending."""
        _, c, h, w = image.shape

        # Calculate tile positions
        tiles = self._calculate_tiles(h, w)

        logger.debug(f"Processing {len(tiles)} tiles")

        # Initialize output and weight accumulator
        device = image.device
        dtype = image.dtype
        output = torch.zeros(1, c, h, w, device=device, dtype=dtype)
        weight = torch.zeros(1, 1, h, w, device=device, dtype=dtype)

        # Process each tile
        for i, (y1, y2, x1, x2) in enumerate(tiles):
            # Extract tile
            tile = image[:, :, y1:y2, x1:x2]

            # Process tile
            processed_tile = processor_fn(tile)

            # Create blend weight
            tile_h, tile_w = y2 - y1, x2 - x1
            tile_weight = self._create_blend_weight(tile_h, tile_w, device)

            # Accumulate
            output[:, :, y1:y2, x1:x2] += processed_tile * tile_weight
            weight[:, :, y1:y2, x1:x2] += tile_weight

        # Normalize by weight
        output = output / weight.clamp(min=1e-8)

        return output

    def _calculate_tiles(self, h: int, w: int) -> List[Tuple[int, int, int, int]]:
        """
        Calculate tile positions with overlap.

        Returns:
            List of (y1, y2, x1, x2) tuples
        """
        tile_size = self.config.tile_size
        overlap = self.config.overlap
        stride = tile_size - overlap

        tiles = []

        # Calculate tile positions
        y_positions = list(range(0, h - overlap, stride))
        x_positions = list(range(0, w - overlap, stride))

        # Ensure we cover the entire image
        if not y_positions or y_positions[-1] + tile_size < h:
            y_positions.append(h - tile_size)

        if not x_positions or x_positions[-1] + tile_size < w:
            x_positions.append(w - tile_size)

        # Generate all tile coordinates
        for y in y_positions:
            for x in x_positions:
                y1 = max(0, y)
                y2 = min(h, y + tile_size)
                x1 = max(0, x)
                x2 = min(w, x + tile_size)

                tiles.append((y1, y2, x1, x2))

        return tiles

    def _create_blend_weight(
        self,
        h: int,
        w: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Create blend weight for tile.

        Args:
            h: Tile height
            w: Tile width
            device: Target device

        Returns:
            Weight tensor [1, 1, h, w]
        """
        if self.config.blend_mode == "none":
            return torch.ones(1, 1, h, w, device=device)

        elif self.config.blend_mode == "linear":
            # Linear ramp from edges
            y_weight = torch.linspace(0, 1, h, device=device)
            y_weight = torch.minimum(y_weight, torch.flip(y_weight, [0]))

            x_weight = torch.linspace(0, 1, w, device=device)
            x_weight = torch.minimum(x_weight, torch.flip(x_weight, [0]))

            weight = y_weight.unsqueeze(1) * x_weight.unsqueeze(0)
            return weight.unsqueeze(0).unsqueeze(0)

        elif self.config.blend_mode == "gaussian":
            # Gaussian falloff from center
            y_center = h / 2
            x_center = w / 2
            sigma_y = h / 4
            sigma_x = w / 4

            y = torch.arange(h, device=device, dtype=torch.float32)
            x = torch.arange(w, device=device, dtype=torch.float32)

            y_dist = ((y - y_center) / sigma_y) ** 2
            x_dist = ((x - x_center) / sigma_x) ** 2

            weight = torch.exp(-(y_dist.unsqueeze(1) + x_dist.unsqueeze(0)) / 2)
            return weight.unsqueeze(0).unsqueeze(0)

        else:
            raise ValueError(f"Invalid blend_mode: {self.config.blend_mode}")

    def estimate_tiles(self, h: int, w: int) -> int:
        """
        Estimate number of tiles needed for given dimensions.

        Args:
            h: Image height
            w: Image width

        Returns:
            Number of tiles
        """
        if h <= self.config.tile_size and w <= self.config.tile_size:
            return 1

        return len(self._calculate_tiles(h, w))
