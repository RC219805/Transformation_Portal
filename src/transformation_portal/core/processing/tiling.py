"""
Tiled Image Processing Utilities.

Enables processing of high-resolution images that exceed GPU memory limits
by splitting them into overlapping tiles, processing them independently,
and blending them back together seamlessly.
"""

import logging
import math
from dataclasses import dataclass
from typing import Callable, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class TileConfig:
    """Configuration for tiled processing."""

    tile_size: int = 512
    tile_overlap: int = 64
    batch_size: int = 4

    # Weighting strategy for blending overlaps
    # 'gaussian' is smoother, 'linear' is faster
    blend_mode: str = "gaussian"


class TiledProcessor:
    """
    Engine for seamless tiled image processing.

    Handles the complexity of:
    1. Padding images to fit tile multiples.
    2. Extracting overlapping crops.
    3. Batching tiles for efficient GPU usage.
    4. Recombining tiles with weighted blending to hide seams.
    """

    def __init__(self, config: TileConfig):
        self.config = config

    def process_image(
        self,
        image: Union[np.ndarray, torch.Tensor],
        processor_func: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device = torch.device("cpu"),
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Process a large image using tiling.

        Args:
            image: Input image (H, W, C) numpy or (B, C, H, W) tensor.
            processor_func: Function that takes a batch of tiles and returns processed tiles.
            device: Device to perform merging on.

        Returns:
            Processed image in same format as input.
        """
        # Normalize to (B, C, H, W) tensor
        if isinstance(image, np.ndarray):
            is_numpy = True
            # (H, W, C) -> (1, C, H, W)
            img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        else:
            is_numpy = False
            img_tensor = image

        img_tensor = img_tensor.to(device)
        b, c, h, w = img_tensor.shape

        # 1. Calculate Padding
        # We need the image size to be covered by tiles
        # Stride = size - overlap
        stride = self.config.tile_size - self.config.tile_overlap

        h_tiles = math.ceil((h - self.config.tile_overlap) / stride)
        w_tiles = math.ceil((w - self.config.tile_overlap) / stride)

        pad_h = (h_tiles * stride + self.config.tile_overlap) - h
        pad_w = (w_tiles * stride + self.config.tile_overlap) - w

        # Reflect pad to avoid border artifacts
        padded_img = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode="reflect")

        # 2. Extract Tiles
        tiles: List[torch.Tensor] = []
        coords: List[Tuple[int, int]] = []

        for i in range(h_tiles):
            for j in range(w_tiles):
                y = i * stride
                x = j * stride

                tile = padded_img[:, :, y : y + self.config.tile_size, x : x + self.config.tile_size]
                tiles.append(tile)
                coords.append((y, x))

        # 3. Process Batch
        processed_tiles: List[torch.Tensor] = []
        for i in range(0, len(tiles), self.config.batch_size):
            batch = torch.cat(tiles[i : i + self.config.batch_size], dim=0)

            # Run the heavy callback (e.g., Neural Network)
            with torch.no_grad():
                processed_batch = processor_func(batch)

            # Split back into list
            processed_tiles.extend(processed_batch.chunk(processed_batch.shape[0], dim=0))

        # 4. Merge Tiles (Weighted Blending)
        # Create output buffer
        out_h, out_w = padded_img.shape[2], padded_img.shape[3]
        out_c = processed_tiles[0].shape[1]

        output = torch.zeros((b, out_c, out_h, out_w), device=device)
        weights = torch.zeros((b, 1, out_h, out_w), device=device)

        # Create tile weight map (Gaussian falloff) to blend seams
        tile_weight = self._create_tile_weight(self.config.tile_size, self.config.tile_overlap, device)

        for tile, (y, x) in zip(processed_tiles, coords):
            output[:, :, y : y + self.config.tile_size, x : x + self.config.tile_size] += tile * tile_weight
            weights[:, :, y : y + self.config.tile_size, x : x + self.config.tile_size] += tile_weight

        # Normalize by weights to average overlaps
        output /= weights + 1e-8

        # 5. Crop to original size
        output_tensor = output[:, :, :h, :w]

        # Return in original format
        if is_numpy:
            output_array = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            return (output_array * 255).clip(0, 255).astype(np.uint8)

        return output_tensor

    def _create_tile_weight(self, size: int, overlap: int, device: torch.device) -> torch.Tensor:
        """Create a 2D weight map for blending."""
        min_weight = 1e-3

        if overlap <= 0:
            return torch.ones((1, 1, size, size), device=device)

        if self.config.blend_mode == "linear":
            coords = torch.arange(size, device=device, dtype=torch.float32)
            edge_distance = torch.minimum(coords, (size - 1) - coords)
            ramp = torch.ones(size, device=device, dtype=torch.float32)
            edge_mask = edge_distance < overlap
            if edge_mask.any():
                ramp[edge_mask] = min_weight + (1.0 - min_weight) * (edge_distance[edge_mask] / float(overlap))
            linear_2d = ramp.unsqueeze(1) * ramp.unsqueeze(0)
            return linear_2d.view(1, 1, size, size)

        # Gaussian window
        # Create 1D gaussian
        sigma = size / 4  # Adjust falloff
        x = torch.arange(size, device=device).float()
        gaussian_1d = torch.exp(-((x - size / 2) ** 2) / (2 * sigma**2))

        # Outer product to make 2D
        gaussian_2d = gaussian_1d.unsqueeze(1) * gaussian_1d.unsqueeze(0)

        # Normalize to 0-1 range
        gaussian_2d -= gaussian_2d.min()
        gaussian_2d /= gaussian_2d.max()
        gaussian_2d = gaussian_2d.clamp_min(min_weight)

        return gaussian_2d.view(1, 1, size, size)
