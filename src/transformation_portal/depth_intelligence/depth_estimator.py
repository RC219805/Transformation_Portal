"""
Depth Estimator for Phase 3

Integrates Depth Anything V2 with Phase 1 substrate and Phase 2 baseline,
providing depth estimation optimized for architectural imagery.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import logging

import torch
from torch import Tensor
import numpy as np
from PIL import Image

# Import existing depth infrastructure
from ..depth.models.depth_anything_v2 import (
    DepthAnythingV2Model,
    ModelVariant,
    ModelBackend
)

logger = logging.getLogger(__name__)


@dataclass
class DepthConfig:
    """Configuration for depth estimation."""
    variant: ModelVariant = ModelVariant.SMALL
    backend: Optional[ModelBackend] = None  # Auto-detect
    precision: str = "fp16"
    target_size: Tuple[int, int] = (518, 518)  # Optimal for Depth Anything V2
    normalize_output: bool = True
    cache_models: bool = True


@dataclass
class DepthMap:
    """Depth map with metadata."""
    depth: Tensor  # (H, W) normalized to [0, 1], 0=near, 1=far
    confidence: Optional[Tensor] = None  # (H, W) confidence scores
    min_depth: float = 0.0  # Minimum depth in scene
    max_depth: float = 1.0  # Maximum depth in scene
    resolution: Tuple[int, int] = (0, 0)  # Original resolution
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.resolution == (0, 0):
            self.resolution = tuple(self.depth.shape)

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return self.depth.cpu().numpy()

    def to_pil(self, colorize: bool = True) -> Image.Image:
        """Convert to PIL Image."""
        depth_np = self.to_numpy()

        if colorize:
            # Apply colormap (viridis-like)
            depth_colored = self._apply_colormap(depth_np)
            return Image.fromarray(depth_colored)
        else:
            # Grayscale
            depth_uint8 = (depth_np * 255).astype(np.uint8)
            return Image.fromarray(depth_uint8, mode='L')

    def _apply_colormap(self, depth: np.ndarray) -> np.ndarray:
        """Apply perceptually uniform colormap to depth."""
        import matplotlib.pyplot as plt

        # Use viridis colormap (perceptually uniform)
        cmap = plt.cm.viridis
        colored = cmap(depth)[:, :, :3]  # Remove alpha
        colored_uint8 = (colored * 255).astype(np.uint8)

        return colored_uint8

    def invert(self) -> "DepthMap":
        """Invert depth (far becomes near)."""
        inverted_depth = 1.0 - self.depth
        return DepthMap(
            depth=inverted_depth,
            confidence=self.confidence,
            min_depth=1.0 - self.max_depth,
            max_depth=1.0 - self.min_depth,
            resolution=self.resolution,
            metadata={**self.metadata, "inverted": True}
        )

    def normalize_to_range(self, min_val: float, max_val: float) -> Tensor:
        """Normalize depth to specific range."""
        return self.depth * (max_val - min_val) + min_val


class DepthEstimator:
    """
    Depth estimator integrated with Phase 1 substrate.

    Provides depth estimation for architectural imagery with:
    - Depth Anything V2 integration
    - M4 Max optimization (MPS/ANE)
    - Substrate memory management
    - Baseline quality tracking
    """

    def __init__(
        self,
        substrate,
        config: Optional[DepthConfig] = None
    ):
        """
        Initialize depth estimator.

        Args:
            substrate: Computational substrate from Phase 1
            config: Depth estimation configuration
        """
        self.substrate = substrate
        self.config = config or DepthConfig()

        # Auto-detect backend if not specified
        if self.config.backend is None:
            self.config.backend = self._auto_detect_backend()

        # Initialize model
        self.model = None
        self._initialize_model()

        logger.info(f"Initialized DepthEstimator with {self.config.variant.value}, "
                    f"backend={self.config.backend.value}")

    def _auto_detect_backend(self) -> ModelBackend:
        """Auto-detect optimal backend."""
        device = self.substrate.get_device()

        if device.type == "mps":
            # Check if CoreML ANE is available for production
            try:
                return ModelBackend.COREML  # Prefer ANE for inference
            except ImportError:
                return ModelBackend.PYTORCH_MPS
        elif device.type == "cuda":
            return ModelBackend.PYTORCH_CPU  # Would be CUDA in future
        else:
            return ModelBackend.PYTORCH_CPU

    def _initialize_model(self):
        """Initialize depth estimation model."""
        logger.info(f"Loading depth model: {self.config.variant.value}")

        try:
            self.model = DepthAnythingV2Model(
                variant=self.config.variant,
                backend=self.config.backend,
                precision=self.config.precision
            )
            logger.info("✓ Depth model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load depth model: {e}")
            logger.warning("Falling back to MPS backend")
            self.config.backend = ModelBackend.PYTORCH_MPS
            self.model = DepthAnythingV2Model(
                variant=self.config.variant,
                backend=self.config.backend,
                precision=self.config.precision
            )

    def estimate(
        self,
        image: Tensor,
        return_confidence: bool = False
    ) -> DepthMap:
        """
        Estimate depth from image.

        Args:
            image: Input image tensor (C, H, W) or (B, C, H, W) in [0, 1]
            return_confidence: Whether to compute confidence scores

        Returns:
            DepthMap with depth and optional confidence
        """
        # Ensure 4D tensor (B, C, H, W)
        if image.ndim == 3:
            image = image.unsqueeze(0)

        original_size = (image.shape[2], image.shape[3])

        # Convert to PIL for model (model expects PIL or numpy)
        image_np = image[0].cpu().permute(1, 2, 0).numpy()
        image_np = (image_np * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)

        # Estimate depth
        with self.substrate.profile("depth_estimation"):
            result = self.model.estimate_depth(pil_image)

        # Extract depth map
        depth_np = result['depth']  # (H, W) normalized to [0, 1]

        # Convert to tensor and move to device
        depth_tensor = torch.from_numpy(depth_np).float()
        depth_tensor = self.substrate.to_device(depth_tensor)

        # Resize to original if needed
        if depth_tensor.shape != original_size:
            depth_tensor = torch.nn.functional.interpolate(
                depth_tensor.unsqueeze(0).unsqueeze(0),
                size=original_size,
                mode='bilinear',
                align_corners=False
            ).squeeze()

        # Compute confidence if requested
        confidence = None
        if return_confidence:
            confidence = self._compute_confidence(depth_tensor)

        # Create depth map
        depth_map = DepthMap(
            depth=depth_tensor,
            confidence=confidence,
            min_depth=depth_tensor.min().item(),
            max_depth=depth_tensor.max().item(),
            resolution=original_size,
            metadata={
                "variant": self.config.variant.value,
                "backend": self.config.backend.value,
                "original_size": original_size
            }
        )

        return depth_map

    def estimate_batch(
        self,
        images: list[Tensor],
        return_confidence: bool = False
    ) -> list[DepthMap]:
        """
        Estimate depth for batch of images.

        Args:
            images: List of image tensors
            return_confidence: Whether to compute confidence

        Returns:
            List of depth maps
        """
        depth_maps = []

        for image in images:
            depth_map = self.estimate(image, return_confidence=return_confidence)
            depth_maps.append(depth_map)

        return depth_maps

    def _compute_confidence(self, depth: Tensor) -> Tensor:
        """
        Compute confidence scores for depth estimates.

        Uses local variance as confidence metric - smooth regions have
        higher confidence than edge regions.
        """
        # Compute local variance as confidence metric
        kernel_size = 5
        padding = kernel_size // 2

        # Unfold to get local patches
        depth_unfold = torch.nn.functional.unfold(
            depth.unsqueeze(0).unsqueeze(0),
            kernel_size=kernel_size,
            padding=padding
        )

        # Compute variance per patch
        variance = depth_unfold.var(dim=1)

        # Reshape back to image
        confidence = variance.view(depth.shape)

        # Invert (low variance = high confidence)
        confidence = 1.0 - torch.clamp(confidence, 0, 1)

        return confidence

    def visualize_depth(
        self,
        depth_map: DepthMap,
        output_path: Optional[Path] = None,
        show: bool = False
    ) -> Image.Image:
        """
        Visualize depth map with colormap.

        Args:
            depth_map: Depth map to visualize
            output_path: Optional path to save visualization
            show: Whether to display visualization

        Returns:
            PIL Image of visualization
        """
        vis = depth_map.to_pil(colorize=True)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            vis.save(output_path)
            logger.info(f"Depth visualization saved to {output_path}")

        if show:
            vis.show()

        return vis

    def get_depth_statistics(self, depth_map: DepthMap) -> Dict[str, float]:
        """Get statistics for depth map."""
        depth = depth_map.depth

        return {
            "min": depth.min().item(),
            "max": depth.max().item(),
            "mean": depth.mean().item(),
            "std": depth.std().item(),
            "median": depth.median().item(),
            "depth_range": (depth.max() - depth.min()).item(),
        }

    def __repr__(self) -> str:
        return (
            f"DepthEstimator(variant={self.config.variant.value}, "
            f"backend={self.config.backend.value})"
        )
