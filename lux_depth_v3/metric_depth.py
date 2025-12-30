"""Metric depth conversion utilities for DA3 models."""

import numpy as np
from typing import Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetricDepthResult:
    """Result of metric depth conversion."""

    depth_meters: np.ndarray  # Depth in meters (H, W) or (N, H, W)
    focal_length_px: float  # Focal length used in pixels
    scale_factor: float  # Scale factor applied
    source_model: str  # Model variant used
    already_metric: bool  # True if no conversion needed

    def save(self, output_path: Path) -> None:
        """Save metric depth to NPZ file."""
        np.savez_compressed(
            output_path,
            depth_meters=self.depth_meters,
            focal_length_px=self.focal_length_px,
            scale_factor=self.scale_factor,
            source_model=self.source_model,
            already_metric=self.already_metric,
        )
        logger.info(f"Saved metric depth to {output_path}")

    @classmethod
    def load(cls, input_path: Path) -> "MetricDepthResult":
        """Load metric depth from NPZ file."""
        data = np.load(input_path)
        return cls(
            depth_meters=data["depth_meters"],
            focal_length_px=float(data["focal_length_px"]),
            scale_factor=float(data["scale_factor"]),
            source_model=str(data["source_model"]),
            already_metric=bool(data["already_metric"]),
        )


class MetricDepthConverter:
    """
    Convert DA3 depth outputs to metric depth in meters.

    Handles two model types:
    1. DA3METRIC-LARGE: Requires focal length conversion
    2. DA3NESTED-GIANT-LARGE: Already in meters (no conversion)

    Formula for DA3METRIC-LARGE:
        metric_depth = focal * net_output / 300.0

    where:
        - focal = average of fx and fy from intrinsics (in pixels)
        - net_output = raw model output (disparity-like)
        - 300.0 = model-specific scale constant
    """

    # Model-specific scale constants
    SCALE_CONSTANTS = {
        "DA3METRIC-LARGE": 300.0,
        "DA3NESTED-GIANT-LARGE": 1.0,  # Already metric
        "DA3NESTED-GIANT-LARGE-1.1": 1.0,  # Already metric
    }

    # Models that output metric depth directly
    METRIC_MODELS = {"DA3NESTED-GIANT-LARGE", "DA3NESTED-GIANT-LARGE-1.1"}

    def __init__(self, model_name: str = "DA3METRIC-LARGE"):
        """
        Initialize converter.

        Args:
            model_name: Name of the DA3 model variant
        """
        self.model_name = model_name
        self.scale_constant = self.SCALE_CONSTANTS.get(
            model_name,
            300.0,  # Default to DA3METRIC constant
        )
        self.is_metric_model = model_name in self.METRIC_MODELS

    def convert(
        self,
        depth: np.ndarray,
        intrinsics: Optional[np.ndarray] = None,
        focal_length_px: Optional[float] = None,
        image_width: Optional[int] = None,
        fov_degrees: Optional[float] = None,
    ) -> MetricDepthResult:
        """
        Convert depth to metric depth in meters.

        Args:
            depth: Raw depth output from DA3 (H, W) or (N, H, W)
            intrinsics: Camera intrinsics (3, 3) or (N, 3, 3)
            focal_length_px: Focal length in pixels (alternative to intrinsics)
            image_width: Image width in pixels (for FOV-based estimation)
            fov_degrees: Horizontal field of view in degrees (fallback estimation)

        Returns:
            MetricDepthResult with depth in meters and metadata

        Raises:
            ValueError: If insufficient information for conversion

        Examples:
            >>> # Using intrinsics (recommended)
            >>> converter = MetricDepthConverter("DA3METRIC-LARGE")
            >>> result = converter.convert(depth, intrinsics=K)

            >>> # Using focal length directly
            >>> result = converter.convert(depth, focal_length_px=500.0)

            >>> # Using FOV estimation (less accurate)
            >>> result = converter.convert(depth, image_width=1920, fov_degrees=60.0)
        """
        # Check if model already outputs metric depth
        if self.is_metric_model:
            return MetricDepthResult(
                depth_meters=depth, focal_length_px=0.0, scale_factor=1.0, source_model=self.model_name, already_metric=True
            )

        # Determine focal length
        focal = self._determine_focal_length(
            intrinsics=intrinsics, focal_length_px=focal_length_px, image_width=image_width, fov_degrees=fov_degrees
        )

        if focal is None:
            raise ValueError(
                "Cannot convert to metric depth: No focal length information provided. "
                "Please provide one of: intrinsics, focal_length_px, or (image_width + fov_degrees)"
            )

        # Apply conversion formula
        scale_factor = focal / self.scale_constant
        metric_depth = depth * scale_factor

        logger.info(f"Converted {self.model_name} depth to meters (focal={focal:.2f}px, scale={scale_factor:.4f})")

        return MetricDepthResult(
            depth_meters=metric_depth,
            focal_length_px=focal,
            scale_factor=scale_factor,
            source_model=self.model_name,
            already_metric=False,
        )

    def _determine_focal_length(
        self,
        intrinsics: Optional[np.ndarray],
        focal_length_px: Optional[float],
        image_width: Optional[int],
        fov_degrees: Optional[float],
    ) -> Optional[float]:
        """
        Determine focal length from various sources.

        Priority order:
        1. Explicit focal_length_px
        2. Extract from intrinsics matrix
        3. Estimate from image_width + fov_degrees

        Returns:
            Focal length in pixels, or None if cannot be determined
        """
        # Priority 1: Explicit focal length
        if focal_length_px is not None:
            return float(focal_length_px)

        # Priority 2: Extract from intrinsics
        if intrinsics is not None:
            focal = self._extract_focal_from_intrinsics(intrinsics)
            if focal is not None:
                return focal

        # Priority 3: Estimate from FOV
        if image_width is not None and fov_degrees is not None:
            return self._estimate_focal_from_fov(image_width, fov_degrees)

        return None

    def _extract_focal_from_intrinsics(self, intrinsics: np.ndarray) -> Optional[float]:
        """
        Extract focal length from camera intrinsics matrix.

        Intrinsics format:
            [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]

        Returns average of fx and fy.
        """
        if intrinsics.ndim == 2 and intrinsics.shape == (3, 3):
            # Single intrinsics matrix
            fx = intrinsics[0, 0]
            fy = intrinsics[1, 1]
            return float((fx + fy) / 2.0)

        elif intrinsics.ndim == 3 and intrinsics.shape[1:] == (3, 3):
            # Multiple intrinsics (N, 3, 3) - use first
            fx = intrinsics[0, 0, 0]
            fy = intrinsics[0, 1, 1]
            logger.warning(f"Multiple intrinsics provided, using first camera (fx={fx:.2f}, fy={fy:.2f})")
            return float((fx + fy) / 2.0)

        else:
            logger.error(f"Invalid intrinsics shape: {intrinsics.shape}")
            return None

    def _estimate_focal_from_fov(self, image_width: int, fov_degrees: float) -> float:
        """
        Estimate focal length from horizontal FOV.

        Formula:
            focal = (image_width / 2) / tan(fov / 2)

        This is an approximation and less accurate than using
        actual camera intrinsics.
        """
        fov_radians = np.deg2rad(fov_degrees)
        focal = (image_width / 2.0) / np.tan(fov_radians / 2.0)

        logger.warning(
            f"Estimating focal length from FOV: "
            f"width={image_width}px, fov={fov_degrees}° → focal={focal:.2f}px "
            f"(This is an approximation - use intrinsics for accuracy)"
        )

        return focal


def convert_to_metric_depth(
    depth: np.ndarray,
    model_name: str = "DA3METRIC-LARGE",
    intrinsics: Optional[np.ndarray] = None,
    focal_length_px: Optional[float] = None,
    image_width: Optional[int] = None,
    fov_degrees: Optional[float] = None,
) -> MetricDepthResult:
    """
    Convenience function to convert depth to metric depth.

    Args:
        depth: Raw depth output from DA3
        model_name: DA3 model variant name
        intrinsics: Camera intrinsics (3, 3)
        focal_length_px: Focal length in pixels
        image_width: Image width (for FOV estimation)
        fov_degrees: Horizontal FOV (for estimation)

    Returns:
        MetricDepthResult with depth in meters

    Examples:
        >>> # Using intrinsics (recommended)
        >>> result = convert_to_metric_depth(depth, intrinsics=K)
        >>> print(result.depth_meters.shape)

        >>> # Using focal length
        >>> result = convert_to_metric_depth(depth, focal_length_px=500.0)

        >>> # Nested model (already metric)
        >>> result = convert_to_metric_depth(
        ...     depth,
        ...     model_name="DA3NESTED-GIANT-LARGE-1.1"
        ... )
    """
    converter = MetricDepthConverter(model_name=model_name)
    return converter.convert(
        depth=depth, intrinsics=intrinsics, focal_length_px=focal_length_px, image_width=image_width, fov_degrees=fov_degrees
    )


# Utility functions for common use cases


def depth_to_meters(depth: np.ndarray, focal_length_px: float, model_name: str = "DA3METRIC-LARGE") -> np.ndarray:
    """
    Quick conversion: depth to meters using focal length.

    Args:
        depth: Raw depth from DA3METRIC-LARGE
        focal_length_px: Focal length in pixels
        model_name: Model variant (default: DA3METRIC-LARGE)

    Returns:
        Depth in meters
    """
    result = convert_to_metric_depth(depth, model_name=model_name, focal_length_px=focal_length_px)
    return result.depth_meters


def get_depth_statistics(depth_meters: np.ndarray, mask: Optional[np.ndarray] = None) -> dict:
    """
    Compute depth statistics in meters.

    Args:
        depth_meters: Metric depth in meters
        mask: Optional mask (True = valid, False = ignore)

    Returns:
        Dictionary with min, max, mean, median, std
    """
    valid_depth = depth_meters if mask is None else depth_meters[mask]

    return {
        "min_m": float(np.min(valid_depth)),
        "max_m": float(np.max(valid_depth)),
        "mean_m": float(np.mean(valid_depth)),
        "median_m": float(np.median(valid_depth)),
        "std_m": float(np.std(valid_depth)),
        "range_m": float(np.ptp(valid_depth)),
    }
