"""Edge-Aware Depth Map Refinement Module.

Implements research-backed post-processing techniques to improve edge fidelity
in DA3 depth maps without sacrificing overall depth accuracy.

Research References:
- Bilateral Filter: Tomasi & Manduchi, 1998 (edge-preserving smoothing)
- Guided Filter: He et al., 2013 (fast edge-preserving filter)
- Joint Bilateral Upsampling: Kopf et al., 2007 (RGB-guided depth refinement)

Performance Targets:
- Edge F1: 0.22 → 0.30+ (primary metric)
- Processing overhead: <100ms per image (acceptable for production)
- No regressions: Chamfer distance, gradient metrics must not degrade
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from lux_depth_v3.config import RefinementConfig

logger = logging.getLogger(__name__)


class DepthRefiner:
    """Edge-aware post-processing for depth maps.

    Applies a configurable pipeline of refinement stages to improve edge
    fidelity while preserving depth accuracy.
    """

    def __init__(self, config: RefinementConfig):
        """Initialize depth refiner.

        Args:
            config: Refinement configuration
        """
        self.config = config

        # Check for OpenCV availability
        try:
            import cv2

            self.cv2 = cv2
            self.has_cv2 = True

            # Check for ximgproc (guided filter)
            try:
                # Try to access guidedFilter function
                if hasattr(cv2, "ximgproc"):
                    import cv2.ximgproc as ximgproc

                    # Verify guidedFilter is available
                    if hasattr(ximgproc, "guidedFilter"):
                        self.ximgproc = ximgproc
                        self.has_ximgproc = True
                        logger.debug("OpenCV ximgproc.guidedFilter available")
                    else:
                        self.has_ximgproc = False
                        logger.debug("OpenCV ximgproc found but guidedFilter not available, using bilateral fallback")
                else:
                    self.has_ximgproc = False
                    logger.debug("OpenCV ximgproc not available, using bilateral fallback")
            except (ImportError, AttributeError) as e:
                self.has_ximgproc = False
                logger.debug(f"OpenCV ximgproc not available ({e}), using bilateral fallback")

        except ImportError:
            self.has_cv2 = False
            logger.warning("OpenCV not available, refinement disabled")

    def refine(
        self,
        depth: np.ndarray,
        rgb: np.ndarray,
        stages: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Apply multi-stage refinement pipeline.

        Args:
            depth: Raw depth map from DA3 (H, W), float32, normalized [0, 1]
            rgb: Original RGB image (H, W, 3), uint8 [0, 255]
            stages: Ordered list of refinement stages to apply
                   Options: 'guided', 'bilateral', 'edge', 'gradient'
                   Default: Use config.stages

        Returns:
            Refined depth map (H, W), float32, normalized [0, 1]
        """
        if not self.has_cv2:
            logger.warning("OpenCV not available, returning original depth")
            return depth

        if not self.config.enable_refinement:
            logger.debug("Refinement disabled in config")
            return depth

        if stages is None:
            stages = self.config.stages

        result = depth.copy()
        stats = {"stages_applied": []}

        for stage in stages:
            if stage == "guided" and self.config.enable_guided:
                result = self._guided_filter_depth(result, rgb)
                stats["stages_applied"].append("guided")

            elif stage == "bilateral" and self.config.enable_bilateral:
                result = self._bilateral_depth_filter(result)
                stats["stages_applied"].append("bilateral")

            elif stage == "edge" and self.config.enable_edge:
                result = self._enhance_edges_with_guidance(result, rgb)
                stats["stages_applied"].append("edge")

            elif stage == "gradient" and self.config.enable_gradient:
                result = self._gradient_smoothness(result, rgb)
                stats["stages_applied"].append("gradient")

        logger.info(f"Refinement complete: {len(stats['stages_applied'])} stages applied")

        return result

    def _bilateral_depth_filter(self, depth_map: np.ndarray) -> np.ndarray:
        """Reduce noise without blurring edges using bilateral filtering.

        Bilateral filter smooths the image while preserving sharp edges by
        considering both spatial proximity and value similarity.

        Args:
            depth_map: Depth map (H, W), float32 [0, 1]

        Returns:
            Filtered depth map (H, W), float32 [0, 1]
        """
        # Normalize to uint8 for OpenCV
        depth_norm = self.cv2.normalize(depth_map, None, 0, 255, self.cv2.NORM_MINMAX).astype(np.uint8)

        # Apply bilateral filter
        filtered = self.cv2.bilateralFilter(
            depth_norm,
            d=self.config.bilateral_d,
            sigmaColor=self.config.bilateral_sigma_color,
            sigmaSpace=self.config.bilateral_sigma_space,
        )

        # Denormalize back to [0, 1]
        filtered_float = self.cv2.normalize(
            filtered.astype(np.float32),
            None,
            depth_map.min(),
            depth_map.max(),
            self.cv2.NORM_MINMAX,
        )

        logger.debug(
            f"Bilateral filter: d={self.config.bilateral_d}, "
            f"σ_color={self.config.bilateral_sigma_color}, "
            f"σ_space={self.config.bilateral_sigma_space}"
        )

        return filtered_float

    def _guided_filter_depth(
        self,
        depth_map: np.ndarray,
        rgb_image: np.ndarray,
    ) -> np.ndarray:
        """Use RGB edges to guide depth smoothing.

        Guided filter is faster than bilateral filtering and avoids gradient
        reversal artifacts. Uses RGB image as guidance to preserve edges
        that appear in the color image.

        Args:
            depth_map: Depth map (H, W), float32 [0, 1]
            rgb_image: RGB image (H, W, 3), uint8 [0, 255]

        Returns:
            Filtered depth map (H, W), float32 [0, 1]
        """
        # Convert RGB to float32 [0, 1] for guidance
        rgb_guide = rgb_image.astype(np.float32) / 255.0
        p = depth_map.astype(np.float32)

        if self.has_ximgproc:
            # Use true guided filter
            filtered = self.ximgproc.guidedFilter(rgb_guide, p, self.config.guided_radius, self.config.guided_eps)
            logger.debug(f"Guided filter: radius={self.config.guided_radius}, eps={self.config.guided_eps}")
        else:
            # Fallback to bilateral filter with similar parameters
            depth_uint8 = (depth_map * 255).astype(np.uint8)
            filtered_uint8 = self.cv2.bilateralFilter(
                depth_uint8,
                d=self.config.guided_radius * 2 + 1,
                sigmaColor=50,
                sigmaSpace=self.config.guided_radius,
            )
            filtered = filtered_uint8.astype(np.float32) / 255.0
            logger.debug(f"Bilateral filter fallback (no ximgproc): radius={self.config.guided_radius}")

        return filtered

    def _enhance_edges_with_guidance(
        self,
        depth_map: np.ndarray,
        rgb_image: np.ndarray,
    ) -> np.ndarray:
        """Compute edge map from RGB, preserve depth at edges, smooth elsewhere.

        This technique identifies edges in the RGB image using Canny edge
        detection, then preserves the original depth values at edge locations
        while applying Gaussian smoothing to non-edge regions.

        Args:
            depth_map: Depth map (H, W), float32 [0, 1]
            rgb_image: RGB image (H, W, 3), uint8 [0, 255]

        Returns:
            Edge-enhanced depth map (H, W), float32 [0, 1]
        """
        # Convert to grayscale for edge detection
        gray = self.cv2.cvtColor(rgb_image, self.cv2.COLOR_RGB2GRAY)

        # Detect edges using Canny
        edges = (
            self.cv2.Canny(
                gray,
                self.config.edge_canny_low,
                self.config.edge_canny_high,
            ).astype(np.float32)
            / 255.0
        )

        # Smooth depth map (non-edge regions)
        kernel_size = int(self.config.edge_blend_sigma * 2) * 2 + 1
        smoothed = self.cv2.GaussianBlur(depth_map, (kernel_size, kernel_size), self.config.edge_blend_sigma)

        # Blend: preserve depth at edges, use smoothed elsewhere
        result = depth_map * edges + smoothed * (1.0 - edges)

        edge_pixels = (edges > 0).sum()
        total_pixels = edges.size
        logger.debug(f"Edge enhancement: {edge_pixels}/{total_pixels} edge pixels ({100 * edge_pixels / total_pixels:.1f}%)")

        return result

    def _gradient_smoothness(
        self,
        depth_map: np.ndarray,
        rgb_image: np.ndarray,
    ) -> np.ndarray:
        """Enforce smoothness away from edges, allow sharp transitions at gradients.

        This technique computes the gradient magnitude in the RGB image and
        applies selective smoothing only in low-gradient regions, allowing
        depth discontinuities to remain sharp at high-gradient locations.

        Args:
            depth_map: Depth map (H, W), float32 [0, 1]
            rgb_image: RGB image (H, W, 3), uint8 [0, 255]

        Returns:
            Gradient-smoothed depth map (H, W), float32 [0, 1]
        """
        # Compute gradients in RGB image
        rgb_float = rgb_image.astype(np.float32)

        ix = self.cv2.Sobel(rgb_float, self.cv2.CV_32F, 1, 0, ksize=3)
        iy = self.cv2.Sobel(rgb_float, self.cv2.CV_32F, 0, 1, ksize=3)

        # Gradient magnitude (average across RGB channels)
        grad_mag = np.sqrt(ix**2 + iy**2).mean(axis=2)

        # Normalize gradient magnitude
        grad_mag_norm = grad_mag / (grad_mag.max() + 1e-8)

        # Apply smoothing only in low-gradient regions
        smooth = depth_map.copy()
        h, w = depth_map.shape

        # Vectorized version for efficiency
        low_gradient_mask = grad_mag_norm < self.config.gradient_threshold

        # Simple 4-neighbor averaging for low-gradient regions
        # Pad depth map to handle boundaries
        padded = np.pad(depth_map, ((1, 1), (1, 1)), mode="edge")

        # Compute average of 4 neighbors
        neighbors_avg = (
            (
                padded[0:-2, 1:-1]  # top
                + padded[2:, 1:-1]  # bottom
                + padded[1:-1, 0:-2]  # left
                + padded[1:-1, 2:]  # right
            )
            / 4.0
        )

        # Apply smoothing only where gradient is low
        smooth = np.where(low_gradient_mask, neighbors_avg, depth_map)

        smooth_pixels = low_gradient_mask.sum()
        total_pixels = low_gradient_mask.size
        logger.debug(
            f"Gradient smoothing: {smooth_pixels}/{total_pixels} pixels smoothed "
            f"({100 * smooth_pixels / total_pixels:.1f}%), "
            f"threshold={self.config.gradient_threshold}"
        )

        return smooth

    def get_stats(self) -> Dict[str, any]:
        """Get refinement statistics.

        Returns:
            Dictionary with configuration and capabilities
        """
        return {
            "enabled": self.config.enable_refinement,
            "stages": self.config.stages,
            "has_opencv": self.has_cv2,
            "has_ximgproc": self.has_ximgproc,
            "bilateral_enabled": self.config.enable_bilateral,
            "guided_enabled": self.config.enable_guided,
            "edge_enabled": self.config.enable_edge,
            "gradient_enabled": self.config.enable_gradient,
        }


def create_refinement_preset(preset_name: str = "balanced") -> RefinementConfig:
    """Create a preset refinement configuration.

    Args:
        preset_name: Preset name
            - 'balanced': Guided + Bilateral (recommended)
            - 'aggressive': All stages enabled (maximum edge preservation)
            - 'conservative': Bilateral only (minimal processing)
            - 'edge_focused': Edge + Guided (prioritize edge fidelity)

    Returns:
        RefinementConfig with preset values
    """
    presets = {
        "balanced": RefinementConfig(
            enable_refinement=True,
            stages=["guided", "bilateral"],
            enable_guided=True,
            enable_bilateral=True,
            enable_edge=False,
            enable_gradient=False,
            guided_radius=8,
            guided_eps=0.01,
            bilateral_d=9,
            bilateral_sigma_color=75,
            bilateral_sigma_space=75,
        ),
        "aggressive": RefinementConfig(
            enable_refinement=True,
            stages=["guided", "bilateral", "edge", "gradient"],
            enable_guided=True,
            enable_bilateral=True,
            enable_edge=True,
            enable_gradient=True,
            guided_radius=12,
            guided_eps=0.005,
            bilateral_d=11,
            bilateral_sigma_color=50,
            bilateral_sigma_space=50,
            edge_canny_low=30,
            edge_canny_high=120,
            gradient_threshold=0.05,
        ),
        "conservative": RefinementConfig(
            enable_refinement=True,
            stages=["bilateral"],
            enable_guided=False,
            enable_bilateral=True,
            enable_edge=False,
            enable_gradient=False,
            bilateral_d=7,
            bilateral_sigma_color=100,
            bilateral_sigma_space=100,
        ),
        "edge_focused": RefinementConfig(
            enable_refinement=True,
            stages=["edge", "guided"],
            enable_guided=True,
            enable_bilateral=False,
            enable_edge=True,
            enable_gradient=False,
            guided_radius=10,
            guided_eps=0.008,
            edge_canny_low=40,
            edge_canny_high=140,
            edge_blend_sigma=5,
        ),
    }

    if preset_name not in presets:
        logger.warning(f"Unknown preset '{preset_name}', using 'balanced'. Available: {list(presets.keys())}")
        preset_name = "balanced"

    return presets[preset_name]
