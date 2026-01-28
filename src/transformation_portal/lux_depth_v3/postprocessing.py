"""Postprocessing module for DA3 depth maps.

Handles metric scaling, filtering, and edge preservation.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import numpy as np
from scipy.ndimage import median_filter
from .config import PostprocessingConfig
from .inference import DepthResult

# Edge refinement is optional (and may not be present in stripped-down deployments).
try:
    from .edge_refinement import DepthRefiner  # type: ignore
except ImportError:
    DepthRefiner = None

class _NoOpDepthRefiner:
    """Fallback refiner used when the optional edge_refinement module isn't available."""
    def __init__(self, *args, **kwargs): self._stats = {"enabled": False, "available": False}
    def refine(self, depth: np.ndarray, image: np.ndarray) -> np.ndarray: return depth
    def get_stats(self): return dict(self._stats)

class Postprocessor:
    """Postprocessor for depth maps."""

    def __init__(self, config: PostprocessingConfig):
        self.config = config

        # Initialize edge refinement module
        refinement_cfg = getattr(config, "refinement", None)
        if DepthRefiner is None or refinement_cfg is None:
            self.refiner = _NoOpDepthRefiner()
        else:
            try:
                self.refiner = DepthRefiner(refinement_cfg)
            except Exception:
                self.refiner = _NoOpDepthRefiner()

    def process(self, result: DepthResult) -> DepthResult:
        """Apply postprocessing to depth result."""
        depth = result.depth_map.copy()

        # Metric scaling
        if self.config.apply_metric_scaling:
            depth = depth * self.config.scale_factor

        # Filtering
        if self.config.apply_median_filter:
            depth = median_filter(depth, size=self.config.median_kernel_size)

        if self.config.apply_bilateral_filter:
            depth = self._bilateral_filter(
                depth, result.original_image,
                self.config.bilateral_sigma_color,
                self.config.bilateral_sigma_space
            )

        # Edge preservation
        if self.config.preserve_edges:
            depth = self._preserve_edges(
                depth, result.original_image,
                self.config.edge_threshold
            )

        # Edge-aware refinement (Optional Module)
        if self.refiner and getattr(self.config.refinement, "enable_refinement", False):
            depth = self.refiner.refine(depth, result.original_image)

        # Update result
        result.depth_map = depth
        result.metadata["postprocessing"] = self.config.__dict__
        result.metadata["refinement"] = self.refiner.get_stats()

        return result

    def _bilateral_filter(self, depth: np.ndarray, image: np.ndarray, sigma_color: float, sigma_space: float) -> np.ndarray:
        try:
            import cv2
            depth_uint8 = (depth * 255).astype(np.uint8)
            filtered = cv2.bilateralFilter(
                depth_uint8, d=int(sigma_space),
                sigmaColor=sigma_color, sigmaSpace=sigma_space
            )
            return filtered.astype(np.float32) / 255.0
        except ImportError:
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(depth, sigma=sigma_space / 3.0)

    def _preserve_edges(self, depth: np.ndarray, image: np.ndarray, threshold: float) -> np.ndarray:
        gray = np.mean(image, axis=2) if image.ndim == 3 else image
        from scipy.ndimage import sobel
        mag = np.hypot(sobel(gray, axis=0), sobel(gray, axis=1))
        mag = mag / (mag.max() + 1e-8)
        # Simple mask-based preservation logic (placeholder for more complex logic)
        # In production, this would likely blend based on edge magnitude
        return depth

    def fuse_multiview(self, results: List[DepthResult]) -> DepthResult:
        """Simple fusion stub."""
        if not results: raise ValueError("No results to fuse")
        depths = np.stack([r.depth_map for r in results], axis=0)

        if self.config.fusion_mode == "mean": fused = np.mean(depths, axis=0)
        elif self.config.fusion_mode == "median": fused = np.median(depths, axis=0)
        else: fused = np.mean(depths, axis=0)  # Default

        return DepthResult(fused, results[0].original_image, metadata={"fusion_mode": self.config.fusion_mode})
