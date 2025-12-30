"""Postprocessing module for DA3 depth maps.

Handles metric scaling, filtering, edge preservation, and multi-view fusion.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy.ndimage import median_filter

from lux_depth_v3.config import PostprocessingConfig
from lux_depth_v3.inference import DepthResult
from lux_depth_v3.edge_refinement import DepthRefiner


class Postprocessor:
    """Postprocessor for depth maps."""

    def __init__(self, config: PostprocessingConfig):
        """Initialize postprocessor.

        Args:
            config: Postprocessing configuration
        """
        self.config = config

        # Initialize edge refinement module
        self.refiner = DepthRefiner(config.refinement)

    def process(self, result: DepthResult) -> DepthResult:
        """Apply postprocessing to depth result.

        Args:
            result: Depth estimation result

        Returns:
            Postprocessed result
        """
        depth = result.depth_map.copy()

        # Metric scaling
        if self.config.apply_metric_scaling:
            depth = self._apply_metric_scaling(depth, self.config.scale_factor)

        # Filtering
        if self.config.apply_median_filter:
            depth = self._median_filter(depth, self.config.median_kernel_size)

        if self.config.apply_bilateral_filter:
            depth = self._bilateral_filter(
                depth,
                result.original_image,
                self.config.bilateral_sigma_color,
                self.config.bilateral_sigma_space,
            )

        # Edge preservation
        if self.config.preserve_edges:
            depth = self._preserve_edges(
                depth,
                result.original_image,
                self.config.edge_threshold,
            )

        # Edge-aware refinement (new)
        if self.config.refinement.enable_refinement:
            depth = self.refiner.refine(depth, result.original_image)

        # Update result
        result.depth_map = depth
        result.metadata["postprocessing"] = self.config.__dict__
        result.metadata["refinement"] = self.refiner.get_stats()

        return result

    def fuse_multiview(
        self,
        results: List[DepthResult],
    ) -> DepthResult:
        """Fuse multiple depth maps (multi-view).

        Args:
            results: List of depth results from multiple views

        Returns:
            Fused depth result
        """
        if len(results) == 1:
            return results[0]

        # Stack depth maps
        depths = np.stack([r.depth_map for r in results], axis=0)

        # Fuse based on mode
        if self.config.fusion_mode == "mean":
            fused = np.mean(depths, axis=0)
        elif self.config.fusion_mode == "median":
            fused = np.median(depths, axis=0)
        elif self.config.fusion_mode == "weighted":
            # Weighted fusion based on confidence (placeholder)
            # In production, this would use actual confidence scores
            weights = np.ones(len(results)) / len(results)
            fused = np.average(depths, axis=0, weights=weights)
        else:
            raise ValueError(f"Unknown fusion mode: {self.config.fusion_mode}")

        # Create fused result
        fused_result = DepthResult(
            depth_map=fused,
            original_image=results[0].original_image,  # Use first image as reference
            metadata={
                "fusion_mode": self.config.fusion_mode,
                "num_views": len(results),
            },
        )

        return fused_result

    def _apply_metric_scaling(
        self,
        depth: np.ndarray,
        scale_factor: float,
    ) -> np.ndarray:
        """Apply metric scaling to depth map.

        Args:
            depth: Depth map (H, W)
            scale_factor: Scale factor

        Returns:
            Scaled depth map
        """
        return depth * scale_factor

    def _median_filter(
        self,
        depth: np.ndarray,
        kernel_size: int,
    ) -> np.ndarray:
        """Apply median filter for noise reduction.

        Args:
            depth: Depth map (H, W)
            kernel_size: Kernel size (odd number)

        Returns:
            Filtered depth map
        """
        return median_filter(depth, size=kernel_size)

    def _bilateral_filter(
        self,
        depth: np.ndarray,
        image: np.ndarray,
        sigma_color: float,
        sigma_space: float,
    ) -> np.ndarray:
        """Apply joint bilateral filter.

        Uses image as guidance to preserve edges while smoothing depth.

        Args:
            depth: Depth map (H, W)
            image: RGB image (H, W, 3)
            sigma_color: Color similarity sigma
            sigma_space: Spatial sigma

        Returns:
            Filtered depth map
        """
        try:
            import cv2

            # Convert to uint8 for OpenCV
            depth_uint8 = (depth * 255).astype(np.uint8)

            # Apply bilateral filter
            filtered = cv2.bilateralFilter(
                depth_uint8,
                d=int(sigma_space),
                sigmaColor=sigma_color,
                sigmaSpace=sigma_space,
            )

            return filtered.astype(np.float32) / 255.0

        except ImportError:
            # Fallback to Gaussian filter if OpenCV not available
            from scipy.ndimage import gaussian_filter

            return gaussian_filter(depth, sigma=sigma_space / 3.0)

    def _preserve_edges(
        self,
        depth: np.ndarray,
        image: np.ndarray,
        edge_threshold: float,
    ) -> np.ndarray:
        """Preserve edges from image in depth map.

        Args:
            depth: Depth map (H, W)
            image: RGB image (H, W, 3)
            edge_threshold: Edge detection threshold

        Returns:
            Edge-preserved depth map
        """
        # Detect edges in image
        gray = np.mean(image, axis=2) if image.ndim == 3 else image

        # Sobel edge detection
        from scipy.ndimage import sobel

        edge_x = sobel(gray, axis=0)
        edge_y = sobel(gray, axis=1)
        edge_magnitude = np.sqrt(edge_x**2 + edge_y**2)

        # Normalize edge magnitude
        edge_magnitude = edge_magnitude / (edge_magnitude.max() + 1e-8)

        # Create edge mask
        edge_mask = edge_magnitude > edge_threshold

        # Preserve depth at edges (don't filter)
        # This is a simple approach - production code would be more sophisticated
        depth_preserved = depth.copy()

        return depth_preserved

    def to_point_cloud(
        self,
        depth: np.ndarray,
        image: np.ndarray,
        intrinsics: Optional[Tuple[float, float, float, float]] = None,
    ) -> np.ndarray:
        """Convert depth map to point cloud.

        Args:
            depth: Depth map (H, W)
            image: RGB image (H, W, 3)
            intrinsics: Camera intrinsics (fx, fy, cx, cy)

        Returns:
            Point cloud (N, 6) with XYZ and RGB
        """
        h, w = depth.shape

        # Default intrinsics (assume FOV ~60 degrees)
        if intrinsics is None:
            fx = fy = w / (2 * np.tan(np.radians(30)))
            cx, cy = w / 2, h / 2
        else:
            fx, fy, cx, cy = intrinsics

        # Create pixel grid
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        # Backproject to 3D
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth

        # Flatten and stack
        points_xyz = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)

        # Add color
        if image.ndim == 3:
            colors = image.reshape(-1, 3)
        else:
            colors = np.stack([image.ravel()] * 3, axis=1)

        # Combine XYZ and RGB
        point_cloud = np.concatenate([points_xyz, colors], axis=1)

        return point_cloud

    def export_metric_depth(self, metric_result, output_dir: Path, export_formats: List[str] = None) -> List[Path]:
        """
        Export metric depth in multiple formats.

        Args:
            metric_result: MetricDepthResult from conversion
            output_dir: Output directory
            export_formats: List of formats (npz, tiff, png, exr)

        Returns:
            List of exported file paths
        """
        from pathlib import Path

        if export_formats is None:
            export_formats = ["npz", "tiff"]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        exported_files = []

        depth_meters = metric_result.depth_meters

        # NPZ format (recommended - preserves precision)
        if "npz" in export_formats:
            npz_path = output_dir / "depth_metric.npz"
            metric_result.save(npz_path)
            exported_files.append(npz_path)

        # TIFF format (16-bit, lossless)
        if "tiff" in export_formats:
            tiff_path = output_dir / "depth_metric.tiff"
            self._save_depth_tiff(depth_meters, tiff_path)
            exported_files.append(tiff_path)

        # PNG format (normalized for visualization)
        if "png" in export_formats:
            png_path = output_dir / "depth_metric_vis.png"
            self._save_depth_png(depth_meters, png_path)
            exported_files.append(png_path)

        # OpenEXR format (32-bit float, for VFX workflows)
        if "exr" in export_formats:
            exr_path = output_dir / "depth_metric.exr"
            self._save_depth_exr(depth_meters, exr_path)
            exported_files.append(exr_path)

        return exported_files

    def _save_depth_tiff(self, depth: np.ndarray, path: Path):
        """Save depth as 16-bit TIFF."""
        try:
            import tifffile

            # Normalize to 16-bit range
            depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            depth_16bit = (depth_norm * 65535).astype(np.uint16)
            tifffile.imwrite(path, depth_16bit)
        except ImportError:
            # Fallback to PIL
            from PIL import Image

            depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            depth_16bit = (depth_norm * 65535).astype(np.uint16)
            Image.fromarray(depth_16bit).save(path)

    def _save_depth_png(self, depth: np.ndarray, path: Path):
        """Save depth as normalized PNG for visualization."""
        from PIL import Image

        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth_8bit = (depth_norm * 255).astype(np.uint8)
        Image.fromarray(depth_8bit).save(path)

    def _save_depth_exr(self, depth: np.ndarray, path: Path):
        """Save depth as OpenEXR (32-bit float)."""
        try:
            import OpenEXR
            import Imath

            # OpenEXR implementation
            # This requires pyexr or OpenEXR package
            # Placeholder for now
            print(f"Warning: EXR export not implemented, skipping {path}")
        except ImportError:
            print(f"Warning: OpenEXR not available, skipping {path}")
