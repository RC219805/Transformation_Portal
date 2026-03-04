"""Postprocessing module for DA3 depth maps.

Handles metric scaling, filtering, and edge preservation.
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Any, List

import numpy as np
from scipy.ndimage import median_filter

from .config import PostprocessingConfig

if TYPE_CHECKING:
    from .inference import DepthResult

logger = logging.getLogger(__name__)

_opencv = None
_opencv_import_attempted = False


def _get_opencv() -> Any:
    """Lazily import cv2.

    Environments without OpenCV don't pay
    import cost upfront.
    """
    global _opencv
    global _opencv_import_attempted

    if _opencv_import_attempted:
        return _opencv

    _opencv_import_attempted = True
    try:
        _opencv = importlib.import_module("cv2")
    except Exception as exc:  # pragma: no cover
        logger.debug("OpenCV unavailable for" " postprocessing bilateral" " filter: %s", exc)
        _opencv = None
    return _opencv


# Edge refinement is optional
# (may not be present in stripped-down deployments).
try:
    from .edge_refinement import DepthRefiner  # type: ignore
except ImportError:
    DepthRefiner = None


class _NoOpDepthRefiner:
    """Fallback refiner when edge_refinement
    module isn't available.
    """

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._stats = {"enabled": False, "available": False}

    def refine(self, depth: np.ndarray, image: np.ndarray) -> np.ndarray:
        return depth

    def get_stats(self) -> dict[str, Any]:
        return dict(self._stats)


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

        # Edge-aware refinement (Optional Module)
        if self.refiner and getattr(
            self.config.refinement,
            "enable_refinement",
            False,
        ):
            depth = self.refiner.refine(depth, result.original_image)

        # Update result
        result.depth_map = depth
        result.metadata["postprocessing"] = self.config.__dict__
        result.metadata["refinement"] = self.refiner.get_stats()

        return result

    def _bilateral_filter(
        self,
        depth: np.ndarray,
        image: np.ndarray,
        sigma_color: float,
        sigma_space: float,
    ) -> np.ndarray:
        """Apply bilateral filter with OpenCV acceleration.

        Uses cv2.bilateralFilter for 2-3x speedup
        via SIMD optimization compared to scipy.

        Args:
            depth: Depth map to filter (float32)
            image: Reference image for
                RGB-guided joint bilateral
                (when cv2.ximgproc available)
            sigma_color: Color space sigma
            sigma_space: Spatial sigma

        Returns:
            Filtered depth map (float32, same range as input)
        """
        try:
            opencv = _get_opencv()
            if opencv is None:
                raise ImportError

            # Heuristics for sigmaColor scaling
            # based on depth value range
            # Legacy configs used 0-255-ish values
            LEGACY_SIGMA_COLOR_THRESHOLD = 100
            # Normalized depth typically in [0,1]
            NORMALIZED_DEPTH_THRESHOLD = 2.0

            # Sanitize depth: remove NaN/inf to prevent outlier explosion
            depth_clean = np.copy(depth)
            depth_clean = np.nan_to_num(
                depth_clean,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            # Use robust percentiles (p1/p99)
            # to avoid outlier explosion.
            # Prevents single hot pixels/artifacts
            # from breaking the scaling.
            depth_p1 = float(np.percentile(depth_clean, 1))
            depth_p99 = float(np.percentile(depth_clean, 99))
            depth_range = depth_p99 - depth_p1

            # Scale sigmaColor based on depth
            # range to handle:
            # - Normalized ~[0,1]: as-is
            # - Metric/unbounded: proportional
            # - Legacy 0-255-ish: normalize
            if depth_range < NORMALIZED_DEPTH_THRESHOLD:
                # Normalized depth [0,1] - use sigma_color directly
                effective_sigma_color = sigma_color
            elif sigma_color > LEGACY_SIGMA_COLOR_THRESHOLD:
                # Legacy config detected
                effective_sigma_color = sigma_color / 255.0 * depth_range
            else:
                # Metric/unbounded depth
                effective_sigma_color = sigma_color * depth_range

            # Try RGB-guided joint bilateral
            # filter if cv2.ximgproc available
            # (better edges)
            try:
                expected_joint_filter_errors = (
                    AttributeError,
                    TypeError,
                    ValueError,
                )
                opencv_error = getattr(opencv, "error", None)
                if isinstance(opencv_error, type) and issubclass(opencv_error, Exception):
                    expected_joint_filter_errors += (opencv_error,)

                # Ensure image is uint8 RGB
                if image.dtype == np.float32:
                    image_u8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
                elif image.dtype == np.float64:
                    image_u8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
                elif image.dtype == np.uint16:
                    # Scale 16-bit to 8-bit
                    image_u8 = (image / 257).astype(np.uint8)
                elif np.issubdtype(
                    image.dtype,
                    np.integer,
                ):
                    # Handle other integer types
                    image_u8 = np.clip(
                        image,
                        0,
                        255,
                    ).astype(np.uint8)
                else:
                    # Already uint8 or unknown
                    image_u8 = image.astype(np.uint8)

                # Convert depth to float32
                depth_f32 = depth.astype(np.float32) if depth.dtype != np.float32 else depth

                # d=0 for auto-derivation
                d_param = 0

                # RGB-guided joint bilateral
                filtered = opencv.ximgproc.jointBilateralFilter(
                    image_u8,
                    depth_f32,
                    d=d_param,
                    sigmaColor=(effective_sigma_color),
                    sigmaSpace=sigma_space,
                )
                return filtered

            except expected_joint_filter_errors as exc:
                logger.debug("Joint bilateral filter" " unavailable/incompatible;" " using bilateral" " fallback: %s", exc)
            except Exception:
                logger.exception("Unexpected joint bilateral" " filter failure; using" " bilateral fallback")

            # cv2.ximgproc not available or
            # type mismatch - fall back to
            # standard bilateral
            # Guide shape validation: ensure depth is 2D
            if depth.ndim != 2:
                logger.warning(
                    "Bilateral filter expects" " 2D depth, got shape %s." " Using first channel.",
                    depth.shape,
                )
                depth = depth[:, :, 0] if depth.ndim == 3 else depth.reshape(depth.shape[:2])

            depth_f32 = depth.astype(np.float32) if depth.dtype != np.float32 else depth

            # Use d=0 for auto-derivation
            # (avoids d=1 degenerate case)
            d_param = 0

            # Apply bilateral filter directly
            # on float32 (no quantization)
            filtered = opencv.bilateralFilter(
                depth_f32,
                d=d_param,
                sigmaColor=effective_sigma_color,
                sigmaSpace=sigma_space,
            )
            return filtered

        except ImportError:
            # Fallback to scipy for environments without OpenCV
            from scipy.ndimage import gaussian_filter

            return gaussian_filter(depth, sigma=sigma_space / 3.0)

    def _preserve_edges(
        self,
        depth: np.ndarray,
        image: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        del threshold
        try:
            gray = np.asarray(image)
        except Exception:
            logger.warning(
                "Edge preservation skipped:" " image is not array-like" " (%s)",
                type(image).__name__,
            )
            return depth

        gray = np.mean(gray, axis=2) if gray.ndim == 3 else np.squeeze(gray)
        if gray.ndim != 2:
            logger.warning(
                "Edge preservation skipped:" " expected 2D grayscale" " image, got shape %s",
                getattr(gray, "shape", None),
            )
            return depth

        # Simple mask-based preservation logic
        # (placeholder for more complex logic)
        # In production, this would likely
        # blend based on edge magnitude
        return depth

    def fuse_multiview(
        self,
        results: List[DepthResult],
    ) -> DepthResult:
        """Simple fusion stub."""
        if not results:
            raise ValueError("No results to fuse")
        depths = np.stack(
            [r.depth_map for r in results],
            axis=0,
        )

        if self.config.fusion_mode == "mean":
            fused = np.mean(depths, axis=0)
        elif self.config.fusion_mode == "median":
            fused = np.median(depths, axis=0)
        else:
            fused = np.mean(depths, axis=0)  # Default

        # Keep import lazy at module scope
        # while ensuring runtime availability.
        from .inference import DepthResult

        return DepthResult(
            depth_map=fused,
            original_image=results[0].original_image,
            metadata={"fusion_mode": self.config.fusion_mode},
        )
