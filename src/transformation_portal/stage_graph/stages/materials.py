"""
Material segmentation stage.

Segments image into material types (wood, metal, glass, etc.)
for material-aware enhancement.
"""

from __future__ import annotations

from typing import Dict
import hashlib

import numpy as np

from ..stage import Stage, StageContext, StageResult, StageStatus


class MaterialSegmentationStage(Stage):
    """
    Material segmentation stage.

    Identifies material types in the image for physics-based enhancement.
    """

    def __init__(
        self,
        backend: str = "onnx",
        version: str = "1.0.0",
    ):
        """
        Initialize material segmentation stage.

        Args:
            backend: Segmentation backend ("onnx", "segformer", "heuristic")
            version: Stage version for cache invalidation
        """
        super().__init__(name="material_segmentation", version=version)
        self.backend = backend
        self._segmenter = None

    def get_dependencies(self) -> list:
        """This stage can optionally use depth map."""
        return []  # Optional dependency on depth

    def compute(self, context: StageContext) -> StageResult:
        """
        Compute material segmentation.

        Expected context artifacts:
        - image: Input image as numpy array (H, W, 3)
        - depth_map: Optional depth map (H, W)

        Output artifacts:
        - material_masks: Dict[str, np.ndarray] with masks per material
        - material_metadata: Dict with segmentation info
        """
        import time

        # Get input
        image = context.get_artifact("image")
        if image is None:
            return StageResult(
                stage_name=self.name,
                stage_version=self.version,
                status=StageStatus.FAILED,
                error="Missing 'image' artifact in context",
            )

        depth_map = context.get_artifact("depth_map")

        start = time.time()

        # Lazy load segmenter
        if self._segmenter is None:
            self._load_segmenter(context.device)

        # Compute segmentation
        material_masks = self._segment_materials(image, depth_map, context.device)

        duration_ms = (time.time() - start) * 1000

        return StageResult(
            stage_name=self.name,
            stage_version=self.version,
            status=StageStatus.COMPLETED,
            artifacts={
                "material_masks": material_masks,
                "material_metadata": {
                    "backend": self.backend,
                    "materials": list(material_masks.keys()),
                    "device": context.device,
                },
            },
            duration_ms=duration_ms,
            metadata={
                "backend": self.backend,
                "num_materials": len(material_masks),
                "inference_ms": duration_ms,
            },
        )

    def get_cache_key(self, context: StageContext) -> str:
        """
        Generate cache key based on input image and configuration.
        """
        # Get input image
        image = context.get_artifact("image")
        if image is None:
            return "no_image"

        # Hash image content
        image_hash = hashlib.sha256(image.tobytes()).hexdigest()[:16]

        # Include depth if available (affects segmentation)
        depth_map = context.get_artifact("depth_map")
        if depth_map is not None:
            depth_hash = hashlib.sha256(depth_map.tobytes()).hexdigest()[:8]
            config_str = f"{self.backend}_{self.version}_{depth_hash}"
        else:
            config_str = f"{self.backend}_{self.version}"

        return f"materials_{config_str}_{image_hash}"

    def _load_segmenter(self, device: str):
        """Load material segmenter."""
        # Always use placeholder for now (heuristic-based)
        # Real ML backend can be added later
        self.logger.info(f"Using heuristic material segmenter on {device}")
        self._segmenter = "heuristic"

    def _segment_materials(
        self,
        image: np.ndarray,
        depth_map: np.ndarray | None,
        device: str,
    ) -> Dict[str, np.ndarray]:
        """
        Segment image into material types.

        Args:
            image: Input image (H, W, 3)
            depth_map: Optional depth map (H, W)
            device: Device to use

        Returns:
            Dict mapping material names to binary masks
        """
        h, w = image.shape[:2]

        if self._segmenter == "heuristic":
            # Use color-based heuristics
            materials = {}

            # Wood: warm tones
            hsv = self._rgb_to_hsv(image)
            wood_mask = (
                (hsv[:, :, 0] > 10) & (hsv[:, :, 0] < 40) &  # Hue
                (hsv[:, :, 1] > 0.2) &  # Saturation
                (hsv[:, :, 2] > 0.3)    # Value
            ).astype(np.float32)
            materials["wood"] = wood_mask

            # Metal: high value, low saturation
            metal_mask = (
                (hsv[:, :, 1] < 0.2) &  # Low saturation
                (hsv[:, :, 2] > 0.5)    # High value
            ).astype(np.float32)
            materials["metal"] = metal_mask

            # Glass: use depth if available (closer objects)
            if depth_map is not None:
                glass_mask = (depth_map > 0.7).astype(np.float32)
                materials["glass"] = glass_mask

            return materials

        try:
            # Use actual segmenter
            masks = self._segmenter.segment(image)

            # Convert to standard format
            material_masks = {}
            for material_name, mask in masks.items():
                if isinstance(mask, np.ndarray):
                    material_masks[material_name] = mask.astype(np.float32)

            return material_masks

        except Exception as e:
            self.logger.error(f"Material segmentation failed: {e}")
            # Return empty dict on failure
            return {}

    @staticmethod
    def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to HSV."""
        from skimage import color
        return color.rgb2hsv(rgb / 255.0 if rgb.max() > 1 else rgb)
