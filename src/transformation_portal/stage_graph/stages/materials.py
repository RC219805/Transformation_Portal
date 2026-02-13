"""
Material segmentation stage.

Segments image into material types (wood, metal, glass, etc.)
for material-aware enhancement.
"""

from __future__ import annotations

import hashlib
import math
from typing import Dict, Optional, Tuple, Union

import numpy as np

from ..stage import Stage, StageContext, StageResult, StageStatus

# Type aliases for segmentation output normalization
Mask = np.ndarray  # Binary mask (H, W) with values 0.0-1.0
MaskWithConfidence = Tuple[np.ndarray, float]  # (mask, confidence_score)
SegmentValue = Union[Mask, MaskWithConfidence]  # Flexible backend output


def _coerce_mask_conf(value: SegmentValue) -> Tuple[np.ndarray, Optional[float]]:
    """
    Accept either mask or (mask, confidence) tuple.

    Args:
        value: Either a mask array or (mask, confidence) tuple

    Returns:
        (mask, confidence) where confidence is:
        - None if not provided
        - None if invalid (non-finite or outside [0,1])
        - float in [0.0, 1.0] if valid

    Raises:
        ValueError: If value is not a valid mask or tuple format
    """
    if isinstance(value, tuple) and len(value) == 2:
        mask, conf = value
        if not isinstance(mask, np.ndarray):
            raise ValueError(f"Invalid mask type in tuple: {type(mask)}")

        # Validate confidence is finite and in [0,1]
        try:
            conf_float = float(conf)
            # Invalid confidence: keep mask, discard confidence
            if not (0.0 <= conf_float <= 1.0 and math.isfinite(conf_float)):
                return mask, None
            return mask, conf_float
        except (ValueError, TypeError):
            # Invalid confidence type: keep mask, discard confidence
            return mask, None
    elif isinstance(value, np.ndarray):
        return value, None
    else:
        raise ValueError(f"Unexpected segmentation output format: {type(value)}")


def _coerce_results(results: Dict[str, SegmentValue]) -> Tuple[Dict[str, np.ndarray], Dict[str, Optional[float]]]:
    """
    Normalize segmentation results to separate masks and confidences.

    Args:
        results: Dict mapping material names to masks or (mask, confidence) tuples

    Returns:
        (masks, confidences) tuple of dicts

    Note:
        Invalid entries are silently skipped. Consumers should validate
        the results if completeness is required.
    """
    masks: Dict[str, np.ndarray] = {}
    confidences: Dict[str, Optional[float]] = {}

    for name, value in results.items():
        try:
            mask, conf = _coerce_mask_conf(value)
            masks[name] = mask
            confidences[name] = conf
        except (ValueError, TypeError):
            # Skip invalid entries silently - caller should log if needed
            pass

    return masks, confidences


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
        material_masks = self._segment_materials(image, depth_map, context)

        duration_ms = (time.time() - start) * 1000

        return StageResult(
            stage_name=self.name,
            stage_version=self.version,
            status=StageStatus.COMPLETED,
            artifacts={
                "material_masks": material_masks,
                "materials": material_masks,
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
        context: StageContext,
    ) -> Dict[str, np.ndarray]:
        """
        Segment image into material types.

        This method handles multiple segmentation backend output formats:
        - Modern backends: return Dict[str, Tuple[mask, confidence]]
        - Legacy backends: return Dict[str, mask]
        - Mixed outputs: some tuples, some masks

        Args:
            image: Input image (H, W, 3)
            depth_map: Optional depth map (H, W)
            context: Stage context for config access

        Returns:
            Dict mapping material names to binary masks (float32)

        Raises:
            Exception: Propagates backend failures if strict mode enabled
        """
        h, w = image.shape[:2]
        device = context.device

        if self._segmenter == "heuristic":
            # Use color-based heuristics
            materials = {}

            # Wood: warm tones
            hsv = self._rgb_to_hsv(image)
            wood_mask = (
                (hsv[:, :, 0] > 10)
                & (hsv[:, :, 0] < 40)  # Hue
                & (hsv[:, :, 1] > 0.2)  # Saturation
                & (hsv[:, :, 2] > 0.3)  # Value
            ).astype(np.float32)
            materials["wood"] = wood_mask

            # Metal: high value, low saturation
            metal_mask = ((hsv[:, :, 1] < 0.2) & (hsv[:, :, 2] > 0.5)).astype(np.float32)  # Low saturation  # High value
            materials["metal"] = metal_mask

            # Glass: use depth if available (closer objects)
            if depth_map is not None:
                glass_mask = (depth_map > 0.7).astype(np.float32)
                materials["glass"] = glass_mask

            return materials

        try:
            # Use actual segmenter
            results = self._segmenter.segment(image)

            # Normalize outputs using helper functions
            # First, validate and log any invalid entries
            for material_name, value in results.items():
                try:
                    _coerce_mask_conf(value)
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Unexpected segmentation output format for {material_name}: {e}. Skipping.")

            material_masks, material_confidences = _coerce_results(results)

            # Convert to float32 and log confidence
            final_masks = {}
            for material_name, mask in material_masks.items():
                if isinstance(mask, np.ndarray):
                    final_masks[material_name] = mask.astype(np.float32)
                    conf = material_confidences.get(material_name)
                    if conf is not None:
                        self.logger.debug(f"{material_name}: {conf:.0%} confidence")
                    else:
                        self.logger.debug(f"{material_name}: detected (no confidence score)")

            return final_masks

        except Exception as e:
            self.logger.error(f"Material segmentation failed: {e}")

            # Default to soft failure (pipeline continues) unless strict mode enabled
            strict = context.get_config("materials_segmentation_strict", False)
            if strict:
                raise  # Hard failure: stop pipeline
            else:
                # Soft failure: log + return empty, pipeline continues
                self.logger.warning(
                    "Returning empty materials (soft failure). " "Set materials_segmentation_strict=True to make this fatal."
                )
                return {}

    @staticmethod
    def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to HSV."""
        from skimage import color

        return color.rgb2hsv(rgb / 255.0 if rgb.max() > 1 else rgb)
