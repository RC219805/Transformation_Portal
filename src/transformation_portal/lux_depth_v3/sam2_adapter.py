"""SAM2 adapter for Materials V3 integration.

This module provides a clean bridge between SAM2Backend (spatial_ai)
and Materials V3's segmentation API.

Architecture (Option B - Clean Separation):
- SAM2Backend stays in spatial_ai with SegmentationInput/Result contracts
- SAM2MaterialsAdapter wraps SAM2Backend for Materials V3 compatibility
- Materials V3 expects: segment(image: np.ndarray) -> Dict[str, Tuple[mask, conf]]
- SAM2Backend expects: segment(SegmentationInput) -> SegmentationResult

Material Labeling:
- SAM2 returns generic masks (no material labels)
- Adapter applies heuristic material classification
- Future: Could integrate CLIP for zero-shot classification

License: Apache 2.0 (SAM2 is commercial-safe, no tier restrictions)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import numpy as np

from transformation_portal.lux_depth_v3.protocols.segmentation_backend import SegmentationBackend, SegmentationBackendInfo
from transformation_portal.spatial_ai.segmentation.contracts import SegmentationInput
from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

logger = logging.getLogger(__name__)


class SAM2MaterialsAdapter:
    """Adapter bridging SAM2Backend to Materials V3 API.

    This adapter:
    1. Converts Materials V3 API (segment(image)) to SAM2 contracts
    2. Applies heuristic material labeling to generic masks
    3. Returns Materials V3 expected format: Dict[material, (mask, confidence)]

    Implements SegmentationBackend protocol for Materials V3 compatibility.

    Performance (estimated, 1024×1024 on M4):
    - CPU: ~3-5s (SAM2-base)
    - MPS: Not supported by pipeline (falls back to CPU)
    - CUDA: ~1-2s (SAM2-base)

    Memory: ~1.2GB model + ~500MB inference overhead
    """

    def __init__(
        self,
        model_size: Literal["base", "large"] = "base",
        device: Literal["cuda", "cpu", "mps"] = "cuda",
        revision: Optional[str] = None,
    ):
        """Initialize SAM2 adapter.

        Args:
            model_size: SAM2 model variant ("base" or "large").
            device: Target device (note: MPS falls back to CPU for pipeline).
            revision: HuggingFace model revision (None = latest).
        """
        self._sam2_backend = SAM2Backend(
            model_size=model_size,
            device=device,
            revision=revision,
        )
        self._model_loaded = False
        self._device = device

        logger.info(f"SAM2MaterialsAdapter initialized: model={model_size}, device={device}")

    @property
    def info(self) -> SegmentationBackendInfo:
        """Return backend metadata for Materials V3."""
        return SegmentationBackendInfo(
            name=f"SAM2 ({self._sam2_backend.model_size})",
            model_id=f"facebook/sam2-hiera-{self._sam2_backend.model_size}-plus",
            requires_gpu=False,  # Works on CPU, but much slower
            requires_weights=True,
            approximate_memory_mb=1200,  # Base model is ~1.2GB
            description=f"SAM2 segmentation with heuristic material labeling (Apache 2.0 license)",
        )

    def load(self, device: str = "auto", weights_path: Optional[Path] = None) -> None:
        """Load SAM2 model (lazy loading via pipeline).

        Args:
            device: Device selection (handled in __init__, ignored here).
            weights_path: Not used (HuggingFace handles model downloads).

        Note:
            Actual model loading is lazy - happens on first segment() call.
            This is because mask-generation pipeline loads on first use.
        """
        logger.debug("SAM2MaterialsAdapter.load() - model will be lazy-loaded on first segment()")
        self._model_loaded = True  # Mark as ready

    def segment(self, image: np.ndarray) -> Dict[str, Tuple[np.ndarray, float]]:
        """Run material segmentation on an image.

        Materials V3 API compatibility method.

        Args:
            image: Input RGB image (H, W, 3), uint8 [0-255]

        Returns:
            Dict mapping material names to (mask, confidence) tuples:
            - mask: Binary mask (H, W) float32 [0.0-1.0]
            - confidence: Classification confidence [0.0-1.0]

        Raises:
            RuntimeError: If segmentation fails.
            ValueError: If image format invalid.
        """
        if not self._model_loaded:
            raise RuntimeError("SAM2 adapter not loaded. Call .load() first.")

        # Validate input
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB image (H, W, 3), got shape {image.shape}")

        if image.dtype != np.uint8:
            raise ValueError(f"Expected uint8 image, got dtype {image.dtype}")

        # Convert uint8 sRGB to float32 linear RGB for SAM2 contract
        linear_rgb = self._srgb_to_linear(image)

        # Create SegmentationInput contract
        seg_input = SegmentationInput(
            image=linear_rgb,
            gamma=1.0,  # Linear RGB as per contract
            mode="auto",  # Automatic mask generation
        )

        # Run SAM2 segmentation
        try:
            result = self._sam2_backend.segment(seg_input)
        except Exception as e:
            logger.error(f"SAM2 segmentation failed: {e}", exc_info=True)
            raise RuntimeError(f"SAM2 segmentation failed: {e}") from e

        # Apply heuristic material labeling
        material_masks = self._label_materials_heuristic(image, result)

        logger.debug(f"SAM2 segmented {len(material_masks)} materials: {list(material_masks.keys())}")

        return material_masks

    def _srgb_to_linear(self, srgb_uint8: np.ndarray) -> np.ndarray:
        """Convert sRGB uint8 to linear RGB float32.

        Args:
            srgb_uint8: (H, W, 3) uint8 [0-255] sRGB image.

        Returns:
            (H, W, 3) float32 linear RGB [0, 1].
        """
        # Normalize to [0, 1]
        srgb_float = srgb_uint8.astype(np.float32) / 255.0

        # Apply inverse sRGB gamma (approximation: gamma 2.2)
        linear = np.power(srgb_float, 2.2)

        return linear

    def _label_materials_heuristic(self, image: np.ndarray, result) -> Dict[str, Tuple[np.ndarray, float]]:
        """Apply heuristic material labeling to SAM2 masks.

        Uses simple color and position heuristics to classify generic masks
        into material categories (glass, water, foliage, sky, etc.).

        Args:
            image: Original RGB image (H, W, 3) uint8.
            result: SegmentationResult from SAM2.

        Returns:
            Dict mapping material names to (mask, confidence) tuples.
        """
        from transformation_portal.spatial_ai.segmentation.contracts import SegmentationResult

        if not isinstance(result, SegmentationResult):
            raise TypeError(f"Expected SegmentationResult, got {type(result)}")

        if result.masks.shape[0] == 0:
            logger.debug("No masks to label")
            return {}

        material_masks: Dict[str, Tuple[np.ndarray, float]] = {}

        # Convert image to HSV for color analysis
        from PIL import Image

        pil_image = Image.fromarray(image)
        hsv_image = np.array(pil_image.convert("HSV"), dtype=np.float32)

        H, W = image.shape[:2]

        for i, mask in enumerate(result.masks):
            # Ensure mask is numpy array (not torch tensor)
            mask_np = np.asarray(mask)
            mask_bool = mask_np.astype(bool)
            if not np.any(mask_bool):  # Use np.any() to avoid tensor ambiguity
                continue

            # Extract HSV statistics
            h_mean = hsv_image[mask_bool, 0].mean()
            s_mean = hsv_image[mask_bool, 1].mean()
            v_mean = hsv_image[mask_bool, 2].mean()

            # Get mask position (for sky detection)
            rows, cols = np.where(mask_bool)
            y_center = rows.mean() / H  # Normalized [0, 1]

            # Get confidence from SAM2 scores
            base_confidence = float(result.scores[i]) if i < len(result.scores) else 0.5

            # Heuristic classification
            material_label = None

            # Sky: High in image, blue/cyan hue, high saturation or low saturation (white clouds)
            if y_center < 0.3 and (150 <= h_mean <= 210 or s_mean < 50):
                material_label = "sky"
                confidence = base_confidence * 0.8  # Sky detection is moderately reliable

            # Water: Blue hue, mid-high saturation
            elif 150 <= h_mean <= 210 and s_mean > 80 and y_center > 0.3:
                material_label = "water"
                confidence = base_confidence * 0.7

            # Foliage: Green hue, mid-high saturation
            elif 60 <= h_mean <= 150 and s_mean > 60:
                material_label = "foliage"
                confidence = base_confidence * 0.75

            # Glass: Low saturation, mid-high brightness (neutral/reflective)
            elif s_mean < 40 and v_mean > 100:
                material_label = "glass"
                confidence = base_confidence * 0.6  # Glass is hard to detect

            # Default: Generic material
            else:
                material_label = "material"
                confidence = base_confidence * 0.5

            # Convert mask to float32 [0, 1]
            mask_float = mask.astype(np.float32)

            # Accumulate masks by material (merge if same material)
            if material_label in material_masks:
                existing_mask, existing_conf = material_masks[material_label]
                # Merge masks (logical OR) and average confidence
                merged_mask = np.maximum(existing_mask, mask_float)
                merged_conf = (existing_conf + confidence) / 2
                material_masks[material_label] = (merged_mask, merged_conf)
            else:
                material_masks[material_label] = (mask_float, confidence)

        logger.debug(f"Heuristic labeling: {len(result.masks)} masks → " f"{len(material_masks)} materials")

        return material_masks
