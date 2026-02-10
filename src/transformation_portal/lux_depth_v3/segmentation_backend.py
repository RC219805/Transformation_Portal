"""Material segmentation backend for Materials V3.

This module provides material segmentation functionality for the Materials V3 pipeline.

Current Implementation: Stub backend (returns empty masks)
Future Enhancement: EfficientSAM-based real material segmentation

Design Notes:
- Stub backend is the default to avoid heavy ML dependencies
- EfficientSAM integration requires additional dependencies (segment-anything, torch)
- Backend is configurable via EnhanceConfig.material_segmentation_backend
- Real segmentation would use EfficientSAM for automatic material detection

EfficientSAM Integration (Future Work):
- Model: EfficientSAM (Facebook AI) - lightweight Segment Anything variant
- License: Apache 2.0 (commercial use allowed)
- Model size: ~30-40MB (vs 2.4GB for SAM)
- Performance: 10-20x faster than original SAM
- Dependency: pip install segment-anything efficientam
- Implementation: Prompt-based segmentation with material classifiers

For now, Materials V3 operates with manually provided masks or empty segmentation.
This allows the pixel ops infrastructure to be tested and deployed independently.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


def segment_materials(image: np.ndarray, config) -> Dict[str, np.ndarray]:
    """Segment image into material masks.

    Current implementation: Stub that returns empty masks.
    Future: EfficientSAM-based automatic material detection.

    Args:
        image: Input image as numpy array (H, W, 3) in RGB
        config: EnhanceConfig instance with segmentation settings
            - enable_material_segmentation: Enable/disable segmentation
            - material_segmentation_backend: Backend to use ("stub" or "efficientam")

    Returns:
        Dict mapping material names to binary masks (H, W) with values 0.0-1.0
        Example: {"glass": mask1, "wood": mask2, ...}

        For stub backend, returns empty dict.

    Future Implementation Notes:
        EfficientSAM backend would:
        1. Run EfficientSAM to generate segment proposals
        2. Classify segments using material classifier (ResNet/ViT)
        3. Return confidence-weighted masks for detected materials
        4. Support materials: glass, water, foliage, wood, stone, metal, fabric, stucco
    """
    # Check if segmentation is enabled
    enable_segmentation = getattr(config, "enable_material_segmentation", False)

    if not enable_segmentation:
        logger.debug("Material segmentation disabled in config")
        return {}

    # Get backend selection
    backend = getattr(config, "material_segmentation_backend", "stub")

    if backend == "stub":
        logger.debug("Using stub segmentation backend (returns empty masks)")
        return {}

    elif backend == "efficientam":
        # Future: EfficientSAM integration
        logger.warning(
            "EfficientSAM backend not yet implemented. "
            "Falling back to stub backend (empty masks). "
            "This is expected - EfficientSAM integration is future work."
        )
        return {}

    else:
        logger.error(f"Unknown segmentation backend: {backend}. Using stub backend.")
        return {}


# Future: EfficientSAM integration placeholder
def _segment_with_efficientam(image: np.ndarray, config) -> Dict[str, np.ndarray]:
    """EfficientSAM-based material segmentation (not yet implemented).

    Future implementation would:
    1. Load EfficientSAM model (cached)
    2. Generate segment proposals
    3. Classify segments by material type
    4. Return confidence-weighted masks

    Args:
        image: Input image (H, W, 3)
        config: Configuration with EfficientSAM settings

    Returns:
        Material masks dict

    Raises:
        NotImplementedError: This is a placeholder for future work
    """
    raise NotImplementedError(
        "EfficientSAM backend not yet implemented. " "See docs/architecture/materials_v3_design.md for integration plan."
    )
