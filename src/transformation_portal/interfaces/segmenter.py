"""
Segmenter Interface

Base contract for material and semantic segmentation operations.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Type aliases for flexible segmentation output formats
Mask = np.ndarray  # Binary mask (H, W) with values typically 0.0-1.0
MaskWithConfidence = Tuple[np.ndarray, float]  # (mask, confidence_score)
SegmentResult = Dict[str, Union[Mask, MaskWithConfidence]]  # Backend can return either format


class SegmentationError(Exception):
    """Raised when segmentation operation fails."""

    pass


class MaterialType(Enum):
    """Standard material types for luxury real estate rendering."""

    WOOD = "wood"
    METAL = "metal"
    GLASS = "glass"
    STONE = "stone"
    MARBLE = "marble"
    FABRIC = "fabric"
    LEATHER = "leather"
    CONCRETE = "concrete"
    WATER = "water"
    VEGETATION = "vegetation"
    SKY = "sky"
    SKIN = "skin"
    UNKNOWN = "unknown"


class Segmenter(ABC):
    """
    Base interface for image segmentation operations.

    Segmenters identify and classify regions in images, typically
    used for material-aware processing or scene understanding.

    Contract Specifications:
    - **Input**: numpy array (H, W, C) in [0, 1] float32 or [0, 255] uint8
    - **Output**: Dictionary mapping categories to masks or (mask, confidence) tuples
    - **Masks**: Binary masks as boolean or float arrays, shape (H, W)
    - **Coverage**: Masks may overlap or have gaps (depend on algorithm)

    Output Format Flexibility:
    Implementations may return either:
    1. Dict[str, np.ndarray] - Legacy format (mask-only)
    2. Dict[str, Tuple[np.ndarray, float]] - Modern format (mask + confidence)
    3. Dict[str, Union[...]] - Mixed format (gradual migration)

    Consumers MUST handle both formats defensively to support
    backward compatibility and gradual backend migration.
    """

    @abstractmethod
    def segment(self, image: np.ndarray, **kwargs) -> SegmentResult:
        """
        Segment image into labeled regions.

        Args:
            image: Input image (H, W, C) numpy array
            **kwargs: Segmenter-specific parameters

        Returns:
            Dictionary mapping category names to:
            - Binary masks (H, W) boolean/float arrays, OR
            - (mask, confidence) tuples where confidence is [0.0-1.0]

            Modern backends MAY return confidence scores. Legacy backends
            return masks only. Consumers MUST handle both formats.

        Raises:
            SegmentationError: If segmentation fails

        Example (legacy backend):
            >>> masks = segmenter.segment(image)
            >>> wood_mask = masks['wood']  # np.ndarray
            >>> wood_pixels = image[wood_mask]

        Example (modern backend):
            >>> results = segmenter.segment(image)
            >>> wood_mask, confidence = results['wood']  # Tuple
            >>> if confidence > 0.7:
            ...     wood_pixels = image[wood_mask]

        Example (consumer handling both):
            >>> results = segmenter.segment(image)
            >>> for material, value in results.items():
            ...     if isinstance(value, tuple):
            ...         mask, conf = value
            ...     else:
            ...         mask, conf = value, None
            ...     process_material(mask, conf)
        """
        pass

    @abstractmethod
    def get_supported_categories(self) -> List[str]:
        """
        Return list of categories this segmenter can detect.

        Returns:
            List of category names (e.g., ['wood', 'metal', 'glass'])
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Return current segmenter configuration.

        Returns:
            Dictionary of configuration parameters (must be JSON-serializable)
        """
        pass


class MaterialSegmenter(Segmenter):
    """
    Specialized segmenter interface for material detection.

    Material segmenters classify pixels based on physical properties
    (reflectance, texture, color) relevant for rendering enhancement.
    """

    @abstractmethod
    def segment_materials(
        self,
        image: np.ndarray,
        materials: Optional[List[MaterialType]] = None,
        **kwargs,
    ) -> Dict[MaterialType, Union[Mask, MaskWithConfidence]]:
        """
        Segment image by material types.

        Args:
            image: Input image (H, W, C) numpy array
            materials: Optional list of materials to detect (None = all supported)
            **kwargs: Segmenter-specific parameters

        Returns:
            Dictionary mapping MaterialType to either:
            - Binary masks: (H, W) boolean/float arrays (legacy format)
            - Tuple format: (mask, confidence) where confidence is [0, 1] (modern format)

            Implementations may use either format. Consumers must handle both.

        Raises:
            SegmentationError: If segmentation fails

        Example (handling both formats):
            >>> results = segmenter.segment_materials(image)
            >>> for material, value in results.items():
            ...     if isinstance(value, tuple):
            ...         mask, confidence = value
            ...         print(f"{material}: {confidence:.0%} confidence")
            ...     else:
            ...         mask = value
            ...         print(f"{material}: detected (no confidence)")
        """
        pass

    @abstractmethod
    def get_material_properties(self, material: MaterialType) -> Dict[str, Any]:
        """
        Get physical properties for a material type.

        Args:
            material: Material type to query

        Returns:
            Dictionary with properties (e.g., roughness, metallic,
            reflectance, typical_colors)
        """
        pass


class SemanticSegmenter(Segmenter):
    """
    Specialized segmenter interface for semantic scene understanding.

    Semantic segmenters classify pixels by object categories
    (furniture, walls, floors) for scene understanding.
    """

    @abstractmethod
    def segment_semantic(
        self, image: np.ndarray, categories: Optional[List[str]] = None, **kwargs
    ) -> Dict[str, Union[Mask, MaskWithConfidence]]:
        """
        Segment image by semantic categories.

        Args:
            image: Input image (H, W, C) numpy array
            categories: Optional list of categories to detect (None = all)
            **kwargs: Segmenter-specific parameters

        Returns:
            Dictionary mapping category names to either:
            - Binary masks: (H, W) boolean/float arrays (legacy)
            - Tuple format: (mask, confidence) (modern)

        Raises:
            SegmentationError: If segmentation fails
        """
        pass

    @abstractmethod
    def get_scene_hierarchy(self) -> Dict[str, List[str]]:
        """
        Get hierarchical organization of semantic categories.

        Returns:
            Dictionary mapping parent categories to lists of children
            (e.g., {'furniture': ['chair', 'table', 'sofa']})
        """
        pass
