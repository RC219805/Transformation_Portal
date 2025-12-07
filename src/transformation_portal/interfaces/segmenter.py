"""
Segmenter Interface

Base contract for material and semantic segmentation operations.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np


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
    - **Output**: Dictionary mapping categories to binary masks
    - **Masks**: Binary masks as boolean arrays, shape (H, W)
    - **Coverage**: Masks may overlap or have gaps (depend on algorithm)
    """
    
    @abstractmethod
    def segment(
        self,
        image: np.ndarray,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Segment image into labeled regions.
        
        Args:
            image: Input image (H, W, C) numpy array
            **kwargs: Segmenter-specific parameters
            
        Returns:
            Dictionary mapping category names to binary masks (H, W) boolean arrays
            
        Raises:
            SegmentationError: If segmentation fails
            
        Example:
            >>> masks = segmenter.segment(image)
            >>> wood_mask = masks['wood']  # boolean array (H, W)
            >>> wood_pixels = image[wood_mask]  # Extract wood pixels
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
        **kwargs
    ) -> Dict[MaterialType, np.ndarray]:
        """
        Segment image by material types.
        
        Args:
            image: Input image (H, W, C) numpy array
            materials: Optional list of materials to detect (None = all supported)
            **kwargs: Segmenter-specific parameters
            
        Returns:
            Dictionary mapping MaterialType to binary masks (H, W) boolean arrays
            
        Raises:
            SegmentationError: If segmentation fails
        """
        pass
    
    @abstractmethod
    def get_material_properties(
        self,
        material: MaterialType
    ) -> Dict[str, Any]:
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
        self,
        image: np.ndarray,
        categories: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Segment image by semantic categories.
        
        Args:
            image: Input image (H, W, C) numpy array
            categories: Optional list of categories to detect (None = all)
            **kwargs: Segmenter-specific parameters
            
        Returns:
            Dictionary mapping category names to binary masks
            
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
