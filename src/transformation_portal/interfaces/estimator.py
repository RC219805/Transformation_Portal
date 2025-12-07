"""
Estimator Interface

Base contract for depth estimation and surface property estimation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import numpy as np


class EstimationError(Exception):
    """Raised when estimation operation fails."""
    pass


class DepthEstimator(ABC):
    """
    Base interface for monocular depth estimation.
    
    Depth estimators predict per-pixel depth (distance from camera)
    from a single RGB image, enabling depth-aware processing.
    
    Contract Specifications:
    - **Input**: RGB image (H, W, 3) in [0, 1] float32 or [0, 255] uint8
    - **Output**: Depth map (H, W) float32, normalized to [0, 1]
    - **Depth Convention**: 0=far (background), 1=near (foreground)
    - **Accuracy**: Relative depth only (not metric depth)
    """
    
    @abstractmethod
    def estimate_depth(
        self,
        image: np.ndarray,
        normalize: bool = True,
        **kwargs
    ) -> np.ndarray:
        """
        Estimate depth map from RGB image.
        
        Args:
            image: Input RGB image (H, W, 3) numpy array
            normalize: Whether to normalize depth to [0, 1] range
            **kwargs: Estimator-specific parameters
            
        Returns:
            Depth map (H, W) float32, normalized to [0, 1] if normalize=True
            (0=far/background, 1=near/foreground)
            
        Raises:
            EstimationError: If depth estimation fails
            
        Example:
            >>> depth = estimator.estimate_depth(image)
            >>> foreground_mask = depth > 0.7  # Near objects
            >>> background_mask = depth < 0.3  # Far objects
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Return information about the depth estimation model.
        
        Returns:
            Dictionary with model metadata (name, version, architecture,
            input_size, etc.)
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Return current estimator configuration.
        
        Returns:
            Dictionary of configuration parameters (must be JSON-serializable)
        """
        pass
    
    def invert_depth(self, depth: np.ndarray) -> np.ndarray:
        """
        Invert depth map (swap near and far).
        
        Args:
            depth: Depth map (H, W) in [0, 1]
            
        Returns:
            Inverted depth map where 0=near, 1=far
        """
        return 1.0 - depth


class NormalEstimator(ABC):
    """
    Base interface for surface normal estimation.
    
    Normal estimators predict per-pixel surface normals (3D orientation)
    from RGB images, useful for relighting and material enhancement.
    
    Contract Specifications:
    - **Input**: RGB image (H, W, 3)
    - **Output**: Normal map (H, W, 3) with unit vectors
    - **Normal Convention**: Camera-space normals, Z-axis points toward camera
    - **Range**: Each channel in [-1, 1], unit length vectors
    """
    
    @abstractmethod
    def estimate_normals(
        self,
        image: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Estimate surface normal map from RGB image.
        
        Args:
            image: Input RGB image (H, W, 3) numpy array
            **kwargs: Estimator-specific parameters
            
        Returns:
            Normal map (H, W, 3) float32 with unit vectors in camera space
            Each pixel is a unit vector (nx, ny, nz) where nz > 0 points toward camera
            
        Raises:
            EstimationError: If normal estimation fails
            
        Example:
            >>> normals = estimator.estimate_normals(image)
            >>> # normals[i,j] is a 3D unit vector for pixel (i,j)
            >>> assert np.allclose(np.linalg.norm(normals, axis=2), 1.0)
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Return information about the normal estimation model.
        
        Returns:
            Dictionary with model metadata
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Return current estimator configuration.
        
        Returns:
            Dictionary of configuration parameters (must be JSON-serializable)
        """
        pass


class UnifiedEstimator(DepthEstimator, NormalEstimator):
    """
    Combined interface for models that estimate both depth and normals.
    
    Some architectures can jointly predict depth and surface normals,
    which can be more efficient than running separate models.
    """
    
    @abstractmethod
    def estimate_geometry(
        self,
        image: np.ndarray,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Jointly estimate depth and normals.
        
        Args:
            image: Input RGB image (H, W, 3) numpy array
            **kwargs: Estimator-specific parameters
            
        Returns:
            Tuple of (depth_map, normal_map):
            - depth_map: (H, W) float32 in [0, 1]
            - normal_map: (H, W, 3) float32 unit vectors
            
        Raises:
            EstimationError: If estimation fails
        """
        pass
