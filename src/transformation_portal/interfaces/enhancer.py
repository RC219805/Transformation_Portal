"""
Enhancer Interface

Base contract for image enhancement algorithms.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np


class EnhancementError(Exception):
    """Raised when enhancement operation fails."""
    pass


class Enhancer(ABC):
    """
    Base interface for image enhancement operations.
    
    Enhancers apply perceptual improvements to images such as clarity,
    sharpness, tone adjustments, and material-specific enhancements.
    
    Contract Specifications:
    - **Input**: numpy array (H, W, C) in [0, 1] float32 or [0, 255] uint8
    - **Output**: Enhanced image, same shape and dtype as input
    - **State**: Must be reusable across multiple images
    - **Parameters**: Enhancement strength should be controllable
    """
    
    @abstractmethod
    def enhance(
        self,
        image: np.ndarray,
        strength: float = 1.0,
        **kwargs
    ) -> np.ndarray:
        """
        Apply enhancement to input image.
        
        Args:
            image: Input image (H, W, C) numpy array
            strength: Enhancement strength in [0, 1] (0=no effect, 1=full effect)
            **kwargs: Enhancer-specific parameters
            
        Returns:
            Enhanced image, same shape and dtype as input
            
        Raises:
            EnhancementError: If enhancement fails
            ValueError: If strength is out of valid range
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Return current enhancer configuration.
        
        Returns:
            Dictionary of configuration parameters (must be JSON-serializable)
        """
        pass
    
    def validate_strength(self, strength: float) -> None:
        """
        Validate enhancement strength parameter.
        
        Args:
            strength: Enhancement strength value
            
        Raises:
            ValueError: If strength is outside [0, 1] range
        """
        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"Strength must be in [0, 1], got {strength}")


class AdaptiveEnhancer(Enhancer):
    """
    Extended enhancer interface for adaptive/context-aware enhancement.
    
    Adaptive enhancers can analyze image content and adjust their
    behavior based on detected materials, lighting, or other factors.
    """
    
    @abstractmethod
    def analyze(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze image to determine adaptive enhancement parameters.
        
        Args:
            image: Input image (H, W, C) numpy array
            
        Returns:
            Dictionary with analysis results (e.g., detected materials,
            lighting conditions, suggested strength values)
        """
        pass
    
    @abstractmethod
    def enhance_adaptive(
        self,
        image: np.ndarray,
        analysis: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Apply adaptive enhancement based on image analysis.
        
        Args:
            image: Input image (H, W, C) numpy array
            analysis: Optional pre-computed analysis (from analyze())
            **kwargs: Enhancer-specific parameters
            
        Returns:
            Enhanced image, same shape and dtype as input
            
        Raises:
            EnhancementError: If enhancement fails
        """
        pass
