"""
Image Processor Interface

Base contract for all image processing operations in Transformation Portal.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np


class ProcessingError(Exception):
    """Raised when image processing fails."""
    pass


class ImageProcessor(ABC):
    """
    Base interface for all image processing operations.
    
    Contract Specifications:
    - **Input**: numpy array (H, W, C) in [0, 1] float32 or [0, 255] uint8
    - **Output**: numpy array of same shape and dtype as input
    - **State**: Processors should be stateless OR clearly document state
    - **Config**: Configuration must be JSON-serializable for reproducibility
    """
    
    @abstractmethod
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Process input image according to processor-specific algorithm.

        Args:
            image: Input image (H, W, C) numpy array in [0, 1] float32 or [0, 255] uint8
            **kwargs: Processor-specific parameters

        Returns:
            Processed image, same shape and dtype as input

        Raises:
            ProcessingError: If processing fails
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Return current processor configuration.

        Returns:
            Dictionary of configuration parameters (must be JSON-serializable for reproducibility)
        """
        pass


class VideoProcessor(ABC):
    """Base interface for video processing operations."""
    
    @abstractmethod
    def process_video(self, input_path: str, output_path: str, **kwargs) -> None:
        """Process entire video file."""
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return current processor configuration."""
        pass
