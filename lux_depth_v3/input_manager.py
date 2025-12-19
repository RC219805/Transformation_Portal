"""Input manager for DA3 pipeline.

Handles image loading, validation, and preprocessing for depth inference.
Supports both single images and multi-view sequences with camera poses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any

import numpy as np
from PIL import Image

from lux_depth_v3.config import InferenceMode


@dataclass
class CameraPose:
    """Camera pose information (extrinsics + intrinsics)."""
    
    # Extrinsics (camera-to-world transformation)
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))  # 3x3 rotation matrix
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))  # 3x1 translation vector
    
    # Intrinsics
    focal_length: Optional[Tuple[float, float]] = None  # (fx, fy)
    principal_point: Optional[Tuple[float, float]] = None  # (cx, cy)
    
    # Optional distortion parameters
    distortion: Optional[np.ndarray] = None  # k1, k2, p1, p2, k3
    
    def to_matrix(self) -> np.ndarray:
        """Convert to 4x4 transformation matrix."""
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation
        matrix[:3, 3] = self.translation
        return matrix
    
    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> CameraPose:
        """Create from 4x4 transformation matrix."""
        return cls(
            rotation=matrix[:3, :3],
            translation=matrix[:3, 3],
        )


@dataclass
class ImageInput:
    """Input image with metadata."""
    
    path: Optional[Path] = None
    array: Optional[np.ndarray] = None
    pose: Optional[CameraPose] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate input."""
        if self.path is None and self.array is None:
            raise ValueError("Either path or array must be provided")
        
        if self.array is not None:
            # Validate array shape (H, W, C) or (H, W)
            if self.array.ndim not in (2, 3):
                raise ValueError(f"Invalid array shape: {self.array.shape}")
            
            if self.array.ndim == 3 and self.array.shape[2] not in (1, 3, 4):
                raise ValueError(f"Invalid number of channels: {self.array.shape[2]}")
    
    def load(self) -> np.ndarray:
        """Load image as numpy array."""
        if self.array is not None:
            return self.array
        
        if self.path is None:
            raise ValueError("No path or array available")
        
        # Load image using Pillow
        with Image.open(self.path) as img:
            # Convert to RGB if needed
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            array = np.array(img)
            
            # Store metadata
            self.metadata.update({
                "original_size": img.size,
                "format": img.format,
                "mode": img.mode,
            })
            
            # Cache array for reuse
            self.array = array
            
            return array
    
    def get_size(self) -> Tuple[int, int]:
        """Get image size (width, height)."""
        if self.array is not None:
            h, w = self.array.shape[:2]
            return (w, h)
        
        if self.path is not None:
            with Image.open(self.path) as img:
                return img.size
        
        raise ValueError("No image data available")


class InputManager:
    """Manages input images and camera poses for DA3 inference."""
    
    def __init__(
        self,
        inference_mode: InferenceMode = InferenceMode.MONOCULAR,
        max_file_size_mb: float = 50.0,
    ):
        """Initialize input manager.
        
        Args:
            inference_mode: Inference mode (monocular or multi-view)
            max_file_size_mb: Maximum file size in MB (security limit)
        """
        self.inference_mode = inference_mode
        self.max_file_size_mb = max_file_size_mb
        self.inputs: List[ImageInput] = []
    
    def add_image(
        self,
        path: Optional[Union[str, Path]] = None,
        array: Optional[np.ndarray] = None,
        pose: Optional[CameraPose] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ImageInput:
        """Add an image to the input queue.
        
        Args:
            path: Path to image file
            array: NumPy array (H, W, C) or (H, W)
            pose: Camera pose (required for multi-view)
            metadata: Additional metadata
        
        Returns:
            ImageInput object
        
        Raises:
            ValueError: If input validation fails
            FileNotFoundError: If image file not found
            SecurityError: If file size exceeds limit
        """
        if path is not None:
            path = Path(path)
            
            # Security: validate file exists
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            
            # Security: validate file size
            file_size_mb = path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                raise ValueError(
                    f"File size ({file_size_mb:.1f}MB) exceeds limit "
                    f"({self.max_file_size_mb}MB): {path}"
                )
            
            # Security: validate file extension
            allowed_extensions = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}
            if path.suffix.lower() not in allowed_extensions:
                raise ValueError(f"Unsupported file type: {path.suffix}")
        
        # Multi-view requires pose
        if self.inference_mode == InferenceMode.MULTI_VIEW and pose is None:
            raise ValueError("Multi-view inference requires camera pose")
        
        # Create input
        img_input = ImageInput(
            path=path,
            array=array,
            pose=pose,
            metadata=metadata or {},
        )
        
        self.inputs.append(img_input)
        return img_input
    
    def add_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "*.jpg",
        recursive: bool = False,
    ) -> int:
        """Add all images from a directory.
        
        Args:
            directory: Directory path
            pattern: Glob pattern for image files
            recursive: Search recursively
        
        Returns:
            Number of images added
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")
        
        # Find images
        glob_method = directory.rglob if recursive else directory.glob
        image_paths = sorted(glob_method(pattern))
        
        # Add images
        count = 0
        for path in image_paths:
            try:
                self.add_image(path=path)
                count += 1
            except (ValueError, FileNotFoundError) as e:
                print(f"Skipping {path}: {e}")
        
        return count
    
    def clear(self):
        """Clear all inputs."""
        self.inputs.clear()
    
    def get_images(self) -> List[ImageInput]:
        """Get all input images."""
        return self.inputs
    
    def validate_inputs(self) -> bool:
        """Validate all inputs.
        
        Returns:
            True if all inputs are valid
        
        Raises:
            ValueError: If validation fails
        """
        if not self.inputs:
            raise ValueError("No inputs to validate")
        
        # Multi-view specific validation
        if self.inference_mode == InferenceMode.MULTI_VIEW:
            if len(self.inputs) < 2:
                raise ValueError("Multi-view requires at least 2 images")
            
            # Check all images have poses
            for i, img_input in enumerate(self.inputs):
                if img_input.pose is None:
                    raise ValueError(f"Image {i} missing camera pose")
        
        # Load and validate each image
        for i, img_input in enumerate(self.inputs):
            try:
                array = img_input.load()
                
                # Validate shape
                if array.ndim not in (2, 3):
                    raise ValueError(f"Image {i} has invalid shape: {array.shape}")
                
                # Validate data type
                if not np.issubdtype(array.dtype, np.integer) and \
                   not np.issubdtype(array.dtype, np.floating):
                    raise ValueError(f"Image {i} has invalid dtype: {array.dtype}")
                
            except Exception as e:
                raise ValueError(f"Failed to validate image {i}: {e}")
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about input images.
        
        Returns:
            Dictionary with statistics
        """
        if not self.inputs:
            return {}
        
        sizes = [img.get_size() for img in self.inputs]
        widths, heights = zip(*sizes)
        
        return {
            "num_images": len(self.inputs),
            "inference_mode": self.inference_mode.value,
            "has_poses": all(img.pose is not None for img in self.inputs),
            "size_range": {
                "width": (min(widths), max(widths)),
                "height": (min(heights), max(heights)),
            },
            "avg_size": (
                int(np.mean(widths)),
                int(np.mean(heights)),
            ),
        }
