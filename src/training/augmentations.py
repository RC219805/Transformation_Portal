#!/usr/bin/env python3
"""
Depth-Aware Data Augmentation Pipeline

Provides augmentations that maintain consistency between RGB images and depth maps.
All geometric transforms are applied identically to both modalities.

Key Features:
- Geometric transforms (flip, rotate, crop, scale) applied to both image and depth
- Color augmentations applied only to RGB images
- Architectural-specific augmentations (grid distortion, perspective transform)
- Proper normalization for both modalities

Author: Transformation Portal Team
Version: 1.0.0
"""

import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

# Try to import torch for tensor operations
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

# Try to import PIL for image operations
try:
    from PIL import Image, ImageEnhance, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Try to import cv2 for advanced transforms
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass
class AugmentationConfig:
    """Configuration for depth-aware augmentations.

    Attributes:
        horizontal_flip: Probability of horizontal flip (0.0-1.0)
        rotation_degrees: Maximum rotation angle in degrees
        brightness: Maximum brightness adjustment factor
        contrast: Maximum contrast adjustment factor
        saturation: Maximum saturation adjustment factor
        hue: Maximum hue adjustment factor
        scale_range: Tuple of (min_scale, max_scale) for random scaling
        crop_size: Size of random crop (height, width)
        grid_distortion_prob: Probability of grid distortion
        perspective_prob: Probability of perspective transform
        gaussian_blur_prob: Probability of Gaussian blur
        normalize: Whether to apply ImageNet normalization
    """
    horizontal_flip: float = 0.5
    rotation_degrees: float = 10.0
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.1
    scale_range: Tuple[float, float] = (0.8, 1.2)
    crop_size: Optional[Tuple[int, int]] = None
    grid_distortion_prob: float = 0.0
    perspective_prob: float = 0.0
    gaussian_blur_prob: float = 0.0
    normalize: bool = True


class GeometricAugmentation:
    """Geometric augmentations applied to both image and depth.

    These transforms maintain spatial consistency between RGB and depth.
    All operations use the same random parameters for both modalities.
    """

    def __init__(self, config: AugmentationConfig):
        """Initialize geometric augmentation.

        Args:
            config: Augmentation configuration
        """
        self.config = config

    def __call__(
        self,
        image: np.ndarray,
        depth: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply geometric augmentations.

        Args:
            image: RGB image array (H, W, 3)
            depth: Depth map array (H, W) or (H, W, 1)

        Returns:
            Tuple of (augmented_image, augmented_depth)
        """
        # Ensure depth is 2D for consistent processing
        depth_squeezed = depth.squeeze() if depth.ndim == 3 else depth

        # Random horizontal flip
        if random.random() < self.config.horizontal_flip:
            image = np.fliplr(image).copy()
            depth_squeezed = np.fliplr(depth_squeezed).copy()

        # Random rotation
        if self.config.rotation_degrees > 0:
            angle = random.uniform(
                -self.config.rotation_degrees,
                self.config.rotation_degrees
            )
            image = self._rotate(image, angle)
            depth_squeezed = self._rotate(depth_squeezed, angle, is_depth=True)

        # Random scale
        if self.config.scale_range != (1.0, 1.0):
            scale = random.uniform(*self.config.scale_range)
            image = self._scale(image, scale)
            depth_squeezed = self._scale(depth_squeezed, scale, is_depth=True)

        # Random crop (if specified)
        if self.config.crop_size is not None:
            image, depth_squeezed = self._random_crop(
                image, depth_squeezed, self.config.crop_size
            )

        return image, depth_squeezed

    def _rotate(
        self,
        arr: np.ndarray,
        angle: float,
        is_depth: bool = False,
    ) -> np.ndarray:
        """Rotate array by given angle.

        Args:
            arr: Input array
            angle: Rotation angle in degrees
            is_depth: Whether this is a depth map

        Returns:
            Rotated array
        """
        if not CV2_AVAILABLE:
            # Fallback: no rotation without cv2
            return arr

        h, w = arr.shape[:2]
        center = (w / 2, h / 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Use nearest neighbor for depth, bilinear for images
        interpolation = cv2.INTER_NEAREST if is_depth else cv2.INTER_LINEAR

        if arr.ndim == 3:
            rotated = cv2.warpAffine(arr, matrix, (w, h), flags=interpolation)
        else:
            rotated = cv2.warpAffine(arr, matrix, (w, h), flags=interpolation)

        return rotated

    def _scale(
        self,
        arr: np.ndarray,
        scale: float,
        is_depth: bool = False,
    ) -> np.ndarray:
        """Scale array by given factor.

        Args:
            arr: Input array
            scale: Scale factor
            is_depth: Whether this is a depth map

        Returns:
            Scaled array
        """
        if not CV2_AVAILABLE:
            return arr

        h, w = arr.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)

        interpolation = cv2.INTER_NEAREST if is_depth else cv2.INTER_LINEAR

        if arr.ndim == 3:
            scaled = cv2.resize(arr, (new_w, new_h), interpolation=interpolation)
        else:
            scaled = cv2.resize(arr, (new_w, new_h), interpolation=interpolation)

        return scaled

    def _random_crop(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        crop_size: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random crop with same parameters.

        Args:
            image: RGB image array
            depth: Depth map array
            crop_size: Target (height, width)

        Returns:
            Tuple of cropped arrays
        """
        h, w = image.shape[:2]
        crop_h, crop_w = crop_size

        # Ensure crop is not larger than image
        crop_h = min(crop_h, h)
        crop_w = min(crop_w, w)

        # Random crop position
        y = random.randint(0, h - crop_h)
        x = random.randint(0, w - crop_w)

        image_cropped = image[y:y + crop_h, x:x + crop_w]
        depth_cropped = depth[y:y + crop_h, x:x + crop_w]

        return image_cropped, depth_cropped


class ColorAugmentation:
    """Color augmentations applied only to RGB images.

    Depth maps are not affected by these transforms as they represent
    geometric information, not photometric properties.
    """

    def __init__(self, config: AugmentationConfig):
        """Initialize color augmentation.

        Args:
            config: Augmentation configuration
        """
        self.config = config

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply color augmentations to RGB image.

        Args:
            image: RGB image array (H, W, 3)

        Returns:
            Augmented image
        """
        # Convert to PIL for easier augmentation
        if PIL_AVAILABLE:
            return self._apply_pil_augmentations(image)
        else:
            return self._apply_numpy_augmentations(image)

    def _apply_pil_augmentations(self, image: np.ndarray) -> np.ndarray:
        """Apply augmentations using PIL.

        Args:
            image: RGB image array

        Returns:
            Augmented image as numpy array
        """
        # Convert to PIL Image
        if image.dtype == np.float32 or image.dtype == np.float64:
            pil_image = Image.fromarray((image * 255).astype(np.uint8))
            is_float = True
        else:
            pil_image = Image.fromarray(image)
            is_float = False

        # Random brightness
        if self.config.brightness > 0:
            factor = 1 + random.uniform(-self.config.brightness, self.config.brightness)
            enhancer = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer.enhance(factor)

        # Random contrast
        if self.config.contrast > 0:
            factor = 1 + random.uniform(-self.config.contrast, self.config.contrast)
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(factor)

        # Random saturation
        if self.config.saturation > 0:
            factor = 1 + random.uniform(-self.config.saturation, self.config.saturation)
            enhancer = ImageEnhance.Color(pil_image)
            pil_image = enhancer.enhance(factor)

        # Convert back to numpy
        result = np.array(pil_image)
        if is_float:
            result = result.astype(np.float32) / 255.0

        return result

    def _apply_numpy_augmentations(self, image: np.ndarray) -> np.ndarray:
        """Apply augmentations using numpy (fallback).

        Args:
            image: RGB image array

        Returns:
            Augmented image
        """
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0

        # Random brightness
        if self.config.brightness > 0:
            factor = 1 + random.uniform(-self.config.brightness, self.config.brightness)
            image = image * factor

        # Random contrast
        if self.config.contrast > 0:
            factor = 1 + random.uniform(-self.config.contrast, self.config.contrast)
            mean = image.mean()
            image = (image - mean) * factor + mean

        image = np.clip(image, 0, 1)
        return image


class ArchitecturalAugmentation:
    """Augmentations specific to architectural imagery.

    Includes transforms that simulate common distortions in
    architectural photography and rendering.
    """

    def __init__(self, config: AugmentationConfig):
        """Initialize architectural augmentation.

        Args:
            config: Augmentation configuration
        """
        self.config = config

    def __call__(
        self,
        image: np.ndarray,
        depth: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply architectural augmentations.

        Args:
            image: RGB image array (H, W, 3)
            depth: Depth map array (H, W)

        Returns:
            Tuple of (augmented_image, augmented_depth)
        """
        # Grid distortion
        if random.random() < self.config.grid_distortion_prob:
            image, depth = self._apply_grid_distortion(image, depth)

        # Perspective transform
        if random.random() < self.config.perspective_prob:
            image, depth = self._apply_perspective(image, depth)

        # Gaussian blur (image only)
        if random.random() < self.config.gaussian_blur_prob:
            image = self._apply_gaussian_blur(image)

        return image, depth

    def _apply_grid_distortion(
        self,
        image: np.ndarray,
        depth: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply grid-based distortion.

        Args:
            image: RGB image array
            depth: Depth map array

        Returns:
            Distorted arrays
        """
        if not CV2_AVAILABLE:
            return image, depth

        h, w = image.shape[:2]

        # Create distortion maps
        grid_size = 4
        distort_strength = 0.02

        # Create base grid
        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)

        for i in range(h):
            for j in range(w):
                # Add sinusoidal distortion
                map_x[i, j] = j + distort_strength * w * np.sin(2 * np.pi * i / (h / grid_size))
                map_y[i, j] = i + distort_strength * h * np.sin(2 * np.pi * j / (w / grid_size))

        # Apply remapping
        image_distorted = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
        depth_distorted = cv2.remap(depth, map_x, map_y, cv2.INTER_NEAREST)

        return image_distorted, depth_distorted

    def _apply_perspective(
        self,
        image: np.ndarray,
        depth: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply perspective transformation.

        Args:
            image: RGB image array
            depth: Depth map array

        Returns:
            Transformed arrays
        """
        if not CV2_AVAILABLE:
            return image, depth

        h, w = image.shape[:2]
        margin = 0.05  # 5% margin for perspective shift

        # Source points (corners)
        src_pts = np.float32([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ])

        # Destination points (with random perspective)
        dst_pts = np.float32([
            [random.uniform(0, margin * w), random.uniform(0, margin * h)],
            [random.uniform(w * (1 - margin), w), random.uniform(0, margin * h)],
            [random.uniform(w * (1 - margin), w), random.uniform(h * (1 - margin), h)],
            [random.uniform(0, margin * w), random.uniform(h * (1 - margin), h)]
        ])

        # Get perspective transform matrix
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # Apply transformation
        image_transformed = cv2.warpPerspective(image, matrix, (w, h))
        depth_transformed = cv2.warpPerspective(
            depth, matrix, (w, h), flags=cv2.INTER_NEAREST
        )

        return image_transformed, depth_transformed

    def _apply_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur to image.

        Args:
            image: RGB image array

        Returns:
            Blurred image
        """
        if PIL_AVAILABLE:
            if image.dtype == np.float32 or image.dtype == np.float64:
                pil_image = Image.fromarray((image * 255).astype(np.uint8))
                is_float = True
            else:
                pil_image = Image.fromarray(image)
                is_float = False

            radius = random.uniform(0.5, 2.0)
            pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=radius))

            result = np.array(pil_image)
            if is_float:
                result = result.astype(np.float32) / 255.0
            return result

        elif CV2_AVAILABLE:
            ksize = random.choice([3, 5, 7])
            return cv2.GaussianBlur(image, (ksize, ksize), 0)

        return image


class DepthAwareAugmentation:
    """Complete depth-aware augmentation pipeline.

    Combines geometric, color, and architectural augmentations
    while maintaining consistency between RGB and depth.

    Example:
        >>> config = AugmentationConfig(
        ...     horizontal_flip=0.5,
        ...     rotation_degrees=10,
        ...     brightness=0.2
        ... )
        >>> augment = DepthAwareAugmentation(config)
        >>> aug_image, aug_depth = augment(image, depth)
    """

    # ImageNet normalization parameters
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        config: Optional[AugmentationConfig] = None,
        is_training: bool = True,
    ):
        """Initialize depth-aware augmentation pipeline.

        Args:
            config: Augmentation configuration
            is_training: Whether this is for training (enables augmentations)
        """
        self.config = config or AugmentationConfig()
        self.is_training = is_training

        # Initialize sub-augmentations
        self.geometric = GeometricAugmentation(self.config)
        self.color = ColorAugmentation(self.config)
        self.architectural = ArchitecturalAugmentation(self.config)

    def __call__(
        self,
        image: np.ndarray,
        depth: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply augmentation pipeline.

        Args:
            image: RGB image array (H, W, 3)
            depth: Depth map array (H, W) or (H, W, 1)

        Returns:
            Tuple of (augmented_image, augmented_depth)
        """
        # Ensure correct data types
        image = self._ensure_float(image)
        depth = self._ensure_float(depth)

        if self.is_training:
            # Apply geometric augmentations (to both)
            image, depth = self.geometric(image, depth)

            # Apply color augmentations (image only)
            image = self.color(image)

            # Apply architectural augmentations
            image, depth = self.architectural(image, depth)

        # Normalize
        if self.config.normalize:
            image = self._normalize_image(image)
            depth = self._normalize_depth(depth)

        return image, depth

    def _ensure_float(self, arr: np.ndarray) -> np.ndarray:
        """Ensure array is float32 in [0, 1] range.

        Args:
            arr: Input array

        Returns:
            Float32 array in [0, 1]
        """
        arr = arr.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        return arr

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Apply ImageNet normalization.

        Args:
            image: RGB image in [0, 1]

        Returns:
            Normalized image
        """
        return (image - self.IMAGENET_MEAN) / self.IMAGENET_STD

    def _normalize_depth(self, depth: np.ndarray) -> np.ndarray:
        """Normalize depth to [0, 1] range.

        Args:
            depth: Depth map

        Returns:
            Normalized depth
        """
        depth = depth.squeeze()
        d_min = depth.min()
        d_max = depth.max()
        if d_max - d_min > 1e-8:
            depth = (depth - d_min) / (d_max - d_min)
        return depth

    def to_tensor(
        self,
        image: np.ndarray,
        depth: np.ndarray,
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Convert numpy arrays to PyTorch tensors.

        Args:
            image: RGB image array (H, W, 3)
            depth: Depth map array (H, W)

        Returns:
            Tuple of (image_tensor, depth_tensor)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for tensor conversion")

        # Image: (H, W, 3) -> (3, H, W)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1).copy())

        # Depth: (H, W) -> (1, H, W)
        if depth.ndim == 2:
            depth = depth[np.newaxis, ...]
        depth_tensor = torch.from_numpy(depth.copy())

        return image_tensor, depth_tensor


def get_train_augmentations(
    config: Optional[Dict] = None,
    image_size: Tuple[int, int] = (518, 518),
) -> DepthAwareAugmentation:
    """Get training augmentation pipeline.

    Args:
        config: Configuration dictionary with augmentation parameters
        image_size: Target image size (height, width)

    Returns:
        Configured augmentation pipeline
    """
    config = config or {}

    aug_config = AugmentationConfig(
        horizontal_flip=config.get("horizontal_flip", 0.5),
        rotation_degrees=config.get("rotation_degrees", 10),
        brightness=config.get("brightness", 0.2),
        contrast=config.get("contrast", 0.2),
        saturation=config.get("saturation", 0.2),
        hue=config.get("hue", 0.1),
        scale_range=tuple(config.get("scale_range", [0.8, 1.2])),
        crop_size=image_size,
        grid_distortion_prob=config.get("grid_distortion_prob", 0.0),
        perspective_prob=config.get("perspective_prob", 0.0),
        gaussian_blur_prob=config.get("gaussian_blur_prob", 0.1),
        normalize=True,
    )

    return DepthAwareAugmentation(aug_config, is_training=True)


def get_val_augmentations(
    image_size: Tuple[int, int] = (518, 518),
) -> DepthAwareAugmentation:
    """Get validation augmentation pipeline (no augmentations, just normalize).

    Args:
        image_size: Target image size (height, width)

    Returns:
        Validation augmentation pipeline
    """
    aug_config = AugmentationConfig(
        horizontal_flip=0.0,
        rotation_degrees=0.0,
        brightness=0.0,
        contrast=0.0,
        saturation=0.0,
        hue=0.0,
        scale_range=(1.0, 1.0),
        crop_size=image_size,
        normalize=True,
    )

    return DepthAwareAugmentation(aug_config, is_training=False)
