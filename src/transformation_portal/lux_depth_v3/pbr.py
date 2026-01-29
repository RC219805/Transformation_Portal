"""PBR Map Generation for Lux Depth V3.

Generates Physically Based Rendering maps from depth data:
- Normal maps (RGB-encoded surface normals)
- Roughness maps (surface micro-detail)
- Ambient Occlusion maps (indirect lighting approximation)

All operations use NumPy/SciPy/Pillow only - no OpenCV dependency.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class PBRConfig:
    """Configuration for PBR map generation.

    All parameters are frozen to ensure immutability and cache-ability.
    """
    # Normal map parameters
    normal_strength: float = 1.0  # Gradient multiplier (higher = more pronounced)
    normal_blur_radius: int = 0   # Pre-blur depth before gradient (0 = disabled)

    # Roughness map parameters
    roughness_strength: float = 1.0  # Detail multiplier
    roughness_blur_radius: int = 3   # Smoothing kernel size

    # Ambient Occlusion parameters
    ao_strength: float = 1.0         # Darkness multiplier
    ao_blur_radius: int = 5          # Occlusion spread
    ao_bias: float = 0.5             # Brightness offset (0.0-1.0)


def _box_blur_gray(img: np.ndarray, radius: int) -> np.ndarray:
    """Fast box blur using uniform filter.

    CRITICAL: Correctly handles padding to prevent shape shrinking.

    Args:
        img: 2D grayscale array (H, W)
        radius: Blur radius in pixels

    Returns:
        Blurred image with SAME shape as input
    """
    if radius <= 0:
        return img.copy()

    # Use scipy's uniform_filter for box blur (mean filter)
    # kernel_size = 2 * radius + 1
    kernel_size = 2 * radius + 1
    return ndimage.uniform_filter(img, size=kernel_size, mode='reflect')


def _sobel(depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Sobel gradients (dx, dy) from depth map.

    Args:
        depth: 2D depth array (H, W), normalized 0-1

    Returns:
        (grad_x, grad_y) both with same shape as input
    """
    # Sobel kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    grad_x = ndimage.convolve(depth, sobel_x, mode='reflect')
    grad_y = ndimage.convolve(depth, sobel_y, mode='reflect')

    return grad_x, grad_y


def _laplacian(depth: np.ndarray) -> np.ndarray:
    """Compute Laplacian (second derivative) for roughness/AO detection.

    Args:
        depth: 2D depth array (H, W), normalized 0-1

    Returns:
        Laplacian response, same shape as input
    """
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    return ndimage.convolve(depth, laplacian_kernel, mode='reflect')


def generate_pbr_maps(
    depth: np.ndarray,
    config: PBRConfig = PBRConfig()
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate PBR maps from depth data.

    Args:
        depth: 2D depth array (H, W), values in 0-1 range
        config: PBR generation parameters

    Returns:
        Tuple of (normal_map, roughness_map, ao_map):
            - normal_map: RGB uint8 (H, W, 3), tangent-space normals
            - roughness_map: Grayscale uint8 (H, W)
            - ao_map: Grayscale uint8 (H, W)

    Example:
        >>> depth = np.random.rand(512, 512)
        >>> normal, roughness, ao = generate_pbr_maps(depth)
        >>> assert normal.shape == (512, 512, 3)
        >>> assert roughness.shape == (512, 512)
    """
    h, w = depth.shape

    # 1. NORMAL MAP
    # Pre-blur depth if requested
    depth_for_normals = _box_blur_gray(depth, config.normal_blur_radius) if config.normal_blur_radius > 0 else depth

    # Compute gradients
    grad_x, grad_y = _sobel(depth_for_normals)

    # Scale by strength
    grad_x *= config.normal_strength
    grad_y *= config.normal_strength

    # Build normal vectors: N = (-dx, -dy, 1)
    normals = np.stack([
        -grad_x,   # X component
        -grad_y,   # Y component
        np.ones_like(grad_x)  # Z component (up)
    ], axis=-1)

    # Normalize to unit length
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-8)  # Avoid division by zero
    normals = normals / norm

    # Map from [-1, 1] to [0, 255]
    normal_map = ((normals + 1.0) * 127.5).astype(np.uint8)

    # 2. ROUGHNESS MAP
    # Compute surface detail via Laplacian
    detail = np.abs(_laplacian(depth))

    # Scale and blur
    detail *= config.roughness_strength
    roughness = _box_blur_gray(detail, config.roughness_blur_radius)

    # Normalize to 0-1
    if roughness.max() > roughness.min():
        roughness = (roughness - roughness.min()) / (roughness.max() - roughness.min())

    roughness_map = (roughness * 255).astype(np.uint8)

    # 3. AMBIENT OCCLUSION MAP
    # Approximate occlusion from depth gradients
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # Blur to spread occlusion
    occlusion = _box_blur_gray(grad_mag, config.ao_blur_radius)

    # Scale and apply bias
    occlusion *= config.ao_strength
    if occlusion.max() > occlusion.min():
        occlusion = (occlusion - occlusion.min()) / (occlusion.max() - occlusion.min())

    # Apply bias (darker = more occluded, so invert and apply bias)
    ao = 1.0 - occlusion
    ao = np.clip(ao * (1.0 - config.ao_bias) + config.ao_bias, 0.0, 1.0)

    ao_map = (ao * 255).astype(np.uint8)

    return normal_map, roughness_map, ao_map
