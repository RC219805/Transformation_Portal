"""Heuristic CPU fallback for PBR texture generation (Phase 2.2).

When GPU is unavailable or NVDIFFREC/MaterialGAN aren't installed,
this module provides CPU-based PBR texture generation using classical
image processing techniques.

Architecture:
- No ML dependencies (CPU-only)
- Fast approximations using edge detection, gradients, and heuristics
- Quality trade-off: lower fidelity than neural methods but functional
- Useful for testing, CI/CD, and low-power environments

Performance:
- 1024x1024: ~2-5 seconds on CPU
- Memory: <500MB
"""

from typing import Optional

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage

# Pixels evaluated per block by _local_variance: bounds the (block, size*size) float64 window copy.
_LOCAL_VARIANCE_BLOCK_PIXELS = 1 << 18


def _local_variance(values: np.ndarray, size: int = 5) -> np.ndarray:
    """Return ``ndimage.generic_filter(values, np.var, size=size)`` without a Python call per pixel.

    generic_filter hands each reflect-padded ``size x size`` neighbourhood to ``np.var`` as a
    float64 buffer and casts the result to the input dtype. This evaluates the same float64
    ``np.var`` reduction on row blocks of sliding windows (scipy's ``reflect`` mode is numpy's
    ``symmetric`` padding), so the result is bit-identical; non-float or non-2-D inputs keep
    the generic_filter path.
    """
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.floating):
        return ndimage.generic_filter(values, np.var, size=size)

    radius = size // 2
    padded = np.pad(values.astype(np.float64), radius, mode="symmetric")
    height, width = values.shape
    variance = np.empty(values.shape, dtype=values.dtype)
    rows_per_block = max(1, _LOCAL_VARIANCE_BLOCK_PIXELS // max(1, width))
    for start in range(0, height, rows_per_block):
        stop = min(height, start + rows_per_block)
        windows = sliding_window_view(padded[start : stop + 2 * radius], (size, size))
        variance[start:stop] = np.var(windows.reshape(stop - start, width, size * size), axis=-1)
    return variance


class HeuristicFallback:
    """CPU-based PBR texture generator using classical image processing.

    Methods:
    - Albedo: Desaturated input (remove shadows via bilateral filtering)
    - Normal: Sobel edge detection + gradient-to-normal conversion
    - Roughness: Variance-based texture analysis
    - Metallic: Specular detection via saturation/brightness analysis
    - AO: Approximated from depth or luminance gradients
    """

    def __init__(self):
        """Initialize heuristic generator."""
        pass

    def generate_pbr_textures(
        self,
        rgb: np.ndarray,
        mask: Optional[np.ndarray] = None,
        depth: Optional[np.ndarray] = None,
        material_hint: Optional[str] = None,
        normal_strength: float = 1.0,
        ao_intensity: float = 0.7,
    ) -> tuple:
        """Generate PBR textures using heuristic methods.

        Args:
            rgb: Linear RGB image (H, W, 3) float32.
            mask: Optional mask (H, W) bool.
            depth: Optional depth map (H, W) float32.
            material_hint: Optional material category.
            normal_strength: Normal intensity multiplier [0, 2].
            ao_intensity: AO darkness multiplier [0, 1].

        Returns:
            Tuple of (albedo, normal, roughness, metallic, ao, height) as np.ndarray.
        """
        H, W = rgb.shape[:2]

        # If mask provided, only process masked region
        active_mask = mask if mask is not None else np.ones((H, W), dtype=bool)

        # 1. Generate Albedo (remove shadows/lighting)
        albedo = self._generate_albedo(rgb, active_mask)

        # 2. Generate Normal Map
        normal = self._generate_normal(rgb, depth, normal_strength, active_mask)

        # 3. Generate Roughness
        roughness = self._generate_roughness(rgb, material_hint, active_mask)

        # 4. Generate Metallic
        metallic = self._generate_metallic(rgb, material_hint, active_mask)

        # 5. Generate Ambient Occlusion
        ao = self._generate_ao(rgb, depth, ao_intensity, active_mask)

        # 6. Generate Height (optional, from depth or luminance)
        height = self._generate_height(rgb, depth, active_mask)

        return albedo, normal, roughness, metallic, ao, height

    def _generate_albedo(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Generate albedo by removing shadows/lighting.

        Uses bilateral filtering to smooth out lighting variations while
        preserving color transitions.
        """
        # Clip to [0, 1] for stability
        rgb_clipped = np.clip(rgb, 0, 1)

        # Filter in float32: quantising linear RGB to 8 bits would flatten the
        # shadows (every value below 1/255 became 0) before the filter runs.
        try:
            import cv2

            # Bilateral filter: smooth lighting while preserving edges
            # d=9: diameter of pixel neighborhood
            # sigmaColor=75/255: color space sigma (75 in 8-bit units, expressed on the
            #   [0, 1] float scale; larger = more colors mixed)
            # sigmaSpace=75: coordinate space sigma (larger = farther pixels mixed)
            albedo = cv2.bilateralFilter(rgb_clipped.astype(np.float32), d=9, sigmaColor=75.0 / 255.0, sigmaSpace=75)

        except ImportError:
            # Fallback to Gaussian if cv2 not available
            albedo = rgb_clipped.copy()
            for c in range(3):
                albedo[:, :, c] = ndimage.gaussian_filter(albedo[:, :, c], sigma=1.0)

        # Apply mask
        albedo[~mask] = 0.0

        return albedo.astype(np.float32)

    def _generate_normal(self, rgb: np.ndarray, depth: Optional[np.ndarray], strength: float, mask: np.ndarray) -> np.ndarray:
        """Generate normal map from gradients.

        Uses Sobel edge detection to compute gradients, then converts
        to tangent-space normals. Prioritizes depth map when available
        for geometry-aware normal estimation.
        """
        H, W = rgb.shape[:2]

        # Use depth if available (much better quality), otherwise use luminance
        if depth is not None:
            # Smooth depth slightly to reduce noise
            surface = ndimage.gaussian_filter(depth, sigma=0.5)

            # Use central differences for better gradient accuracy
            # Scale factor for depth (meters) to normal space
            scale_factor = 5.0  # Adjustable for different depth ranges
            dx = ndimage.sobel(surface, axis=1) * strength * scale_factor
            dy = ndimage.sobel(surface, axis=0) * strength * scale_factor
        else:
            # Convert RGB to luminance (fallback)
            surface = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]

            # Compute gradients using Sobel
            dx = ndimage.sobel(surface, axis=1) * strength
            dy = ndimage.sobel(surface, axis=0) * strength

        # Convert gradients to normal map
        # Normal = (-dx, -dy, 1) normalized
        normal = np.zeros((H, W, 3), dtype=np.float32)
        normal[:, :, 0] = -dx
        normal[:, :, 1] = -dy
        normal[:, :, 2] = 1.0

        # Normalize to unit vectors
        norm = np.sqrt(np.sum(normal**2, axis=2, keepdims=True))
        norm = np.maximum(norm, 1e-6)  # Avoid division by zero
        normal = normal / norm

        # Apply mask
        normal[~mask] = [0, 0, 1]  # Default normal pointing up

        return normal

    def _generate_roughness(self, rgb: np.ndarray, material_hint: Optional[str], mask: np.ndarray) -> np.ndarray:
        """Generate roughness map from texture variance.

        High-variance regions = rough surfaces.
        Low-variance regions = smooth surfaces.
        Material hints provide physically-accurate baseline values.
        """
        H, W = rgb.shape[:2]

        # Convert to luminance
        luminance = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]

        # Compute local variance (texture indicator)
        # Use a sliding window approach
        variance = _local_variance(luminance, size=5)

        # Normalize to [0, 1]
        if np.max(variance) > 1e-6:
            variance = variance / np.max(variance)
        else:
            variance = np.zeros_like(variance)

        # Material-specific adjustments with physically-accurate values
        MATERIAL_ROUGHNESS = {
            "metal": (0.1, 0.3),  # Polished metal range
            "wood": (0.6, 0.8),  # Natural wood
            "fabric": (0.7, 0.9),  # Textile surfaces
            "concrete": (0.8, 0.95),  # Rough concrete
            "glass": (0.02, 0.1),  # Glass/ceramic
            "stone": (0.5, 0.75),  # Natural stone
            "leather": (0.4, 0.6),  # Leather surfaces
            "ceramic": (0.2, 0.4),  # Glazed ceramic
        }

        if material_hint in MATERIAL_ROUGHNESS:
            base, var_scale = MATERIAL_ROUGHNESS[material_hint]
            roughness = base + variance * (var_scale - base)
        else:
            # Default: moderate roughness with variance modulation
            roughness = 0.5 + variance * 0.3

        roughness = np.clip(roughness, 0, 1)

        # Apply mask
        roughness[~mask] = 0.5  # Default neutral roughness

        return roughness.astype(np.float32)

    def _generate_metallic(self, rgb: np.ndarray, material_hint: Optional[str], mask: np.ndarray) -> np.ndarray:
        """Generate metallic map from specular analysis.

        Metals have:
        - Low saturation (grayscale-ish)
        - High brightness in highlights
        - Reflective appearance
        """
        H, W = rgb.shape[:2]

        # Material-specific metallic values
        if material_hint == "metal":
            # High metallic
            metallic = np.ones((H, W), dtype=np.float32) * 0.9
        elif material_hint in ["wood", "fabric", "concrete", "ceramic", "leather"]:
            # Non-metallic
            metallic = np.zeros((H, W), dtype=np.float32)
        elif material_hint == "glass":
            # Glass is dielectric but highly reflective
            metallic = np.zeros((H, W), dtype=np.float32)
        else:
            # Heuristic: detect metallic regions via saturation
            # Metals have low saturation relative to brightness
            max_channel = np.max(rgb, axis=2)
            min_channel = np.min(rgb, axis=2)
            saturation = (max_channel - min_channel) / (max_channel + 1e-6)

            # Low saturation + high brightness = metallic
            brightness = np.mean(rgb, axis=2)
            metallic = (1.0 - saturation) * brightness
            metallic = np.clip(metallic, 0, 1)

        # Apply mask
        metallic[~mask] = 0.0

        return metallic.astype(np.float32)

    def _generate_ao(self, rgb: np.ndarray, depth: Optional[np.ndarray], intensity: float, mask: np.ndarray) -> np.ndarray:
        """Generate ambient occlusion approximation.

        AO darkens crevices and corners. We approximate this using:
        - Depth gradients + concavity detection (if depth available)
        - Luminance valleys (otherwise)
        """
        H, W = rgb.shape[:2]

        if depth is not None:
            # Enhanced depth-based AO with concavity detection
            # Smooth depth to reduce noise
            depth_smooth = ndimage.gaussian_filter(depth, sigma=1.0)

            # Compute second derivatives (concavity/convexity)
            # Concave areas (crevices) should be darker
            laplacian = ndimage.laplace(depth_smooth)

            # Normalize laplacian to [0, 1] where positive = concave (darker)
            concavity = np.clip(laplacian, 0, None)  # Only keep concave regions
            if np.max(concavity) > 1e-6:
                concavity = concavity / np.max(concavity)

            # Also compute local depth variance (corners/edges)
            depth_variance = _local_variance(depth_smooth, size=5)
            if np.max(depth_variance) > 1e-6:
                depth_variance = depth_variance / np.max(depth_variance)

            # Combine concavity and variance for AO
            # Concavity contributes 70%, variance 30%
            occlusion = (concavity * 0.7 + depth_variance * 0.3) * intensity

            # AO = 1 - occlusion (AO=1 means fully lit)
            ao = 1.0 - np.clip(occlusion, 0, 1)
        else:
            # Luminance-based fallback
            luminance = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]

            # Dark regions are more likely to be occluded
            # But be conservative to avoid over-darkening
            ao = 1.0 - (1.0 - luminance) * intensity * 0.3

        ao = np.clip(ao, 0, 1)

        # Apply mask
        ao[~mask] = 1.0  # Fully lit outside mask

        return ao.astype(np.float32)

    def _generate_height(self, rgb: np.ndarray, depth: Optional[np.ndarray], mask: np.ndarray) -> np.ndarray:
        """Generate height/displacement map.

        Uses depth if available, otherwise luminance as proxy.
        """
        H, W = rgb.shape[:2]

        if depth is not None:
            # Normalize depth to [0, 1]
            height = depth - np.min(depth)
            height = height / (np.max(height) + 1e-6)
        else:
            # Use luminance as height proxy
            luminance = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
            height = luminance

        height = np.clip(height, 0, 1)

        # Apply mask
        height[~mask] = 0.5  # Neutral height outside mask

        return height.astype(np.float32)
