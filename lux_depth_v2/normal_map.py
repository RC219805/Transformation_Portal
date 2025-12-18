#!/usr/bin/env python3
"""
Corrected Normal Map Computation from Depth
============================================

Fixes the "fundamentally wrong" normal map issue identified in user feedback.

The Problem:
- Current implementation likely uses wrong Z scale, forcing normals to point 
  almost straight at camera (uniform purple/blue)
- Not usable for PBR/relighting

The Fix:
- Compute normals from normalized depth [0,1]
- Use sane Z scale based on field of view
- Optional smoothing to reduce ringing
- Proper tangent-space encoding for PBR pipelines

Reference: User feedback 2025-12-17 - "Your normal map is fundamentally wrong"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class NormalMapConfig:
    """Configuration for normal map generation."""
    
    # Z scale (controls normal "steepness")
    # Lower = more pronounced surface variation
    # Higher = flatter normals (more camera-facing)
    z_scale: float = 1.0  # Reasonable default for architectural scenes
    
    # Gradient computation
    gradient_method: str = "scharr"  # scharr | sobel | central_diff
    gradient_smooth_sigma: float = 0.5  # Light smoothing to reduce ringing
    
    # Normal smoothing (optional)
    smooth_normals: bool = False
    smooth_sigma: float = 1.0
    
    # Output format
    tangent_space: bool = True  # True = tangent space (for PBR), False = world space
    normalize: bool = True  # Ensure unit-length normals


class NormalMapGenerator:
    """
    Generate correct, usable normal maps from depth.
    
    Key fixes:
    1. Normalize depth to [0, 1] before gradient computation
    2. Use sane Z scale (not excessive constant)
    3. Proper gradient computation with optional smoothing
    4. Output in tangent space for PBR compatibility
    """
    
    def __init__(self, config: NormalMapConfig):
        self.config = config
        logger.info(f"NormalMapGenerator initialized: z_scale={config.z_scale} method={config.gradient_method}")
    
    def _compute_gradients(self, depth_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute depth gradients (dz/dx, dz/dy) using specified method.
        
        Args:
            depth_norm: Normalized depth [0, 1] as float32
            
        Returns:
            (dzdx, dzdy) - depth gradients
        """
        if not CV2_AVAILABLE:
            # Fallback: simple central difference
            dzdx = np.zeros_like(depth_norm)
            dzdy = np.zeros_like(depth_norm)
            
            dzdx[:, 1:-1] = (depth_norm[:, 2:] - depth_norm[:, :-2]) / 2.0
            dzdx[:, 0] = depth_norm[:, 1] - depth_norm[:, 0]
            dzdx[:, -1] = depth_norm[:, -1] - depth_norm[:, -2]
            
            dzdy[1:-1, :] = (depth_norm[2:, :] - depth_norm[:-2, :]) / 2.0
            dzdy[0, :] = depth_norm[1, :] - depth_norm[0, :]
            dzdy[-1, :] = depth_norm[-1, :] - depth_norm[-2, :]
            
            return dzdx, dzdy
        
        # Use OpenCV for better gradient computation
        if self.config.gradient_method == "scharr":
            # Scharr filter (better rotational symmetry than Sobel)
            dzdx = cv2.Scharr(depth_norm, cv2.CV_32F, 1, 0)
            dzdy = cv2.Scharr(depth_norm, cv2.CV_32F, 0, 1)
        elif self.config.gradient_method == "sobel":
            # Standard Sobel
            dzdx = cv2.Sobel(depth_norm, cv2.CV_32F, 1, 0, ksize=3)
            dzdy = cv2.Sobel(depth_norm, cv2.CV_32F, 0, 1, ksize=3)
        else:  # central_diff
            # Simple central difference
            dzdx = np.zeros_like(depth_norm)
            dzdy = np.zeros_like(depth_norm)
            
            dzdx[:, 1:-1] = (depth_norm[:, 2:] - depth_norm[:, :-2]) / 2.0
            dzdx[:, 0] = depth_norm[:, 1] - depth_norm[:, 0]
            dzdx[:, -1] = depth_norm[:, -1] - depth_norm[:, -2]
            
            dzdy[1:-1, :] = (depth_norm[2:, :] - depth_norm[:-2, :]) / 2.0
            dzdy[0, :] = depth_norm[1, :] - depth_norm[0, :]
            dzdy[-1, :] = depth_norm[-1, :] - depth_norm[-2, :]
        
        # Optional: light smoothing to reduce ringing
        sigma = self.config.gradient_smooth_sigma
        if sigma > 0 and CV2_AVAILABLE:
            ksize = max(3, int(sigma * 6) | 1)  # Ensure odd
            dzdx = cv2.GaussianBlur(dzdx, (ksize, ksize), sigma)
            dzdy = cv2.GaussianBlur(dzdy, (ksize, ksize), sigma)
        
        return dzdx, dzdy
    
    def generate(
        self, 
        depth: np.ndarray,
        strength: float = 1.0
    ) -> np.ndarray:
        """
        Generate normal map from depth.
        
        Args:
            depth: Depth map as float32 [0, 1] or uint16
            strength: Multiplier for gradient strength (default 1.0)
            
        Returns:
            Normal map as float32 RGB [0, 1] in tangent space
            (R=X+, G=Y+, B=Z+ where Z points toward camera)
        """
        # Normalize depth to [0, 1]
        if depth.dtype == np.uint16:
            depth_norm = depth.astype(np.float32) / 65535.0
        elif depth.dtype == np.uint8:
            depth_norm = depth.astype(np.float32) / 255.0
        else:
            depth_norm = depth.astype(np.float32)
            d_min, d_max = depth_norm.min(), depth_norm.max()
            if d_max > d_min:
                depth_norm = (depth_norm - d_min) / (d_max - d_min)
        
        # Compute gradients
        dzdx, dzdy = self._compute_gradients(depth_norm)
        
        # Scale gradients by strength and z_scale
        # Negative signs: depth increases away from camera, but we want normals
        # to point based on surface orientation
        dzdx = -dzdx * strength
        dzdy = -dzdy * strength
        
        # Construct normal vectors: n = [-dzdx, -dzdy, z_scale]
        # The z_scale controls how "steep" the normals are
        # Higher z_scale = flatter normals (more camera-facing)
        # Lower z_scale = steeper normals (more surface variation)
        h, w = depth_norm.shape
        normals = np.zeros((h, w, 3), dtype=np.float32)
        normals[:, :, 0] = dzdx  # X component (red)
        normals[:, :, 1] = dzdy  # Y component (green)
        normals[:, :, 2] = self.config.z_scale  # Z component (blue)
        
        # Normalize to unit length
        if self.config.normalize:
            norm = np.sqrt(
                normals[:, :, 0]**2 + 
                normals[:, :, 1]**2 + 
                normals[:, :, 2]**2
            )
            norm = np.maximum(norm, 1e-8)  # Avoid division by zero
            normals = normals / norm[:, :, None]
        
        # Optional: smooth normal vectors (reduces noise)
        if self.config.smooth_normals and CV2_AVAILABLE:
            sigma = self.config.smooth_sigma
            ksize = max(3, int(sigma * 6) | 1)
            for c in range(3):
                normals[:, :, c] = cv2.GaussianBlur(normals[:, :, c], (ksize, ksize), sigma)
            
            # Re-normalize after smoothing
            if self.config.normalize:
                norm = np.sqrt(normals[:, :, 0]**2 + normals[:, :, 1]**2 + normals[:, :, 2]**2)
                norm = np.maximum(norm, 1e-8)
                normals = normals / norm[:, :, None]
        
        # Convert from [-1, 1] to [0, 1] for image output (tangent space)
        if self.config.tangent_space:
            normals = (normals + 1.0) / 2.0
            normals = np.clip(normals, 0.0, 1.0)
        
        logger.info(f"✓ Normal map generated: {normals.shape}, range=[{normals.min():.3f}, {normals.max():.3f}]")
        return normals
    
    def validate_normal_map(self, normals: np.ndarray) -> Dict[str, float]:
        """
        Validate normal map quality.
        
        A good normal map should:
        - Have variation in all 3 channels (not uniform purple/blue)
        - Have reasonable distribution of directions
        - Not be too flat or too steep
        """
        # Convert back to [-1, 1] if in tangent space
        if self.config.tangent_space:
            n = normals * 2.0 - 1.0
        else:
            n = normals
        
        # Compute statistics
        nx_std = n[:, :, 0].std()
        ny_std = n[:, :, 1].std()
        nz_mean = n[:, :, 2].mean()
        
        # Angle distribution (how much surface varies from camera-facing)
        # Angle from +Z axis
        angles_deg = np.arccos(np.clip(n[:, :, 2], -1, 1)) * 180 / np.pi
        angle_median = np.median(angles_deg)
        angle_std = angles_deg.std()
        
        metrics = {
            "nx_std": float(nx_std),
            "ny_std": float(ny_std),
            "nz_mean": float(nz_mean),
            "angle_median_deg": float(angle_median),
            "angle_std_deg": float(angle_std),
        }
        
        logger.info(f"Normal map validation: X_std={nx_std:.3f} Y_std={ny_std:.3f} Z_mean={nz_mean:.3f} angle_med={angle_median:.1f}°")
        
        # Quality check
        if nx_std < 0.05 and ny_std < 0.05:
            logger.warning("⚠️  Normal map is too flat (low X/Y variation) - may indicate wrong Z scale")
        if nz_mean > 0.95:
            logger.warning("⚠️  Normals mostly point at camera (high Z) - may indicate wrong Z scale")
        if angle_median < 10:
            logger.warning("⚠️  Surface too flat (low angles) - increase gradient strength or decrease z_scale")
        
        return metrics


def create_normal_map_generator(
    z_scale: float = 1.0,
    gradient_method: str = "scharr",
    smooth_sigma: float = 0.5
) -> NormalMapGenerator:
    """Convenience factory for normal map generator."""
    config = NormalMapConfig(
        z_scale=z_scale,
        gradient_method=gradient_method,
        gradient_smooth_sigma=smooth_sigma
    )
    return NormalMapGenerator(config)


# Preset configurations for common use cases

PRESETS = {
    "architectural": NormalMapConfig(
        z_scale=1.0,
        gradient_method="scharr",
        gradient_smooth_sigma=0.5,
        smooth_normals=False,
        tangent_space=True
    ),
    "subtle": NormalMapConfig(
        z_scale=2.0,  # Flatter normals
        gradient_method="scharr",
        gradient_smooth_sigma=1.0,
        smooth_normals=True,
        smooth_sigma=1.5,
        tangent_space=True
    ),
    "pronounced": NormalMapConfig(
        z_scale=0.5,  # Steeper normals
        gradient_method="scharr",
        gradient_smooth_sigma=0.3,
        smooth_normals=False,
        tangent_space=True
    ),
}


def generate_normal_map(
    depth: np.ndarray,
    preset: str = "architectural",
    strength: float = 1.0
) -> np.ndarray:
    """
    One-shot normal map generation with presets.
    
    Args:
        depth: Depth map as float32 [0,1] or uint16
        preset: "architectural" | "subtle" | "pronounced"
        strength: Multiplier for gradient strength
        
    Returns:
        Normal map as float32 RGB [0, 1]
    """
    config = PRESETS.get(preset, PRESETS["architectural"])
    generator = NormalMapGenerator(config)
    return generator.generate(depth, strength=strength)
