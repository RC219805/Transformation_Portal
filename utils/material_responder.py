#!/usr/bin/env python3
"""
Material Response Integration Module
=====================================

Wrapper for material-aware surface enhancement, providing physics-based
enhancements that respect material properties.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class SurfaceType(Enum):
    """Supported surface/material types."""
    WOOD = "wood"
    METAL = "metal"
    GLASS = "glass"
    STONE = "stone"
    FABRIC = "fabric"
    CONCRETE = "concrete"
    CERAMIC = "ceramic"
    WATER = "water"


@dataclass
class MaterialResponseConfig:
    """Configuration for material response system."""
    strength: float = 0.75  # Global enhancement strength (0-1)
    surface_types: List[str] = None  # Surfaces to enhance (None = all)
    depth_aware: bool = True  # Use depth map if available
    preserve_highlights: bool = True  # Preserve specular highlights
    device: str = "auto"
    
    def __post_init__(self):
        """Set defaults."""
        if self.surface_types is None:
            self.surface_types = ["wood", "metal", "glass", "stone"]


class MaterialResponder:
    """
    Material-aware surface enhancer.
    
    Applies physics-based enhancements that respect material properties:
    - Wood: Grain enhancement, warmth
    - Metal: Specular highlights, reflectivity
    - Glass: Transparency, clarity
    - Stone: Texture, depth
    """
    
    def __init__(self, config: MaterialResponseConfig):
        self.config = config
        self._load_material_profiles()
        logger.info(f"Material responder initialized (strength: {config.strength})")
    
    def _load_material_profiles(self):
        """Load material enhancement profiles."""
        self.profiles = {
            SurfaceType.WOOD: {
                'saturation_boost': 1.15,
                'warmth_shift': 0.02,  # Slight red/yellow shift
                'texture_emphasis': 0.3,
                'hue_range': (10, 40),  # Brown tones
            },
            SurfaceType.METAL: {
                'contrast_boost': 1.25,
                'highlight_protection': 0.9,
                'specular_enhance': 0.4,
                'saturation_reduce': 0.9,  # Reduce color in metals
            },
            SurfaceType.GLASS: {
                'clarity_boost': 0.2,
                'highlight_protection': 0.95,
                'edge_enhance': 0.15,
                'transparency_preserve': True,
            },
            SurfaceType.STONE: {
                'texture_emphasis': 0.4,
                'saturation_reduce': 0.95,
                'detail_enhance': 0.3,
                'hue_range': (0, 60),  # Grays to warm tones
            },
            SurfaceType.FABRIC: {
                'saturation_boost': 1.1,
                'texture_emphasis': 0.25,
                'softness_preserve': 0.8,
            },
            SurfaceType.CONCRETE: {
                'texture_emphasis': 0.2,
                'saturation_reduce': 0.9,
                'detail_enhance': 0.2,
            },
            SurfaceType.CERAMIC: {
                'highlight_protection': 0.95,
                'saturation_boost': 1.05,
                'smoothness': 0.9,
            },
            SurfaceType.WATER: {
                'saturation_boost': 1.2,
                'clarity_boost': 0.3,
                'hue_range': (180, 240),  # Blue tones
                'reflection_enhance': 0.3,
            },
        }
    
    def detect_materials(self, image: np.ndarray) -> Dict[SurfaceType, np.ndarray]:
        """
        Detect materials in image (simplified heuristic-based).
        
        Args:
            image: RGB image as float32 [0, 1]
            
        Returns:
            Dict mapping SurfaceType to confidence map [0, 1]
        """
        h, w = image.shape[:2]
        material_maps = {}
        
        # Convert to HSV for analysis
        from colorsys import rgb_to_hsv
        hsv_image = np.zeros_like(image)
        for i in range(h):
            for j in range(w):
                r, g, b = image[i, j]
                hsv_image[i, j] = rgb_to_hsv(r, g, b)
        
        hue = hsv_image[:, :, 0] * 360  # Convert to degrees
        sat = hsv_image[:, :, 1]
        val = hsv_image[:, :, 2]
        
        # Wood detection: Brown hues, moderate saturation
        wood_mask = np.zeros((h, w), dtype=np.float32)
        wood_hue = (hue >= 10) & (hue <= 40)
        wood_sat = (sat >= 0.2) & (sat <= 0.8)
        wood_val = (val >= 0.2) & (val <= 0.7)
        wood_mask = (wood_hue & wood_sat & wood_val).astype(np.float32)
        material_maps[SurfaceType.WOOD] = wood_mask
        
        # Metal detection: Low saturation, high value, high contrast
        metal_mask = np.zeros((h, w), dtype=np.float32)
        metal_sat = sat < 0.2
        metal_val = val > 0.4
        metal_mask = (metal_sat & metal_val).astype(np.float32)
        material_maps[SurfaceType.METAL] = metal_mask
        
        # Glass detection: Very low saturation, high value
        glass_mask = np.zeros((h, w), dtype=np.float32)
        glass_sat = sat < 0.3
        glass_val = val > 0.5
        glass_mask = (glass_sat & glass_val).astype(np.float32) * 0.8
        material_maps[SurfaceType.GLASS] = glass_mask
        
        # Stone detection: Low saturation, varied value
        stone_mask = np.zeros((h, w), dtype=np.float32)
        stone_sat = sat < 0.3
        stone_mask = stone_sat.astype(np.float32) * 0.6
        material_maps[SurfaceType.STONE] = stone_mask
        
        # Smooth masks
        from scipy.ndimage import gaussian_filter
        for surface_type in material_maps:
            material_maps[surface_type] = gaussian_filter(
                material_maps[surface_type], sigma=3
            )
        
        return material_maps
    
    def enhance_surface(
        self,
        image: np.ndarray,
        surface_type: SurfaceType,
        confidence_map: np.ndarray,
        depth_map: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply material-specific enhancement.
        
        Args:
            image: RGB image as float32 [0, 1]
            surface_type: Type of surface to enhance
            confidence_map: Per-pixel confidence [0, 1]
            depth_map: Optional depth map for depth-aware enhancement
            
        Returns:
            Enhanced image as float32 [0, 1]
        """
        if surface_type not in self.profiles:
            logger.warning(f"No profile for {surface_type.value}")
            return image
        
        profile = self.profiles[surface_type]
        enhanced = image.copy()
        
        # Apply strength modulation
        effective_confidence = confidence_map * self.config.strength
        
        # Depth-aware modulation (enhance foreground more)
        if self.config.depth_aware and depth_map is not None:
            # Depth: 0=far, 1=near
            # Boost enhancement for near surfaces
            depth_boost = 0.5 + 0.5 * depth_map
            effective_confidence = effective_confidence * depth_boost
        
        # Saturation adjustment
        if 'saturation_boost' in profile:
            boost = profile['saturation_boost']
            from colorsys import rgb_to_hsv, hsv_to_rgb
            
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    if effective_confidence[i, j] > 0.1:
                        r, g, b = enhanced[i, j]
                        h, s, v = rgb_to_hsv(r, g, b)
                        
                        # Adjust saturation
                        factor = 1.0 + (boost - 1.0) * effective_confidence[i, j]
                        s = np.clip(s * factor, 0, 1)
                        
                        enhanced[i, j] = hsv_to_rgb(h, s, v)
        
        # Warmth shift (for wood)
        if 'warmth_shift' in profile:
            shift = profile['warmth_shift']
            warmth = np.array([shift, shift * 0.5, -shift * 0.3])
            enhanced = enhanced + warmth * effective_confidence[:, :, None]
        
        # Contrast boost (for metal)
        if 'contrast_boost' in profile:
            boost = profile['contrast_boost']
            mean_val = enhanced.mean(axis=-1, keepdims=True)
            contrast_enhanced = mean_val + (enhanced - mean_val) * boost
            enhanced = enhanced * (1 - effective_confidence[:, :, None]) + \
                      contrast_enhanced * effective_confidence[:, :, None]
        
        # Texture emphasis (simple unsharp mask)
        if 'texture_emphasis' in profile:
            from scipy.ndimage import gaussian_filter
            emphasis = profile['texture_emphasis']
            
            blurred = np.stack([
                gaussian_filter(enhanced[:, :, c], sigma=1.0)
                for c in range(3)
            ], axis=-1)
            
            detail = enhanced - blurred
            texture_enhanced = enhanced + detail * emphasis * effective_confidence[:, :, None]
            enhanced = np.clip(texture_enhanced, 0, 1)
        
        return np.clip(enhanced, 0, 1)
    
    def enhance(
        self,
        image: np.ndarray,
        surfaces: Optional[List[str]] = None,
        depth_map: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply material response to image.
        
        Args:
            image: RGB image as float32 [0, 1]
            surfaces: List of surface types to enhance (None = use config)
            depth_map: Optional depth map for depth-aware enhancement
            
        Returns:
            Enhanced image as float32 [0, 1]
        """
        if surfaces is None:
            surfaces = self.config.surface_types
        
        logger.info(f"Applying material response (surfaces: {surfaces})...")
        
        # Detect materials
        material_maps = self.detect_materials(image)
        
        # Apply enhancements
        enhanced = image.copy()
        
        for surface_name in surfaces:
            try:
                surface_type = SurfaceType(surface_name.lower())
            except ValueError:
                logger.warning(f"Unknown surface type: {surface_name}")
                continue
            
            if surface_type in material_maps:
                confidence_map = material_maps[surface_type]
                
                # Only enhance if significant material presence
                if confidence_map.max() > 0.1:
                    logger.info(f"  Enhancing {surface_type.value} "
                               f"(coverage: {confidence_map.mean()*100:.1f}%)")
                    enhanced = self.enhance_surface(
                        enhanced,
                        surface_type,
                        confidence_map,
                        depth_map
                    )
        
        logger.info("✓ Material response applied")
        return enhanced


def create_material_responder(
    strength: float = 0.75,
    surfaces: Optional[List[str]] = None,
    depth_aware: bool = True
) -> MaterialResponder:
    """
    Convenience function to create material responder.
    
    Args:
        strength: Enhancement strength (0-1)
        surfaces: List of surfaces to enhance
        depth_aware: Use depth information if available
        
    Returns:
        Configured MaterialResponder
    """
    config = MaterialResponseConfig(
        strength=strength,
        surface_types=surfaces,
        depth_aware=depth_aware
    )
    return MaterialResponder(config)


if __name__ == "__main__":
    # Test material responder
    logging.basicConfig(level=logging.INFO)
    
    responder = create_material_responder()
    
    # Create test image
    test_image = np.random.rand(256, 256, 3).astype(np.float32)
    
    # Add some "wood-like" regions (browns)
    test_image[50:150, 50:150, 0] = 0.6  # R
    test_image[50:150, 50:150, 1] = 0.4  # G
    test_image[50:150, 50:150, 2] = 0.2  # B
    
    enhanced = responder.enhance(test_image, surfaces=["wood", "metal"])
    
    print(f"✓ Material response test successful")
    print(f"  Input shape: {test_image.shape}")
    print(f"  Enhanced shape: {enhanced.shape}")
    print(f"  Value range: [{enhanced.min():.3f}, {enhanced.max():.3f}]")
