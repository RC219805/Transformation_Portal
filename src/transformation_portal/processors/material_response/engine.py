#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Material Response Engine for Transformation Portal.

Provides a configurable Material Response processor with profile-based
surface enhancement for luxury real estate and architectural visualization.

The engine implements the three core Material Response tenets:
1. Respect energy conservation in highlights (preserve specular sheen)
2. Preserve midtone texture (keep materials tactile and dimensional)
3. Blend transitions between materials (authored, not procedural)

Example:
    from transformation_portal.processors.material_response.engine import (
        MaterialResponseEngine
    )
    from transformation_portal.processors.material_response.profiles import (
        PROFILES
    )

    engine = MaterialResponseEngine.from_config({
        'profile': 'luxury_interior',
        'texture_boost': 0.25,
        'ambient_occlusion': 0.12
    })
    result = engine.apply(image)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

# Optional scipy import for advanced image processing
try:
    from scipy.ndimage import gaussian_filter, sobel
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    gaussian_filter = None  # type: ignore
    sobel = None  # type: ignore


@dataclass
class MaterialResponseConfig:
    """Configuration for Material Response Engine.

    Attributes:
        profile: Material profile name (e.g., 'luxury_interior').
        texture_boost: High-frequency texture enhancement (0.0-1.0).
        ambient_occlusion: Contact shadow intensity (0.0-1.0).
        highlight_warmth: Warm highlight mix (0.0-1.0).
        haze_strength: Volumetric haze blend (0.0-1.0).
        haze_tint: RGB haze tint values.
        floor_plank_contrast: Wood floor definition enhancement.
        floor_specular: Specular streak intensity on flooring.
        textile_contrast: Linen/fabric separation.
        leather_sheen: Leather surface sheen.
        window_light_wrap: Window light wrap intensity.
        window_reflection: Window reflection on floors.
    """
    profile: str = "luxury_interior"
    texture_boost: float = 0.25
    ambient_occlusion: float = 0.12
    highlight_warmth: float = 0.08
    haze_strength: float = 0.06
    haze_tint: Tuple[float, float, float] = (0.82, 0.88, 0.96)
    floor_plank_contrast: float = 0.12
    floor_specular: float = 0.18
    textile_contrast: float = 0.18
    leather_sheen: float = 0.16
    window_light_wrap: float = 0.14
    window_reflection: float = 0.12
    wall_texture: float = 0.1

    def __post_init__(self):
        """Clamp values to valid ranges."""
        self.texture_boost = max(0.0, min(1.0, self.texture_boost))
        self.ambient_occlusion = max(0.0, min(1.0, self.ambient_occlusion))
        self.highlight_warmth = max(0.0, min(1.0, self.highlight_warmth))
        self.haze_strength = max(0.0, min(1.0, self.haze_strength))
        self.floor_plank_contrast = max(0.0, min(1.0, self.floor_plank_contrast))
        self.floor_specular = max(0.0, min(1.0, self.floor_specular))
        self.textile_contrast = max(0.0, min(1.0, self.textile_contrast))
        self.leather_sheen = max(0.0, min(1.0, self.leather_sheen))
        self.window_light_wrap = max(0.0, min(1.0, self.window_light_wrap))
        self.window_reflection = max(0.0, min(1.0, self.window_reflection))
        self.wall_texture = max(0.0, min(1.0, self.wall_texture))


@dataclass
class MaterialMask:
    """Container for material region masks.

    Attributes:
        floor: Floor region mask.
        wall: Wall region mask.
        textile: Textile region mask.
        wood: Wood surface mask.
        metal: Metal/glass region mask.
        highlight: Highlight region mask.
        midtone: Midtone region mask.
    """
    floor: np.ndarray
    wall: np.ndarray
    textile: np.ndarray
    wood: np.ndarray
    metal: np.ndarray
    highlight: np.ndarray
    midtone: np.ndarray


class MaterialResponseEngine:
    """Material Response processing engine.

    Implements physics-based surface enhancement with profile support
    for luxury real estate and architectural visualization.
    """

    def __init__(self, config: MaterialResponseConfig):
        """Initialize the engine with configuration.

        Args:
            config: MaterialResponseConfig instance.
        """
        self.config = config

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> "MaterialResponseEngine":
        """Create engine from configuration dictionary.

        Args:
            config_dict: Configuration dictionary.

        Returns:
            MaterialResponseEngine instance.

        Example:
            engine = MaterialResponseEngine.from_config({
                'profile': 'luxury_interior',
                'texture_boost': 0.3
            })
        """
        # Load profile defaults if specified
        profile_name = config_dict.get('profile', 'luxury_interior')

        # Import profiles dynamically to avoid circular imports
        try:
            from .profiles import get_profile
            profile_defaults = get_profile(profile_name)
        except ImportError:
            # Circular import - use empty defaults
            profile_defaults = {}
        except KeyError:
            # Invalid profile name - raise error to notify user
            raise KeyError(
                f"MaterialResponseEngine.from_config: Invalid profile name '{profile_name}'. "
                "Available profiles can be found in transformation_portal.processors.material_response.profiles."
            )

        # Merge profile defaults with explicit config
        merged = {**profile_defaults, **config_dict}

        # Handle haze_tint tuple
        if 'haze_tint' in merged and isinstance(merged['haze_tint'], (list, tuple)):
            merged['haze_tint'] = tuple(merged['haze_tint'][:3])

        # Create config
        config = MaterialResponseConfig(
            profile=merged.get('profile', 'luxury_interior'),
            texture_boost=merged.get('texture_boost', 0.25),
            ambient_occlusion=merged.get('ambient_occlusion', 0.12),
            highlight_warmth=merged.get('highlight_warmth', 0.08),
            haze_strength=merged.get('haze_strength', 0.06),
            haze_tint=merged.get('haze_tint', (0.82, 0.88, 0.96)),
            floor_plank_contrast=merged.get('floor_plank_contrast', 0.12),
            floor_specular=merged.get('floor_specular', 0.18),
            textile_contrast=merged.get('textile_contrast', 0.18),
            leather_sheen=merged.get('leather_sheen', 0.16),
            window_light_wrap=merged.get('window_light_wrap', 0.14),
            window_reflection=merged.get('window_reflection', 0.12),
            wall_texture=merged.get('wall_texture', 0.1),
        )

        return cls(config)

    def apply(
        self,
        image: Image.Image,
        profile: Optional[str] = None,
        strength: float = 1.0
    ) -> Image.Image:
        """Apply Material Response enhancement to an image.

        Args:
            image: Input PIL Image.
            profile: Optional profile name to override config.
            strength: Overall strength multiplier (0.0-1.0).

        Returns:
            Enhanced PIL Image.
        """
        # Check for scipy availability
        if not HAS_SCIPY:
            import logging
            logging.getLogger(__name__).warning(
                "scipy not available, returning original image. "
                "Install scipy for Material Response processing."
            )
            return image

        # Convert to float32 RGB array
        if image.mode != 'RGB':
            image = image.convert('RGB')

        rgb = np.array(image).astype(np.float32) / 255.0
        h, w = rgb.shape[:2]

        # Compute material masks
        masks = self._compute_material_masks(rgb)

        # Apply enhancement stages
        result = rgb.copy()

        # 1. Texture boost (midtone detail enhancement)
        if self.config.texture_boost > 0:
            result = self._enhance_texture(result, masks.midtone, strength)

        # 2. Floor enhancement (wood grain and specular)
        if self.config.floor_plank_contrast > 0 or self.config.floor_specular > 0:
            result = self.enhance_floor(result, masks.floor, masks.wood, strength)

        # 3. Textile enhancement
        if self.config.textile_contrast > 0:
            result = self.enhance_textiles(result, masks.textile, strength)

        # 4. Metal/glass enhancement
        if masks.metal.max() > 0.01:
            result = self.enhance_metals(result, masks.metal, strength)

        # 5. Wall texture
        if self.config.wall_texture > 0:
            result = self._enhance_walls(result, masks.wall, strength)

        # 6. Ambient occlusion (contact shadows)
        if self.config.ambient_occlusion > 0:
            result = self._apply_ambient_occlusion(result, rgb, masks.floor, strength)

        # 7. Window effects
        if self.config.window_light_wrap > 0 or self.config.window_reflection > 0:
            result = self._apply_window_effects(result, masks.floor, h, w, strength)

        # 8. Highlight warmth (energy conservation)
        if self.config.highlight_warmth > 0:
            result = self._apply_highlight_warmth(result, masks.highlight, strength)

        # 9. Atmospheric haze
        if self.config.haze_strength > 0:
            result = self.add_atmospheric_effects(result, h, w, strength)

        # Ensure valid range
        result = np.clip(result, 0.0, 1.0)

        return Image.fromarray((result * 255).astype(np.uint8), 'RGB')

    def enhance_floor(
        self,
        rgb: np.ndarray,
        floor_mask: np.ndarray,
        wood_mask: np.ndarray,
        strength: float = 1.0
    ) -> np.ndarray:
        """Enhance floor surfaces with wood grain and specular effects.

        Args:
            rgb: Input RGB array (float32, 0-1).
            floor_mask: Floor region mask.
            wood_mask: Wood surface mask.
            strength: Strength multiplier.

        Returns:
            Enhanced RGB array.
        """
        result = rgb.copy()
        luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

        # Wood grain enhancement
        if self.config.floor_plank_contrast > 0 and wood_mask.max() > 0.01:
            # High-frequency floor detail
            blurred_floor = gaussian_filter(rgb * floor_mask[..., np.newaxis], sigma=(1.6, 1.2, 0))
            floor_detail = rgb * floor_mask[..., np.newaxis] - blurred_floor

            plank_strength = self.config.floor_plank_contrast * strength
            result = np.clip(
                result + plank_strength * floor_detail * floor_mask[..., np.newaxis],
                0.0, 1.0
            )

            # Directional grain detection
            grain = np.abs(sobel(luminance * wood_mask, axis=1))
            grain = gaussian_filter(grain, sigma=(0.8, 3.0))
            if grain.max() > 0:
                grain = grain / grain.max()

            # Warm wood color contribution
            warm_wood = np.array([0.86, 0.74, 0.58], dtype=np.float32)
            wood_weight = 0.12 * plank_strength * wood_mask[..., np.newaxis] * grain[..., np.newaxis]
            result = np.clip(result + wood_weight * (warm_wood - result), 0.0, 1.0)

        # Specular streaks
        if self.config.floor_specular > 0:
            floor_grad = np.abs(sobel(luminance * floor_mask, axis=1))
            if floor_grad.max() > 0:
                floor_grad = floor_grad / floor_grad.max()
            streaks = gaussian_filter(floor_grad, sigma=(2.0, 5.0))

            spec_strength = self.config.floor_specular * strength
            spec_color = np.array([1.0, 0.94, 0.80], dtype=np.float32)
            streak_weight = spec_strength * streaks[..., np.newaxis] * floor_mask[..., np.newaxis]
            result = np.clip(result + streak_weight * (spec_color - result), 0.0, 1.0)

        return result

    def enhance_textiles(
        self,
        rgb: np.ndarray,
        textile_mask: np.ndarray,
        strength: float = 1.0
    ) -> np.ndarray:
        """Enhance textile surfaces with micro-contrast.

        Args:
            rgb: Input RGB array (float32, 0-1).
            textile_mask: Textile region mask.
            strength: Strength multiplier.

        Returns:
            Enhanced RGB array.
        """
        if textile_mask.max() < 0.01:
            return rgb

        result = rgb.copy()

        # High-frequency textile detail
        textile_detail = rgb - gaussian_filter(rgb, sigma=(1.4, 1.4, 0))
        textile_strength = self.config.textile_contrast * strength
        textile_weight = textile_strength * textile_mask[..., np.newaxis]
        result = np.clip(result + textile_weight * textile_detail, 0.0, 1.0)

        return result

    def enhance_metals(
        self,
        rgb: np.ndarray,
        metal_mask: np.ndarray,
        strength: float = 1.0
    ) -> np.ndarray:
        """Enhance metal and glass surfaces with specular preservation.

        Args:
            rgb: Input RGB array (float32, 0-1).
            metal_mask: Metal/glass region mask.
            strength: Strength multiplier.

        Returns:
            Enhanced RGB array.
        """
        if metal_mask.max() < 0.01:
            return rgb

        result = rgb.copy()
        luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

        # Specular enhancement
        specular = gaussian_filter(luminance * metal_mask, sigma=2.0)
        specular = np.clip((specular - 0.35) / 0.5, 0.0, 1.0)

        cool_metal = np.array([0.93, 0.95, 0.98], dtype=np.float32)
        metal_strength = 0.1 * strength
        metal_weight = metal_strength * metal_mask[..., np.newaxis] * specular[..., np.newaxis]
        result = np.clip(result + metal_weight * (cool_metal - result), 0.0, 1.0)

        return result

    def add_atmospheric_effects(
        self,
        rgb: np.ndarray,
        height: int,
        width: int,
        strength: float = 1.0
    ) -> np.ndarray:
        """Add atmospheric haze effect.

        Args:
            rgb: Input RGB array (float32, 0-1).
            height: Image height.
            width: Image width.
            strength: Strength multiplier.

        Returns:
            RGB array with atmospheric effects.
        """
        result = rgb.copy()

        # Vertical gradient for haze
        y_norm = np.linspace(0, 1, height).reshape(-1, 1)
        y_norm = np.broadcast_to(y_norm, (height, width))

        haze_amount = self.config.haze_strength * strength * y_norm.astype(np.float32)
        tint = np.array(self.config.haze_tint, dtype=np.float32)
        tint = np.clip(tint, 0.0, 1.0)

        result = np.clip(result * (1.0 - haze_amount[..., np.newaxis]) + tint * haze_amount[..., np.newaxis], 0.0, 1.0)

        return result

    def _compute_material_masks(self, rgb: np.ndarray) -> MaterialMask:
        """Compute material region masks from image.

        Args:
            rgb: Input RGB array (float32, 0-1).

        Returns:
            MaterialMask with all computed masks.
        """
        h, w = rgb.shape[:2]

        # Compute luminance and saturation
        luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        saturation = np.maximum(rgb.max(axis=2) - rgb.min(axis=2), 1e-6)

        # Vertical position for perspective-based detection
        y_norm = np.linspace(0, 1, h).reshape(-1, 1)
        y_norm = np.broadcast_to(y_norm, (h, w)).astype(np.float32)

        # Floor mask (lower portion)
        floor_mask = np.clip((y_norm - 0.55) / 0.45, 0.0, 1.0).astype(np.float32)

        # Wall mask (upper-mid, low saturation)
        wall_mask = (
            np.clip((luminance - 0.32) / 0.45, 0.0, 1.0) *
            np.clip((0.26 - saturation) / 0.26, 0.0, 1.0) *
            np.clip(1.0 - floor_mask, 0.0, 1.0)
        )
        wall_mask = gaussian_filter(wall_mask, sigma=1.5)

        # Wood detection (warm mid-tones on floor)
        warm_bias = rgb[..., 0] - 0.5 * (rgb[..., 1] + rgb[..., 2])
        wood_mask = (
            np.clip((warm_bias + 0.08) / 0.18, 0.0, 1.0) *
            np.clip((saturation - 0.06) / 0.22, 0.0, 1.0) *
            np.clip((luminance - 0.18) / 0.5, 0.0, 1.0) *
            floor_mask
        )
        wood_mask = gaussian_filter(wood_mask, sigma=2.5)

        # Textile detection (soft, mid-brightness, neutral)
        textile_mask = (
            np.clip((luminance - 0.35) / 0.4, 0.0, 1.0) *
            np.clip((0.28 - saturation) / 0.28, 0.0, 1.0) *
            np.clip(1.0 - floor_mask, 0.0, 1.0)
        )
        textile_mask = gaussian_filter(textile_mask, sigma=1.8)

        # Metal/glass detection
        neutral_mask = np.clip((0.12 - saturation) / 0.12, 0.0, 1.0)
        edge_mag = np.abs(sobel(luminance, axis=0)) + np.abs(sobel(luminance, axis=1))
        edge_mag = gaussian_filter(edge_mag, sigma=1.0)
        if edge_mag.max() > 0:
            edge_mag = edge_mag / edge_mag.max()
        metal_mask = neutral_mask * edge_mag * np.clip(luminance, 0.25, 0.85)
        metal_mask = gaussian_filter(metal_mask, sigma=2.0)

        # Highlight mask
        highlight_mask = np.clip((luminance - 0.68) / 0.32, 0.0, 1.0)
        highlight_mask = gaussian_filter(highlight_mask, sigma=2.0)

        # Midtone mask
        midtone_mask = np.clip(1.0 - np.abs(luminance - 0.5) / 0.35, 0.0, 1.0)
        midtone_mask = gaussian_filter(midtone_mask, sigma=1.5)

        return MaterialMask(
            floor=floor_mask,
            wall=wall_mask.astype(np.float32),
            textile=textile_mask.astype(np.float32),
            wood=wood_mask.astype(np.float32),
            metal=metal_mask.astype(np.float32),
            highlight=highlight_mask.astype(np.float32),
            midtone=midtone_mask.astype(np.float32),
        )

    def _enhance_texture(
        self,
        rgb: np.ndarray,
        midtone_mask: np.ndarray,
        strength: float
    ) -> np.ndarray:
        """Apply texture boost to midtones."""
        blurred = gaussian_filter(rgb, sigma=(1.1, 1.1, 0))
        texture_detail = rgb - blurred
        texture_strength = self.config.texture_boost * strength
        texture_weight = texture_strength * midtone_mask[..., np.newaxis]
        return np.clip(rgb + texture_weight * texture_detail, 0.0, 1.0)

    def _enhance_walls(
        self,
        rgb: np.ndarray,
        wall_mask: np.ndarray,
        strength: float
    ) -> np.ndarray:
        """Apply subtle wall texture enhancement."""
        if wall_mask.max() < 0.01:
            return rgb

        wall_detail = rgb - gaussian_filter(rgb, sigma=(2.2, 2.2, 0))
        wall_strength = self.config.wall_texture * strength
        wall_weight = wall_strength * wall_mask[..., np.newaxis]
        return np.clip(rgb + wall_weight * wall_detail, 0.0, 1.0)

    def _apply_ambient_occlusion(
        self,
        rgb: np.ndarray,
        original: np.ndarray,
        floor_mask: np.ndarray,
        strength: float
    ) -> np.ndarray:
        """Apply ambient occlusion (contact shadows)."""
        luminance = 0.2126 * original[..., 0] + 0.7152 * original[..., 1] + 0.0722 * original[..., 2]

        # Edge-based occlusion
        grad_x = sobel(luminance, axis=1)
        grad_y = sobel(luminance, axis=0)
        edge_mag = np.hypot(grad_x, grad_y)
        if edge_mag.max() > 0:
            edge_mag = edge_mag / edge_mag.max()

        occlusion = gaussian_filter(edge_mag, sigma=1.2)
        occlusion = np.clip(occlusion, 0.0, 1.0)

        # Floor contact shadow
        floor_contact = gaussian_filter(floor_mask * (1.0 - floor_mask), sigma=2.0)
        contact_weight = np.clip(floor_contact, 0.0, 1.0)

        ao_strength = self.config.ambient_occlusion * strength
        shadow = 1.0 - ao_strength * (occlusion + 0.6 * contact_weight)

        return np.clip(rgb * shadow[..., np.newaxis], 0.0, 1.0)

    def _apply_window_effects(
        self,
        rgb: np.ndarray,
        floor_mask: np.ndarray,
        h: int,
        w: int,
        strength: float
    ) -> np.ndarray:
        """Apply window light wrap and reflection effects."""
        result = rgb.copy()
        luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

        x_norm = np.linspace(0, 1, w).reshape(1, -1)
        x_norm = np.broadcast_to(x_norm, (h, w)).astype(np.float32)

        # Window light wrap
        if self.config.window_light_wrap > 0:
            window_side = np.clip((x_norm - 0.45) / 0.55, 0.0, 1.0)
            wrap = gaussian_filter(luminance * window_side, sigma=3.2)
            if wrap.max() > wrap.min():
                wrap = np.clip((wrap - wrap.min()) / (wrap.max() - wrap.min() + 1e-6), 0.0, 1.0)

            wrap_strength = self.config.window_light_wrap * strength
            wrap_color = np.array([1.0, 0.95, 0.82], dtype=np.float32)
            result = np.clip(result + wrap_strength * wrap[..., np.newaxis] * (wrap_color - result), 0.0, 1.0)

        # Window reflection on floor
        if self.config.window_reflection > 0:
            bright_columns = np.mean(np.clip(luminance - 0.65, 0.0, 1.0), axis=0)
            bright_columns = gaussian_filter(bright_columns, sigma=4.0)
            if bright_columns.max() > 0:
                bright_columns = bright_columns / bright_columns.max()
            reflection = np.tile(bright_columns, (h, 1))
            reflection = gaussian_filter(reflection, sigma=(5.0, 3.0))
            reflection = reflection * floor_mask

            refl_strength = self.config.window_reflection * strength
            refl_color = np.array([1.0, 0.98, 0.9], dtype=np.float32)
            refl_mix = refl_strength * reflection[..., np.newaxis] * (refl_color - result)
            result = np.clip(result + refl_mix, 0.0, 1.0)

        return result

    def _apply_highlight_warmth(
        self,
        rgb: np.ndarray,
        highlight_mask: np.ndarray,
        strength: float
    ) -> np.ndarray:
        """Apply warm tint to highlights (energy conservation)."""
        warm_highlight = np.array([1.0, 0.80, 0.58], dtype=np.float32)
        warmth_strength = self.config.highlight_warmth * strength
        highlight_weight = warmth_strength * highlight_mask[..., np.newaxis]
        return np.clip(rgb + highlight_weight * (warm_highlight - rgb), 0.0, 1.0)


__all__ = [
    'MaterialResponseEngine',
    'MaterialResponseConfig',
    'MaterialMask',
]
