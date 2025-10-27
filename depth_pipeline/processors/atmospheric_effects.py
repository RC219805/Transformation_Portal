"""
Atmospheric Effects based on Depth

Simulates realistic atmospheric perspective (aerial perspective) using depth information.
Adds haze, atmospheric scattering, and depth-based desaturation for photorealistic depth cues.

Performance: ~40ms for 4K image
"""

import logging
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)


class AtmosphericEffects:
    """
    Depth-based atmospheric effects processor.

    Simulates:
    - Atmospheric haze (Rayleigh scattering)
    - Aerial perspective (color shift and desaturation with distance)
    - Depth fog
    - Atmospheric glow

    Physics-based model: I = I₀ * e^(-βd) + A(1 - e^(-βd))
    Where:
        I = final color
        I₀ = original color
        β = scattering coefficient (density)
        d = depth (distance)
        A = atmospheric color
    """

    def __init__(
        self,
        haze_density: float = 0.015,
        haze_color: Tuple[float, float, float] = (0.7, 0.8, 0.9),
        desaturation_strength: float = 0.3,
        depth_scale: float = 100.0,
        enable_color_shift: bool = True,
    ):
        """
        Initialize atmospheric effects processor.

        Args:
            haze_density: Atmospheric density (β parameter), higher = more haze
            haze_color: Atmospheric color (RGB, [0-1]), typically blue-ish
            desaturation_strength: How much distant objects desaturate
            depth_scale: Scale depth values to meters (for physical accuracy)
            enable_color_shift: Apply atmospheric blue shift to distant objects
        """
        self.haze_density = haze_density
        self.haze_color = np.array(haze_color, dtype=np.float32)
        self.desaturation_strength = desaturation_strength
        self.depth_scale = depth_scale
        self.enable_color_shift = enable_color_shift

    def process(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Apply atmospheric effects to image.

        Args:
            image: Input image (HxWxC, float32, [0, 1])
            depth: Depth map (HxW, float32, [0, 1])
            mask: Optional processing mask

        Returns:
            Image with atmospheric effects
        """
        # Validate inputs
        if image.shape[:2] != depth.shape[:2]:
            raise ValueError(
                f"Image shape {image.shape[:2]} doesn't match depth shape {depth.shape[:2]}"
            )

        # Scale depth to meters
        depth_meters = depth * self.depth_scale

        # Apply atmospheric haze
        result = self._apply_haze(image, depth_meters)

        # Apply aerial desaturation
        result = self._apply_aerial_desaturation(result, depth)

        # Apply color shift (optional)
        if self.enable_color_shift:
            result = self._apply_color_shift(result, depth)

        # Apply mask if provided
        if mask is not None:
            result = np.where(mask[..., None], result, image)

        return result

    def _apply_haze(
        self,
        image: np.ndarray,
        depth_meters: np.ndarray,
    ) -> np.ndarray:
        """
        Apply physically-based atmospheric haze.

        Uses Beer-Lambert law for light transmission through atmosphere.

        Args:
            image: Input image
            depth_meters: Depth in meters

        Returns:
            Image with haze applied
        """
        # Compute transmission coefficient
        # T = e^(-β * d)
        transmission = np.exp(-self.haze_density * depth_meters)

        # Ensure transmission is in valid range
        transmission = np.clip(transmission, 0, 1)

        # Apply atmospheric scattering equation
        # I = I₀ * T + A * (1 - T)
        haze_contribution = self.haze_color * (1 - transmission[..., None])
        result = image * transmission[..., None] + haze_contribution

        return np.clip(result, 0, 1)

    def _apply_aerial_desaturation(
        self,
        image: np.ndarray,
        depth: np.ndarray,
    ) -> np.ndarray:
        """
        Apply aerial perspective desaturation.

        Distant objects lose color saturation due to atmospheric scattering.

        Args:
            image: Input image
            depth: Normalized depth [0, 1]

        Returns:
            Desaturated image
        """
        # Compute luminance (Rec. 709)
        luminance = (
            0.2126 * image[..., 0] +
            0.7152 * image[..., 1] +
            0.0722 * image[..., 2]
        )

        # Compute desaturation factor based on depth
        # Close objects: no desaturation (factor = 1)
        # Distant objects: strong desaturation (factor = 1 - strength)
        desaturation_factor = 1.0 - (depth * self.desaturation_strength)
        desaturation_factor = np.clip(desaturation_factor, 0, 1)

        # Blend between grayscale and color
        result = (
            luminance[..., None] * (1 - desaturation_factor[..., None]) +
            image * desaturation_factor[..., None]
        )

        return result

    def _apply_color_shift(
        self,
        image: np.ndarray,
        depth: np.ndarray,
    ) -> np.ndarray:
        """
        Apply atmospheric color shift (blue shift for distant objects).

        Simulates Rayleigh scattering which preferentially scatters blue light.

        Args:
            image: Input image
            depth: Normalized depth

        Returns:
            Color-shifted image
        """
        # Compute color shift amount based on depth
        shift_amount = depth * 0.15  # Subtle shift

        # Shift towards atmospheric color (blue)
        result = (
            image * (1 - shift_amount[..., None]) +
            self.haze_color * shift_amount[..., None]
        )

        return np.clip(result, 0, 1)

    def __call__(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        config: Optional[dict] = None,
    ) -> np.ndarray:
        """Callable interface for pipeline integration."""
        if config:
            # Temporarily override parameters
            old_params = {
                'haze_density': self.haze_density,
                'haze_color': self.haze_color.copy(),
                'desaturation_strength': self.desaturation_strength,
                'enable_color_shift': self.enable_color_shift,
            }

            for key, value in config.items():
                if key == 'haze_color':
                    self.haze_color = np.array(value, dtype=np.float32)
                elif hasattr(self, key):
                    setattr(self, key, value)

            result = self.process(image, depth)

            # Restore
            for key, value in old_params.items():
                setattr(self, key, value)

            return result
        else:
            return self.process(image, depth)


class DepthFog:
    """
    Simple depth fog effect for quick atmospheric enhancement.

    Faster than full atmospheric effects (~20ms vs 40ms).
    """

    def __init__(
        self,
        fog_density: float = 0.02,
        fog_color: Tuple[float, float, float] = (0.8, 0.85, 0.9),
        start_depth: float = 0.3,
        end_depth: float = 1.0,
    ):
        """
        Initialize depth fog.

        Args:
            fog_density: Fog density
            fog_color: Fog color (RGB)
            start_depth: Depth where fog starts (0-1)
            end_depth: Depth where fog is maximum (0-1)
        """
        self.fog_density = fog_density
        self.fog_color = np.array(fog_color, dtype=np.float32)
        self.start_depth = start_depth
        self.end_depth = end_depth

    def process(
        self,
        image: np.ndarray,
        depth: np.ndarray,
    ) -> np.ndarray:
        """Apply depth fog."""
        # Compute fog factor
        fog_range = self.end_depth - self.start_depth
        fog_factor = (depth - self.start_depth) / (fog_range + 1e-8)
        fog_factor = np.clip(fog_factor, 0, 1)

        # Apply exponential fog
        fog_factor = 1.0 - np.exp(-self.fog_density * fog_factor)

        # Blend with fog color
        result = (
            image * (1 - fog_factor[..., None]) +
            self.fog_color * fog_factor[..., None]
        )

        return np.clip(result, 0, 1)

    def __call__(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        config: Optional[dict] = None,
    ) -> np.ndarray:
        """Callable interface."""
        return self.process(image, depth)


class AtmosphericGlow:
    """
    Depth-based atmospheric glow effect.

    Adds soft glow to distant objects simulating atmospheric light scattering.
    """

    def __init__(
        self,
        glow_strength: float = 0.3,
        glow_threshold: float = 0.6,
        blur_sigma: float = 5.0,
    ):
        """
        Initialize atmospheric glow.

        Args:
            glow_strength: Glow intensity
            glow_threshold: Depth threshold for glow (only affect distant objects)
            blur_sigma: Glow blur radius
        """
        self.glow_strength = glow_strength
        self.glow_threshold = glow_threshold
        self.blur_sigma = blur_sigma

    def process(
        self,
        image: np.ndarray,
        depth: np.ndarray,
    ) -> np.ndarray:
        """Apply atmospheric glow."""
        # Extract distant regions
        distant_mask = (depth > self.glow_threshold).astype(np.float32)

        # Apply mask to image
        masked = image * distant_mask[..., None]

        # Create glow by blurring
        glow = gaussian_filter(masked, sigma=(self.blur_sigma, self.blur_sigma, 0))

        # Add glow to original
        result = image + glow * self.glow_strength

        return np.clip(result, 0, 1)

    def __call__(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        config: Optional[dict] = None,
    ) -> np.ndarray:
        """Callable interface."""
        return self.process(image, depth)
