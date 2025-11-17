"""Sky blending and replacement utilities.

Integrates generated skies into architectural images:
- Sky detection and masking
- Seamless blending at horizon
- Reflection updates (water, glass)
- Lighting consistency
- HDR preservation for IBL

For luxury real estate:
- Natural sky replacement
- Enhanced golden hour skies
- Location-specific atmospheres
- Maintains architectural realism
"""

import logging
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image


logger = logging.getLogger(__name__)


class SkyBlender:
    """Blend generated skies into architectural images.

    Provides intelligent sky replacement with:
    - Automatic sky detection
    - Seamless horizon blending
    - Reflection updates
    - Lighting consistency

    Example:
        >>> blender = SkyBlender()
        >>> image = load_image("property.jpg")
        >>> new_sky = sky_generator.generate_sky(params)
        >>> result = blender.blend_sky(
        ...     image,
        ...     new_sky,
        ...     blend_width=50,
        ...     update_reflections=True
        ... )
    """

    def __init__(self):
        """Initialize sky blender."""
        logger.info("SkyBlender initialized")

    def blend_sky(
        self,
        image: np.ndarray,
        sky: np.ndarray,
        mask: Optional[np.ndarray] = None,
        blend_width: int = 50,
        update_reflections: bool = False,
        reflection_strength: float = 0.5
    ) -> np.ndarray:
        """Blend new sky into image.

        Args:
            image: Original image (H, W, 3)
            sky: Generated sky (H, W, 3)
            mask: Sky mask (H, W) where 1=sky, 0=not sky (auto-detected if None)
            blend_width: Feathering width in pixels at horizon
            update_reflections: Update water/glass reflections
            reflection_strength: Reflection update strength (0-1)

        Returns:
            Image with blended sky
        """
        # Detect sky region if mask not provided
        if mask is None:
            logger.info("Auto-detecting sky region...")
            mask = self._detect_sky_mask(image)

        # Resize sky to match image
        if sky.shape[:2] != image.shape[:2]:
            sky = cv2.resize(sky, (image.shape[1], image.shape[0]))

        # Create smooth transition mask
        blend_mask = self._create_blend_mask(mask, blend_width)

        # Ensure sky is same dtype as image
        if sky.dtype != image.dtype:
            if image.dtype == np.uint8:
                if sky.dtype == np.float32:
                    sky = (sky * 255).clip(0, 255).astype(np.uint8)
            elif image.dtype == np.float32:
                if sky.dtype == np.uint8:
                    sky = sky.astype(np.float32) / 255.0

        # Blend sky
        result = image.copy()

        for c in range(3):
            result[:, :, c] = (
                image[:, :, c] * (1 - blend_mask) +
                sky[:, :, c] * blend_mask
            )

        # Update reflections if requested
        if update_reflections:
            result = self._update_reflections(
                result,
                sky,
                blend_mask,
                reflection_strength
            )

        logger.info("Sky blending complete")

        return result.astype(image.dtype)

    def _detect_sky_mask(
        self,
        image: np.ndarray,
        threshold_method: str = "adaptive"
    ) -> np.ndarray:
        """Detect sky region in image.

        Uses color and brightness cues to identify sky.

        Args:
            image: Input image
            threshold_method: "adaptive" or "simple"

        Returns:
            Binary mask (H, W) where 1=sky
        """
        # Convert to float
        img_float = image.astype(np.float32) / 255.0 if image.dtype == np.uint8 else image

        # Sky is typically:
        # 1. In upper portion of image
        # 2. Brighter than foreground
        # 3. Blue-ish (usually)

        # Create initial mask based on brightness and position
        brightness = np.mean(img_float, axis=2)

        # Upper portion bias
        height = image.shape[0]
        y_coords = np.arange(height)[:, np.newaxis]
        position_bias = 1.0 - (y_coords / height)  # 1 at top, 0 at bottom

        # Combine brightness and position
        sky_probability = brightness * 0.7 + position_bias * 0.3

        # Threshold
        if threshold_method == "adaptive":
            # Use Otsu's method
            sky_prob_uint8 = (sky_probability * 255).astype(np.uint8)
            threshold, mask = cv2.threshold(
                sky_prob_uint8,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            mask = mask.astype(np.float32) / 255.0
        else:
            # Simple threshold
            mask = (sky_probability > 0.6).astype(np.float32)

        # Morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)

        # Keep only largest connected component (main sky region)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_uint8,
            connectivity=8
        )

        if num_labels > 1:
            # Find largest component (excluding background)
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask_uint8 = (labels == largest_label).astype(np.uint8) * 255

        mask = mask_uint8.astype(np.float32) / 255.0

        return mask

    def _create_blend_mask(
        self,
        mask: np.ndarray,
        blend_width: int
    ) -> np.ndarray:
        """Create smooth transition mask for blending.

        Args:
            mask: Binary sky mask
            blend_width: Feathering width in pixels

        Returns:
            Smooth blend mask (0-1)
        """
        # Convert to uint8 for distance transform
        mask_uint8 = (mask * 255).astype(np.uint8)

        # Find boundary
        _, binary = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)

        # Distance transform from boundary
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

        # Create gradient in blend region
        blend_mask = np.clip(dist_transform / blend_width, 0, 1)

        # Apply Gaussian blur for smooth transition
        blend_mask = cv2.GaussianBlur(blend_mask, (21, 21), 10)

        return blend_mask

    def _update_reflections(
        self,
        image: np.ndarray,
        sky: np.ndarray,
        sky_mask: np.ndarray,
        strength: float
    ) -> np.ndarray:
        """Update reflections to match new sky.

        Detects reflective surfaces (water, glass) and updates
        their reflection to match the new sky.

        Args:
            image: Image with blended sky
            sky: New sky
            sky_mask: Sky region mask
            strength: Update strength (0-1)

        Returns:
            Image with updated reflections
        """
        # Detect potential reflection regions
        # Reflections are typically:
        # 1. In lower portion of image
        # 2. Similar color to sky
        # 3. Darker than sky (Fresnel effect)

        # Simple heuristic: lower third of image with similar hue to sky
        height = image.shape[0]
        lower_region = image[int(height * 0.6):, :]

        # Calculate reflection of sky (vertically flipped and darkened)
        reflected_sky = np.flipud(sky) * 0.6  # Darken reflection

        # Very simple reflection update in lower region
        # Production version would use more sophisticated detection
        result = image.copy()

        lower_height = lower_region.shape[0]
        if reflected_sky.shape[0] >= lower_height:
            reflection_region = reflected_sky[:lower_height]

            # Blend reflection with original
            result[int(height * 0.6):, :] = (
                lower_region * (1 - strength * 0.3) +
                reflection_region * (strength * 0.3)
            )

        return result

    def replace_sky_in_panorama(
        self,
        panorama: np.ndarray,
        sky_params: dict,
        sky_generator: any,
        blend_width: int = 100
    ) -> np.ndarray:
        """Replace sky in panoramic image.

        Args:
            panorama: Panoramic image (equirectangular)
            sky_params: SkyParameters dictionary
            sky_generator: SkyGANGenerator instance
            blend_width: Blend width

        Returns:
            Panorama with replaced sky
        """
        # Generate sky matching panorama resolution
        sky = sky_generator.generate_sky(
            sky_params,
            resolution=(panorama.shape[1], panorama.shape[0])
        )

        # Detect sky mask
        mask = self._detect_sky_mask(panorama)

        # Blend
        result = self.blend_sky(
            panorama,
            sky,
            mask,
            blend_width=blend_width,
            update_reflections=True
        )

        return result

    def create_sky_mask_manual(
        self,
        image_shape: Tuple[int, int],
        horizon_y: int,
        building_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Create sky mask manually with specified horizon.

        Args:
            image_shape: Image shape (height, width)
            horizon_y: Horizon line y-coordinate
            building_mask: Optional building silhouette mask to exclude

        Returns:
            Sky mask
        """
        height, width = image_shape
        mask = np.zeros((height, width), dtype=np.float32)

        # Everything above horizon is sky
        mask[:horizon_y, :] = 1.0

        # Exclude buildings if provided
        if building_mask is not None:
            mask = mask * (1 - building_mask)

        return mask

    def match_sky_color_temperature(
        self,
        image: np.ndarray,
        sky: np.ndarray,
        image_sky_region: np.ndarray
    ) -> np.ndarray:
        """Match new sky color temperature to original.

        Adjusts new sky to match the color temperature/tone of
        the original sky for consistency.

        Args:
            image: Original image
            sky: New sky to adjust
            image_sky_region: Region of original image containing sky

        Returns:
            Color-matched sky
        """
        # Calculate average color of original sky
        if image.dtype == np.uint8:
            original_sky_color = np.mean(image_sky_region, axis=(0, 1)) / 255.0
        else:
            original_sky_color = np.mean(image_sky_region, axis=(0, 1))

        # Calculate average color of new sky
        if sky.dtype == np.uint8:
            new_sky_color = np.mean(sky, axis=(0, 1)) / 255.0
            sky_float = sky.astype(np.float32) / 255.0
        else:
            new_sky_color = np.mean(sky, axis=(0, 1))
            sky_float = sky.copy()

        # Calculate color shift
        color_shift = original_sky_color / (new_sky_color + 1e-6)

        # Apply shift
        matched_sky = sky_float * color_shift

        # Convert back to original dtype
        if sky.dtype == np.uint8:
            matched_sky = (matched_sky * 255).clip(0, 255).astype(np.uint8)

        return matched_sky

    def __repr__(self) -> str:
        return "SkyBlender()"
