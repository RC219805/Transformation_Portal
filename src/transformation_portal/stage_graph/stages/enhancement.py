"""
Image enhancement stage.

Applies material-aware enhancements including:
- Tone mapping
- Clarity enhancement
- Material-specific processing
- Atmospheric effects
"""

from __future__ import annotations

from typing import Dict
import hashlib

import numpy as np

from ..stage import Stage, StageContext, StageResult, StageStatus


class EnhancementStage(Stage):
    """
    Material-aware image enhancement stage.

    Applies physics-based enhancements respecting material properties.
    """

    def __init__(
        self,
        enhancement_strength: float = 0.7,
        clarity_strength: float = 0.5,
        material_strength: float = 0.6,
        version: str = "1.0.0",
    ):
        """
        Initialize enhancement stage.

        Args:
            enhancement_strength: Global enhancement strength [0, 1]
            clarity_strength: Clarity enhancement strength [0, 1]
            material_strength: Material enhancement strength [0, 1]
            version: Stage version for cache invalidation
        """
        super().__init__(name="enhancement", version=version)
        self.enhancement_strength = enhancement_strength
        self.clarity_strength = clarity_strength
        self.material_strength = material_strength

    def get_dependencies(self) -> list:
        """Depends on depth and materials for best results."""
        # Note: These are soft dependencies - stage handles missing inputs
        return ["depth_estimation", "material_segmentation"]

    def compute(self, context: StageContext) -> StageResult:
        """
        Apply enhancements.

        Expected context artifacts:
        - image: Input image as numpy array (H, W, 3)
        - depth_map: Depth map (H, W)
        - material_masks: Dict[str, np.ndarray] with material masks

        Output artifacts:
        - enhanced_image: Enhanced image (H, W, 3)
        - enhancement_metadata: Dict with enhancement info
        """
        import time

        # Get inputs
        image = context.get_artifact("image")
        if image is None:
            return StageResult(
                stage_name=self.name,
                stage_version=self.version,
                status=StageStatus.FAILED,
                error="Missing 'image' artifact in context",
            )

        depth_map = context.get_artifact("depth_map")
        material_masks = context.get_artifact("material_masks", {})

        start = time.time()

        # Apply enhancements
        enhanced_image = self._enhance_image(
            image, depth_map, material_masks
        )

        duration_ms = (time.time() - start) * 1000

        return StageResult(
            stage_name=self.name,
            stage_version=self.version,
            status=StageStatus.COMPLETED,
            artifacts={
                "enhanced_image": enhanced_image,
                "enhancement_metadata": {
                    "enhancement_strength": self.enhancement_strength,
                    "clarity_strength": self.clarity_strength,
                    "material_strength": self.material_strength,
                    "materials_applied": list(material_masks.keys()),
                },
            },
            duration_ms=duration_ms,
            metadata={
                "has_depth": depth_map is not None,
                "has_materials": len(material_masks) > 0,
                "processing_ms": duration_ms,
            },
        )

    def get_cache_key(self, context: StageContext) -> str:
        """Generate cache key based on all inputs."""
        # Get input image
        image = context.get_artifact("image")
        if image is None:
            return "no_image"

        # Hash image
        image_hash = hashlib.sha256(image.tobytes()).hexdigest()[:16]

        # Hash depth map
        depth_map = context.get_artifact("depth_map")
        depth_hash = ""
        if depth_map is not None:
            depth_hash = hashlib.sha256(depth_map.tobytes()).hexdigest()[:8]

        # Hash material masks
        material_masks = context.get_artifact("material_masks", {})
        material_hash = ""
        if material_masks:
            # Sort keys for deterministic order
            sorted_keys = sorted(material_masks.keys())
            material_bytes = b"".join(
                material_masks[k].tobytes() for k in sorted_keys
            )
            material_hash = hashlib.sha256(material_bytes).hexdigest()[:8]

        # Configuration
        config_str = (
            f"{self.enhancement_strength:.2f}_"
            f"{self.clarity_strength:.2f}_"
            f"{self.material_strength:.2f}_"
            f"{self.version}"
        )

        return f"enhance_{config_str}_{image_hash}_{depth_hash}_{material_hash}"

    def _enhance_image(
        self,
        image: np.ndarray,
        depth_map: np.ndarray | None,
        material_masks: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Apply material-aware enhancements.

        Args:
            image: Input image (H, W, 3)
            depth_map: Optional depth map (H, W)
            material_masks: Dict of material masks

        Returns:
            Enhanced image (H, W, 3)
        """
        # Start with input
        enhanced = image.copy().astype(np.float32)

        # Normalize to [0, 1] if needed
        if enhanced.max() > 1.0:
            enhanced = enhanced / 255.0

        # Apply global tone mapping
        enhanced = self._apply_tone_mapping(enhanced, depth_map)

        # Apply clarity enhancement
        if self.clarity_strength > 0:
            enhanced = self._apply_clarity(enhanced, self.clarity_strength)

        # Apply material-specific enhancements
        if material_masks and self.material_strength > 0:
            enhanced = self._apply_material_enhancements(
                enhanced, material_masks, self.material_strength
            )

        # Convert back to uint8
        enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)

        return enhanced

    def _apply_tone_mapping(
        self, image: np.ndarray, depth_map: np.ndarray | None
    ) -> np.ndarray:
        """Apply depth-aware tone mapping."""
        if depth_map is None or self.enhancement_strength == 0:
            return image

        # Simple zone-based tone mapping
        # Foreground: slight boost
        # Background: slight compression

        foreground = depth_map > 0.7
        background = depth_map < 0.3

        result = image.copy()

        # Boost foreground
        fg_boost = 1.0 + (0.15 * self.enhancement_strength)
        result[foreground] = result[foreground] * fg_boost

        # Compress background slightly
        bg_compress = 1.0 - (0.1 * self.enhancement_strength)
        result[background] = result[background] * bg_compress

        return np.clip(result, 0, 1)

    def _apply_clarity(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Apply clarity enhancement (local contrast)."""
        from scipy.ndimage import gaussian_filter

        # Multi-scale unsharp mask
        blurred = gaussian_filter(image, sigma=2.0, axes=(0, 1))
        detail = image - blurred

        # Apply with strength
        enhanced = image + detail * strength

        return np.clip(enhanced, 0, 1)

    def _apply_material_enhancements(
        self,
        image: np.ndarray,
        material_masks: Dict[str, np.ndarray],
        strength: float,
    ) -> np.ndarray:
        """Apply material-specific enhancements."""
        result = image.copy()

        for material_name, mask in material_masks.items():
            if mask.max() == 0:
                continue

            # Expand mask to 3 channels
            mask_3d = mask[:, :, None]

            if material_name == "wood":
                # Wood: boost warmth and saturation
                result = self._blend(
                    result,
                    self._adjust_warmth(result, 1.1),
                    mask_3d * strength,
                )

            elif material_name == "metal":
                # Metal: boost highlights and contrast
                result = self._blend(
                    result,
                    self._adjust_contrast(result, 1.15),
                    mask_3d * strength,
                )

            elif material_name == "glass":
                # Glass: subtle highlight boost
                result = self._blend(
                    result,
                    result ** 0.95,  # Slight gamma adjustment
                    mask_3d * strength * 0.5,  # Subtle
                )

        return result

    @staticmethod
    def _blend(base: np.ndarray, overlay: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Blend two images using mask."""
        return base * (1 - mask) + overlay * mask

    @staticmethod
    def _adjust_warmth(image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image warmth (boost reds/yellows)."""
        result = image.copy()
        result[:, :, 0] = result[:, :, 0] * factor  # Red channel
        result[:, :, 1] = result[:, :, 1] * (factor * 0.5 + 0.5)  # Green channel
        return np.clip(result, 0, 1)

    @staticmethod
    def _adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image contrast."""
        mean = image.mean()
        result = (image - mean) * factor + mean
        return np.clip(result, 0, 1)
