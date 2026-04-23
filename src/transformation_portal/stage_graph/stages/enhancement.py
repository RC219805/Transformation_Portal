"""
Image enhancement stage.

Applies material-aware enhancements including:
- Tone mapping
- Clarity enhancement
- Material-specific processing
- Atmospheric effects
"""

from __future__ import annotations

import hashlib
from typing import Dict, Optional

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
        version: str = "1.1.0",
        output_dtype: Optional[np.dtype] = None,
        tone_low_tex_strength: float = 0.6,
        tone_depth_smoothing: bool = True,
    ):
        """
        Initialize enhancement stage.

        Args:
            enhancement_strength: Global enhancement strength [0, 1]
            clarity_strength: Clarity enhancement strength [0, 1]
            material_strength: Material enhancement strength [0, 1]
            version: Stage version for cache invalidation
            output_dtype: Output dtype (np.uint8 or np.uint16). If None, defaults to uint8.
            tone_low_tex_strength: How hard to attenuate the depth-driven
                luminance adjustment on low-gradient regions (sky, water,
                smooth walls). 0.0 disables the guard; 1.0 flattens the
                adjustment to unity there. Defaults to 0.6.
            tone_depth_smoothing: When True, low-pass the depth map before
                it drives the per-pixel luminance multiplier. Prevents
                high-frequency striping in the depth map from projecting
                into visible vertical luminance bands.
        """
        super().__init__(name="enhancement", version=version)
        self.enhancement_strength = enhancement_strength
        self.clarity_strength = clarity_strength
        self.material_strength = material_strength
        self.output_dtype = output_dtype if output_dtype is not None else np.dtype("uint8")
        self.tone_low_tex_strength = float(np.clip(tone_low_tex_strength, 0.0, 1.0))
        self.tone_depth_smoothing = bool(tone_depth_smoothing)

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
        enhanced_image = self._enhance_image(image, depth_map, material_masks)

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
            material_bytes = b"".join(material_masks[k].tobytes() for k in sorted_keys)
            material_hash = hashlib.sha256(material_bytes).hexdigest()[:8]

        # Configuration
        config_str = (
            f"{self.enhancement_strength:.2f}_"
            f"{self.clarity_strength:.2f}_"
            f"{self.material_strength:.2f}_"
            f"{self.tone_low_tex_strength:.2f}_"
            f"{int(self.tone_depth_smoothing)}_"
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
            image: Input image (H, W, 3) - uint8 or uint16
            depth_map: Optional depth map (H, W)
            material_masks: Dict of material masks

        Returns:
            Enhanced image (H, W, 3) - same dtype as input (controlled by output_dtype)
        """
        # Detect input range for normalization
        input_dtype = image.dtype
        is_16bit = input_dtype == np.uint16
        max_value = 65535.0 if is_16bit else 255.0

        # Start with input - convert to float32 [0, 1] for processing
        enhanced = image.copy().astype(np.float32)

        # Normalize to [0, 1] preserving precision
        if enhanced.max() > 1.0:
            enhanced = enhanced / max_value

        # Apply global tone mapping
        enhanced = self._apply_tone_mapping(enhanced, depth_map)

        # Apply clarity enhancement
        if self.clarity_strength > 0:
            enhanced = self._apply_clarity(enhanced, self.clarity_strength)

        # Apply material-specific enhancements
        if material_masks and self.material_strength > 0:
            enhanced = self._apply_material_enhancements(enhanced, material_masks, self.material_strength)

        # Convert back to original dtype (uint8 or uint16)
        # Use output_dtype if specified, otherwise match input
        target_dtype = self.output_dtype if hasattr(self, "output_dtype") else input_dtype

        if target_dtype == np.uint16:
            # 16-bit output
            enhanced = np.clip(enhanced * 65535.0, 0, 65535).astype(np.uint16)
        else:
            # 8-bit output (default)
            enhanced = np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)

        return enhanced

    def _apply_tone_mapping(self, image: np.ndarray, depth_map: np.ndarray | None) -> np.ndarray:
        """Apply depth-aware tone mapping.

        CRITICAL: Depth maps from Depth Pro use normalized depth representation:
        - HIGH depth values (closer to 1.0) = FAR objects (sky, distant background)
        - LOW depth values (closer to 0.0) = NEAR objects (foreground architecture, people)

        Note: This is sometimes called "inverse depth" in computer vision because
        it's proportional to 1/distance, but we refer to it as "normalized depth"
        to avoid confusion.

        After p01-p99 normalization to [0,1]:
        - Distribution is heavily skewed toward low values (median ~0.18-0.25)
        - Far objects (sky) are typically 0.4-1.0
        - Near objects (architecture) are typically 0.0-0.2

        For luxury real estate rendering:
        - NEAR objects (LOW depth values) should be enhanced (boosted)
        - FAR objects (HIGH depth values) should be subtle (compressed)

        Uses adaptive depth-based adjustment centered on actual data distribution.
        """
        if depth_map is None or self.enhancement_strength == 0:
            return image

        image_height, image_width = image.shape[:2]
        if depth_map.shape != (image_height, image_width):
            self.logger.warning(
                "Depth map shape %s does not match image shape %s; resizing depth map for tone mapping",
                depth_map.shape,
                (image_height, image_width),
            )
            depth_map = self._resize_depth_map(depth_map, (image_height, image_width))

        # Percentiles are computed on the RAW depth map so the decision about
        # "near vs. far" stays aligned with the scene, not with our smoothing
        # kernel.
        depth_p75 = float(np.percentile(depth_map, 75))
        center_point = depth_p75

        # Decouple depth high-frequencies from per-pixel luminance.
        #
        # The multiplier below (`adjustment`) is broadcast across RGB, so any
        # fine vertical structure in depth lands as a vertical luminance band
        # in the output. That is exactly how "paneling" can appear on sky
        # and water in aerials even though V2 itself is full-frame. We low-
        # pass the depth map before it feeds the curve; the global tonal
        # intent (near vs. far) is preserved because the low-pass kernel is
        # much smaller than scene-scale depth variation.
        depth_for_tone = self._lowpass_depth_for_tone(depth_map) if self.tone_depth_smoothing else depth_map

        depth_normalized = (depth_for_tone - center_point) / (1.0 - center_point + 1e-6)
        depth_normalized = np.clip(depth_normalized, -1.0, 1.0)

        # Apply smooth sigmoid for gradual transition
        depth_factor = np.tanh(depth_normalized * 2.0)  # Sharper curve

        # Calculate adjustment factor
        # depth_factor = -1 (near) → adjustment = 1.12 (boost)
        # depth_factor = 0 (mid) → adjustment = 1.0 (neutral)
        # depth_factor = +1 (far) → adjustment = 0.92 (compress)
        max_boost = 0.12 * self.enhancement_strength
        max_compress = -0.08 * self.enhancement_strength

        # Map depth_factor [-1, +1] to adjustment [1+max_compress, 1+max_boost]
        # depth_factor = -1 (near) → adjustment = 1 + max_boost (boost)
        # depth_factor = 0 (mid) → adjustment = 1.0 (neutral)
        # depth_factor = +1 (far) → adjustment = 1 + max_compress (compress)
        if depth_factor.ndim == 0:  # Scalar
            if depth_factor < 0:
                # Near objects: boost
                adjustment = 1.0 - depth_factor * max_boost
            else:
                # Far objects: compress
                adjustment = 1.0 - depth_factor * abs(max_compress)
        else:  # Array
            adjustment = np.ones_like(depth_factor)
            near_mask = depth_factor < 0
            far_mask = depth_factor >= 0
            adjustment[near_mask] = 1.0 - depth_factor[near_mask] * max_boost
            adjustment[far_mask] = 1.0 - depth_factor[far_mask] * abs(max_compress)

        adjustment = np.clip(adjustment, 1.0 + max_compress, 1.0 + max_boost)

        # Low-gradient-energy guard.
        #
        # On near-flat regions (clear sky, still water, smooth walls) even a
        # small residual variation in the multiplier reads as a visible
        # luminance step to the eye, because there is no scene texture to
        # mask it. We measure local image gradient energy and pull the
        # multiplier toward 1.0 where texture is absent. Textured regions
        # (foliage, stonework, railings) see essentially no change.
        if adjustment.ndim > 0 and self.tone_low_tex_strength > 0.0:
            low_tex = self._low_texture_weight(image)
            # Move `adjustment` toward 1.0 in low-texture areas.
            attenuation = 1.0 - self.tone_low_tex_strength * low_tex
            adjustment = 1.0 + (adjustment - 1.0) * attenuation

        # Apply adjustment (broadcast to RGB)
        result = image * adjustment[:, :, np.newaxis]

        return np.clip(result, 0, 1)

    def _lowpass_depth_for_tone(self, depth_map: np.ndarray) -> np.ndarray:
        """Low-pass the depth map before it drives per-pixel luminance.

        The smoothing scale still tracks image size so we suppress pixel-scale
        striping and tile discontinuities from upstream depth backends, but the
        effective blur radius is bounded to keep runtime predictable on large
        frames and to avoid over-smoothing scene-scale depth intent.
        """
        from scipy.ndimage import gaussian_filter

        h, w = depth_map.shape[:2]
        sigma = float(np.clip(max(h, w) / 1024.0, 2.0, 8.0))
        return gaussian_filter(depth_map.astype(np.float32, copy=False), sigma=sigma)

    @staticmethod
    def _low_texture_weight(image: np.ndarray) -> np.ndarray:
        """Per-pixel `low_tex` weight in [0, 1] (1 = flat region).

        Uses a coarse Gaussian gradient magnitude on the luminance channel.
        Normalized by the 95th percentile so a single bright edge does not
        suppress the guard everywhere.
        """
        from scipy.ndimage import gaussian_gradient_magnitude

        # Rec. 709 luminance; image is float32 in [0, 1].
        if image.ndim == 3 and image.shape[-1] >= 3:
            luma = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
        else:
            luma = image if image.ndim == 2 else image[..., 0]
        luma = luma.astype(np.float32, copy=False)

        grad = gaussian_gradient_magnitude(luma, sigma=4.0)
        p95 = float(np.percentile(grad, 95))
        if p95 <= 1e-6:
            # Entirely flat frame → guard fires everywhere.
            return np.ones_like(grad, dtype=np.float32)
        normalized = np.clip(grad / p95, 0.0, 1.0)
        low_tex = 1.0 - normalized
        return low_tex.astype(np.float32, copy=False)

    @staticmethod
    def _resize_depth_map(depth_map: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        """Resize depth map to match target (height, width)."""
        from PIL import Image

        target_height, target_width = target_shape
        depth_array = np.asarray(depth_map, dtype=np.float32)
        if depth_array.ndim != 2:
            raise ValueError(f"Depth map must be 2D (H, W), got shape {depth_array.shape}")

        depth_image = Image.fromarray(depth_array, mode="F")
        resized = depth_image.resize((target_width, target_height), Image.Resampling.BILINEAR)
        return np.asarray(resized, dtype=np.float32)

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
                    result**0.95,  # Slight gamma adjustment
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
