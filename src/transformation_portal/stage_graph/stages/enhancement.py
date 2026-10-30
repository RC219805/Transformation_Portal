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

_LOW_TEXTURE_PROXY_MAX_SIDE = 1024
_V2_MATERIAL_ADJUSTMENT_MATERIALS = frozenset({"wood", "metal", "glass"})


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
        version: str = "1.1.1",
        output_dtype: Optional[np.dtype] = None,
        tone_low_tex_strength: float = 0.6,
        tone_depth_smoothing: bool = True,
        geometry_recipe: str = "legacy_encoded_crop_stretch_v1",
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
        self.geometry_recipe = geometry_recipe
        self._uses_corrected_geometry()

    def _uses_corrected_geometry(self) -> bool:
        from transformation_portal.lux_depth_v3.geometry import NORMALIZED_GEOMETRY_RECIPE, validate_geometry_recipe

        return validate_geometry_recipe(self.geometry_recipe) == NORMALIZED_GEOMETRY_RECIPE

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

        corrected_geometry = self._uses_corrected_geometry()
        self._native_materials_identity(context)
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
        geometry = None
        if corrected_geometry:
            from transformation_portal.lux_depth_v3.geometry_handoff import GeometryBoundInputs

            geometry = context.get_artifact("geometry_handoff")
            if not isinstance(geometry, GeometryBoundInputs):
                raise ValueError("Corrected enhancement requires a verified geometry handoff")
            material_masks = context.get_artifact("material_masks", {})
            geometry.validate_stage(image, depth_map, material_masks)
            # Material edits can depend on earlier edits' image statistics.
            # Their execution order must agree with the canonical cache input.
            material_masks = dict(sorted(material_masks.items()))
        else:
            material_masks = self._normalize_material_masks(context.get_artifact("material_masks", {}), image)
        material_masks_supplied_count = len(material_masks)
        v2_material_adjustments_applied = self._v2_material_adjustments_would_apply(material_masks)

        start = time.time()

        # Apply enhancements
        enhanced_image = self._enhance_image(
            image,
            depth_map,
            material_masks,
            **({"support_mask": geometry.allowed_pixels()} if geometry is not None else {}),
        )
        if geometry is not None:
            # Support and partial/transparent alpha protect the complete edit,
            # including global clarity and material operations, not just depth.
            allowed = geometry.allowed_pixels()
            maximum = float(np.iinfo(enhanced_image.dtype).max)
            preserved = np.rint(np.clip(image, 0, 1) * maximum).astype(enhanced_image.dtype)
            enhanced_image[~allowed] = preserved[~allowed]

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
                    "material_masks_supplied": material_masks_supplied_count > 0,
                    "material_masks_supplied_count": material_masks_supplied_count,
                    "v2_material_adjustments_applied": v2_material_adjustments_applied,
                    # Deprecated compatibility alias. This now means actual V2
                    # material adjustment, not "material mask keys supplied".
                    "materials_applied": v2_material_adjustments_applied,
                },
            },
            duration_ms=duration_ms,
            metadata={
                "has_depth": depth_map is not None,
                "has_materials": len(material_masks) > 0,
                "processing_ms": duration_ms,
            },
        )

    @staticmethod
    def _normalize_material_masks(
        material_masks: object,
        image: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Drop malformed masks before V2 handoff counting or application."""
        if not isinstance(material_masks, dict) or not isinstance(image, np.ndarray) or image.ndim < 2:
            return {}

        target_shape = tuple(image.shape[:2])
        normalized: Dict[str, np.ndarray] = {}
        for raw_material_name, raw_mask in material_masks.items():
            material_name = str(raw_material_name).strip()
            if not material_name:
                continue
            try:
                mask = np.asarray(raw_mask)
            except (TypeError, ValueError):
                continue
            if mask.ndim != 2 or tuple(mask.shape) != target_shape:
                continue
            if np.iscomplexobj(mask) or not (np.issubdtype(mask.dtype, np.number) or np.issubdtype(mask.dtype, np.bool_)):
                continue
            try:
                mask = mask.astype(np.float32, copy=False)
            except (TypeError, ValueError):
                continue
            if not np.all(np.isfinite(mask)):
                continue
            normalized[material_name] = mask
        return normalized

    def _v2_material_adjustments_would_apply(self, material_masks: Dict[str, np.ndarray]) -> bool:
        """Return true when V2 will apply at least one material adjustment."""
        if not material_masks or self.material_strength <= 0:
            return False

        for material_name, mask in material_masks.items():
            if material_name not in _V2_MATERIAL_ADJUSTMENT_MATERIALS:
                continue
            try:
                if np.asarray(mask).max() != 0:
                    return True
            except (TypeError, ValueError):
                continue
        return False

    def get_cache_key(self, context: StageContext) -> str:
        """Generate cache key based on all inputs."""
        corrected_geometry = self._uses_corrected_geometry()
        native_materials = self._native_materials_identity(context)
        # Get input image
        image = context.get_artifact("image")
        if image is None:
            if corrected_geometry:
                raise ValueError("Corrected enhancement requires its verified photographic image before cache lookup")
            return "no_image"

        if corrected_geometry:
            from transformation_portal.ingest.canonical_json import canonicalize_json
            from transformation_portal.lux_depth_v3.geometry_handoff import GeometryBoundInputs, array_identity

            geometry = context.get_artifact("geometry_handoff")
            if not isinstance(geometry, GeometryBoundInputs):
                raise ValueError("Corrected enhancement requires a verified geometry handoff")
            geometry.validate_stage(image, context.get_artifact("depth_map"), context.get_artifact("material_masks", {}))
            payload = {
                "recipe": self.geometry_recipe,
                "handoff_identity_sha256": geometry.identity_sha256,
                "image": array_identity(image),
                "depth_map": array_identity(geometry.depth_map),
                "support": array_identity(geometry.support),
                "material_masks": {name: array_identity(mask) for name, mask in sorted(geometry.masks.items())},
                "material_execution_order": "sorted_name_v1",
                "alpha": None if geometry.alpha is None else array_identity(geometry.alpha),
                "stage_version": self.version,
                "output_dtype": np.dtype(self.output_dtype).str,
                "strengths": [self.enhancement_strength, self.clarity_strength, self.material_strength],
                "tone_low_tex_strength": self.tone_low_tex_strength,
                "tone_depth_smoothing": self.tone_depth_smoothing,
            }
            if native_materials is not None:
                payload["native_materials"] = native_materials
            return "enhance_geometry_" + hashlib.sha256(canonicalize_json(payload)).hexdigest()
        return self._legacy_cache_key(context)

    def _native_materials_identity(self, context: StageContext) -> dict | None:
        """Bind validated native execution meaning and its actual output master."""
        native = context.get_artifact("native_materials")
        if native is None:
            if self.version == "1.4.0":
                raise ValueError("Native Materials enhancement requires its validated execution receipt")
            return None
        from transformation_portal.core.image_artifact import ImageMaster
        from transformation_portal.ingest.canonical_json import canonicalize_json
        from transformation_portal.lux_depth_v3.geometry_handoff import GeometryBoundInputs
        from transformation_portal.lux_depth_v3.native_materials import revalidate_native_materials

        if not self._uses_corrected_geometry() or self.version != "1.4.0":
            raise ValueError("Native Materials requires its versioned corrected enhancement stage")
        if not isinstance(native, dict) or set(native) != {"binding", "receipt", "master"}:
            raise ValueError("Native Materials enhancement requires its complete native context")
        binding, receipt, master = native["binding"], native["receipt"], native["master"]
        revalidate_native_materials(binding)
        geometry = context.get_artifact("geometry_handoff")
        if not isinstance(master, ImageMaster) or not isinstance(geometry, GeometryBoundInputs):
            raise ValueError("Native Materials enhancement requires its exact master and native geometry")
        sources = (
            [source for source in binding["sources"] if source["path"] == receipt.get("selected_source_path")]
            if isinstance(receipt, dict)
            else []
        )
        if len(sources) != 1 or (
            sources[0]["sha256"] != master.source_sha256
            or sources[0]["source_bit_depth"] != master.source_bit_depth
            or sources[0]["master_sha256"] != receipt.get("input_master_sha256")
        ):
            raise ValueError("Native Materials receipt differs from its selected frozen source")
        if not isinstance(receipt, dict) or (
            receipt.get("schema") != "tp.lux.native_materials.execution.v1"
            or receipt.get("recipe") != "native_sdr_v1"
            or receipt.get("binding_sha256") != binding["binding_sha256"]
            or receipt.get("source_sha256") != master.source_sha256
            or master.source_sha256 != geometry.source_sha256
            or receipt.get("output_master_sha256") != master.content_hash()
            or receipt.get("support_sha256") != hashlib.sha256(geometry.support.tobytes(order="C")).hexdigest()
        ):
            raise ValueError("Native Materials receipt differs from its actual native execution")
        execution = receipt.get("execution")
        response_plan = execution.get("response_plan") if isinstance(execution, dict) else None
        if (
            not isinstance(execution, dict)
            or not isinstance(response_plan, dict)
            or (
                execution.get("schema") != "tp.materials.execution.v1"
                or execution.get("input_master_sha256") != receipt["input_master_sha256"]
                or execution.get("output_master_sha256") != receipt["output_master_sha256"]
                or execution.get("evidence_sha256") != receipt.get("effective_evidence_sha256")
                or execution.get("operations_sha256") != binding["operations_sha256"]
                or execution.get("plan_sha256") != hashlib.sha256(canonicalize_json(response_plan)).hexdigest()
                or response_plan.get("source_sha256") != master.source_sha256
                or response_plan.get("master_sha256") != receipt["input_master_sha256"]
                or response_plan.get("evidence_sha256") != receipt.get("effective_evidence_sha256")
                or response_plan.get("operations_sha256") != binding["operations_sha256"]
                or response_plan.get("policy") != binding["policy"]
            )
        ):
            raise ValueError("Native Materials execution receipt differs from its frozen policy and actual plan")
        return {"binding_sha256": binding["binding_sha256"], "execution_receipt": receipt}

    def _legacy_cache_key(self, context: StageContext) -> str:
        """Keep historical key bytes unchanged for the legacy recipe."""
        image = context.get_artifact("image")

        # Hash image
        image_hash = hashlib.sha256(image.tobytes()).hexdigest()[:16]

        # Hash depth map
        depth_map = context.get_artifact("depth_map")
        depth_hash = ""
        if depth_map is not None:
            depth_hash = hashlib.sha256(depth_map.tobytes()).hexdigest()[:8]

        # Hash material masks
        material_masks = self._normalize_material_masks(
            context.get_artifact("material_masks", {}),
            image,
        )
        material_hash = ""
        if material_masks:
            # Sort keys for deterministic order
            sorted_keys = sorted(material_masks.keys())
            material_parts: list[bytes] = []
            for material_name in sorted_keys:
                material_name_bytes = material_name.encode("utf-8")
                material_parts.extend(
                    (
                        len(material_name_bytes).to_bytes(4, byteorder="big"),
                        material_name_bytes,
                        material_masks[material_name].tobytes(),
                    )
                )
            material_bytes = b"".join(material_parts)
            material_hash = hashlib.sha256(material_bytes).hexdigest()[:8]

        # Configuration
        shape_key = "x".join(str(side) for side in image.shape)
        config_str = (
            # Callers can supply their own stage version; bind the corrected
            # normalization semantics and precision independently of that value.
            f"dtype_normalization_v2_{image.dtype}_{shape_key}_{np.dtype(self.output_dtype)}_"
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
        *,
        support_mask: np.ndarray | None = None,
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
        # Integer encoding is determined by dtype, including near-black frames
        # whose highest sample is one. Float callers retain the existing
        # normalized [0, 1] contract (and legacy 0..255 float compatibility).
        input_dtype = image.dtype

        # Start with input - convert to float32 [0, 1] for processing
        enhanced = image.copy().astype(np.float32)

        # Normalize to [0, 1] preserving precision
        if np.issubdtype(input_dtype, np.integer):
            enhanced = enhanced / float(np.iinfo(input_dtype).max)
        elif enhanced.max() > 1.0:
            enhanced = enhanced / 255.0

        # Apply global tone mapping
        enhanced = self._apply_tone_mapping(
            enhanced, depth_map, **({"support_mask": support_mask} if support_mask is not None else {})
        )

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

    def _apply_tone_mapping(
        self, image: np.ndarray, depth_map: np.ndarray | None, *, support_mask: np.ndarray | None = None
    ) -> np.ndarray:
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
        if support_mask is not None:
            if support_mask.dtype != np.bool_ or support_mask.shape != (image_height, image_width):
                raise ValueError("Tone support must be an exact native boolean mask")
            if depth_map.shape != support_mask.shape or not np.isfinite(depth_map[support_mask]).all():
                raise ValueError("Supported tone depth must be finite on the exact native grid")
            if not support_mask.any():
                return image
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
        depth_p75 = float(np.percentile(depth_map if support_mask is None else depth_map[support_mask], 75))
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
        depth_for_tone = (
            self._lowpass_depth_for_tone(depth_map, **({"support_mask": support_mask} if support_mask is not None else {}))
            if self.tone_depth_smoothing
            else depth_map
        )

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
        if support_mask is not None:
            result = np.where(support_mask[..., None], result, image)

        return np.clip(result, 0, 1)

    def _lowpass_depth_for_tone(self, depth_map: np.ndarray, *, support_mask: np.ndarray | None = None) -> np.ndarray:
        """Low-pass the depth map before it drives per-pixel luminance.

        The smoothing scale still tracks image size so we suppress pixel-scale
        striping and tile discontinuities from upstream depth backends, but the
        effective blur radius is bounded to keep runtime predictable on large
        frames and to avoid over-smoothing scene-scale depth intent.
        """
        from scipy.ndimage import gaussian_filter

        h, w = depth_map.shape[:2]
        sigma = float(np.clip(max(h, w) / 1024.0, 2.0, 8.0))
        if support_mask is not None:
            weights = gaussian_filter(support_mask.astype(np.float32), sigma=sigma)
            numerator = gaussian_filter(np.where(support_mask, depth_map, 0).astype(np.float32), sigma=sigma)
            smoothed = np.zeros(depth_map.shape, np.float32)
            np.divide(numerator, weights, out=smoothed, where=weights > 0)
            return smoothed
        return gaussian_filter(depth_map.astype(np.float32, copy=False), sigma=sigma)

    @staticmethod
    def _low_texture_weight(image: np.ndarray) -> np.ndarray:
        """Per-pixel `low_tex` weight in [0, 1] (1 = flat region).

        Uses a bounded luminance proxy for large frames, then resizes the
        weight back to full resolution. This keeps the default guard from
        adding an unbounded full-frame convolution and percentile on 4K/8K
        inputs.
        """
        from scipy.ndimage import gaussian_gradient_magnitude

        # Rec. 709 luminance; image is float32 in [0, 1].
        if image.ndim == 3 and image.shape[-1] >= 3:
            luma = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
        else:
            luma = image if image.ndim == 2 else image[..., 0]
        luma = luma.astype(np.float32, copy=False)

        h, w = luma.shape[:2]
        max_side = max(h, w)
        resized_proxy = False
        proxy_scale = 1.0
        luma_proxy: np.ndarray = luma
        if max_side > _LOW_TEXTURE_PROXY_MAX_SIDE:
            import cv2

            proxy_scale = _LOW_TEXTURE_PROXY_MAX_SIDE / float(max_side)
            proxy_h = max(1, int(round(h * proxy_scale)))
            proxy_w = max(1, int(round(w * proxy_scale)))
            luma_proxy = cv2.resize(luma, (proxy_w, proxy_h), interpolation=cv2.INTER_AREA)
            resized_proxy = True

        # Scale the local-gradient sigma into proxy pixels. Keep a floor of 1
        # proxy pixel so highly downsampled large frames still get stable local
        # gradient estimates instead of alias-sensitive single-pixel checks.
        sigma = max(1.0, 4.0 * proxy_scale)
        grad = gaussian_gradient_magnitude(luma_proxy, sigma=sigma)
        p95 = float(np.percentile(grad, 95))
        if p95 <= 1e-6:
            # Entirely flat frame → guard fires everywhere.
            low_tex_proxy = np.ones_like(grad, dtype=np.float32)
        else:
            normalized = np.clip(grad / p95, 0.0, 1.0)
            low_tex_proxy = 1.0 - normalized

        if resized_proxy:
            import cv2

            low_tex = cv2.resize(low_tex_proxy.astype(np.float32, copy=False), (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            low_tex = low_tex_proxy

        return np.clip(low_tex, 0.0, 1.0).astype(np.float32, copy=False)

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
