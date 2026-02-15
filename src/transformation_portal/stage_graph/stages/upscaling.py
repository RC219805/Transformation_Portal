"""
Upscaling stage.

Integrates with Phase 4 upscaling backend (UpscalerRegistry).
Supports bicubic (always available) and Real-ESRGAN (ML, disabled due to CVE-2024-27763).
"""

from __future__ import annotations

import hashlib

import numpy as np

from transformation_portal.upscaling import UpscalerRegistry

from ..stage import Stage, StageContext, StageResult, StageStatus


class UpscalingStage(Stage):
    """
    Image upscaling stage.

    Uses UpscalerRegistry to select backend with graceful fallback to bicubic.
    Preserves dtype (float32 → float32, uint8 → uint8) for Phase 3 compatibility.

    Backends:
        - bicubic: Always available, commercial-safe (OpenCV)
        - realesrgan: Disabled due to CVE-2024-27763

    Performance:
        - Bicubic: ~100-200 images/hour for 4K→8K upscaling
        - Memory: ~50MB per image (bicubic)

    Usage:
        >>> stage = UpscalingStage(backend="bicubic", scale_factor=2.0)
        >>> context = StageContext(artifacts={"image": image_array})
        >>> result = stage.execute(context)
        >>> upscaled = result.artifacts["upscaled_image"]
    """

    def __init__(
        self,
        scale_factor: float = 2.0,
        backend: str = "bicubic",
        allow_fallback: bool = True,
        version: str = "1.0.0",
    ):
        """
        Initialize upscaling stage.

        Args:
            scale_factor: Upscaling factor (1.0-4.0)
            backend: Backend name ("bicubic", "realesrgan", "default")
            allow_fallback: If True, fallback to bicubic on error
            version: Stage version for cache invalidation
        """
        super().__init__(name="upscaling", version=version)
        self.scale_factor = scale_factor
        self.backend = backend
        self.allow_fallback = allow_fallback
        self._upscaler = None
        self._registry = UpscalerRegistry()

    def get_dependencies(self) -> list:
        """
        Stage dependencies.

        Returns:
            ["enhancement"] for backward-compatible execution order.

        Note:
            - The declared dependency ensures that, in default/Golden Path
              pipelines, upscaling runs after enhancement.
            - The compute method still supports multiple artifact types
              (enhanced_image, image, depth_map); this dependency only
              constrains default ordering, not input flexibility.
        """
        return ["enhancement"]

    def compute(self, context: StageContext) -> StageResult:
        """
        Upscale image using backend registry.

        Expected context artifacts:
        - image: Input image as numpy array (H, W, 3), uint8 or float32
        - enhanced_image: Optional enhanced image (takes priority)
        - depth_map: Optional depth map (H, W), float32 [0,1]

        Output artifacts:
        - upscaled_image: Upscaled image (H*scale, W*scale, 3)
        - upscale_metadata: Dict with backend info, timing, shapes

        Note:
            Preserves dtype (float32 → float32, uint8 → uint8).
            Critical for Phase 3 depth pipeline compatibility.
        """
        import time

        # Get input - try multiple artifact names for compatibility
        image = context.get_artifact("enhanced_image")
        if image is None:
            image = context.get_artifact("image")
        if image is None:
            image = context.get_artifact("depth_map")  # Support depth map upscaling

        if image is None:
            return StageResult(
                stage_name=self.name,
                stage_version=self.version,
                status=StageStatus.FAILED,
                error="Missing 'image', 'enhanced_image', or 'depth_map' artifact in context",
            )

        # Validate input
        if not isinstance(image, np.ndarray):
            return StageResult(
                stage_name=self.name,
                stage_version=self.version,
                status=StageStatus.FAILED,
                error=f"Invalid image type: {type(image)}. Expected numpy array.",
            )

        # Handle grayscale depth maps (H, W) → expand to (H, W, 3)
        original_was_grayscale = False
        original_shape = image.shape  # Capture original shape before expansion
        if image.ndim == 2:
            original_was_grayscale = True
            image = np.stack([image] * 3, axis=-1)  # Duplicate to 3 channels
            self.logger.debug(f"Expanded grayscale image {image.shape[:-1]} to RGB for upscaling")

        # Skip if scale factor is 1.0
        if abs(self.scale_factor - 1.0) < 0.01:
            # Return original (remove expansion if it was grayscale)
            output_image = image[:, :, 0] if original_was_grayscale else image
            return StageResult(
                stage_name=self.name,
                stage_version=self.version,
                status=StageStatus.SKIPPED,
                artifacts={
                    "upscaled_image": output_image,
                    "upscale_metadata": {
                        "scale_factor": 1.0,
                        "skipped": True,
                        "reason": "scale_factor=1.0",
                    },
                },
            )

        start = time.time()

        # Lazy load upscaler using registry
        if self._upscaler is None:
            self._load_upscaler(context.device)

        # Track actual backend used (may differ due to fallback)
        actual_backend = self._upscaler.name

        # Upscale using backend
        try:
            upscaled_image = self._upscaler.upscale(image, scale_factor=self.scale_factor)

            # Convert back to grayscale if input was grayscale
            if original_was_grayscale:
                upscaled_image = upscaled_image[:, :, 0]  # Extract single channel

            duration_ms = (time.time() - start) * 1000

            return StageResult(
                stage_name=self.name,
                stage_version=self.version,
                status=StageStatus.COMPLETED,
                artifacts={
                    "upscaled_image": upscaled_image,
                    "upscale_metadata": {
                        "scale_factor": self.scale_factor,
                        "backend_requested": self.backend,
                        "backend_used": actual_backend,
                        "input_shape": original_shape,  # Original shape before grayscale expansion
                        "output_shape": upscaled_image.shape,
                        "input_dtype": str(image.dtype),
                        "output_dtype": str(upscaled_image.dtype),
                        "was_grayscale": original_was_grayscale,
                    },
                },
                duration_ms=duration_ms,
                metadata={
                    "backend": actual_backend,
                    "scale_factor": self.scale_factor,
                    "upscale_ms": duration_ms,
                },
            )

        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            self.logger.error(f"Upscaling failed with {actual_backend}: {e}")

            return StageResult(
                stage_name=self.name,
                stage_version=self.version,
                status=StageStatus.FAILED,
                error=f"Upscaling failed: {e}",
                duration_ms=duration_ms,
                metadata={
                    "backend": actual_backend,
                    "scale_factor": self.scale_factor,
                },
            )

    def get_cache_key(self, context: StageContext) -> str:
        """
        Generate cache key based on input and configuration.

        Args:
            context: Execution context with artifacts

        Returns:
            Cache key string including image hash and config
        """
        # Get input image (try multiple artifact names)
        image = context.get_artifact("enhanced_image")
        if image is None:
            image = context.get_artifact("image")
        if image is None:
            image = context.get_artifact("depth_map")

        if image is None:
            return "no_image"

        # Hash image content
        image_hash = hashlib.sha256(image.tobytes()).hexdigest()[:16]

        # Configuration string
        config_str = f"{self.backend}_{self.scale_factor:.1f}_{self.version}"

        return f"upscale_{config_str}_{image_hash}"

    def _load_upscaler(self, device: str):
        """
        Load upscaling backend using registry.

        Args:
            device: Device to use (cpu, cuda, mps)

        Note:
            Uses UpscalerRegistry.get() with fallback behavior controlled
            by self.allow_fallback flag.
        """
        try:
            self._upscaler = self._registry.get(
                backend_name=self.backend,
                device=device,
                fallback_to_bicubic=self.allow_fallback,
            )

            backend_name = self._upscaler.name
            requires_ml = self._upscaler.requires_ml

            self.logger.info(
                f"Loaded upscaler backend: {backend_name} "
                f"(device={device}, requires_ml={requires_ml}, "
                f"fallback={'enabled' if self.allow_fallback else 'disabled'})"
            )

        except Exception as e:
            self.logger.error(f"Failed to load upscaler backend '{self.backend}': {e}")
            raise
