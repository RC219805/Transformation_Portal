"""DepthCrafter temporal depth backend for video consistency.

Implements ADR-026 multi-model ensemble third backend for temporal consistency
in video workflows. DepthCrafter provides temporally-coherent depth estimation
across video frames using an exponential moving average (EMA) temporal filter.

When the DepthCrafter model checkpoint is unavailable, the backend falls back
to a synthetic luminance-based depth with temporal smoothing applied, ensuring
the ensemble always has a temporal-consistency signal.

License: Apache 2.0 (commercial use permitted).

See ADR-026 Section 4.1 for ensemble specifications.
"""

from __future__ import annotations

import hashlib
import logging
from collections import deque
from typing import TYPE_CHECKING, Deque, Optional, Union

import numpy as np
from PIL import Image

from .protocol import DepthResult, LicenseType, StatefulBackend

if TYPE_CHECKING:
    from ...lux_depth_v3.config import EnhanceConfig

logger = logging.getLogger(__name__)

__version__ = "0.1.0"

# Default temporal filter alpha (EMA smoothing factor).
# Lower values = stronger smoothing (more temporal consistency).
_DEFAULT_TEMPORAL_ALPHA = 0.3

# Maximum number of past frames to keep in the temporal buffer.
_MAX_TEMPORAL_BUFFER = 30


class DepthCrafterBackend:
    """DepthCrafter temporal depth backend implementing DepthBackend protocol.

    Provides temporally-coherent depth estimation for video workflows.
    When the DepthCrafter model checkpoint is unavailable, falls back to
    synthetic depth with temporal EMA filtering.

    The temporal filter uses an exponential moving average:
        smoothed_t = alpha * depth_t + (1 - alpha) * smoothed_{t-1}

    This eliminates inter-frame flicker in depth maps for video pipelines.

    Attributes:
        name: Backend identifier ("depthcrafter").
        license_type: COMMERCIAL (Apache 2.0 license).
        requires_checkpoint: True (DepthCrafter v1 checkpoint).

    Example:
        >>> config = EnhanceConfig()
        >>> backend = DepthCrafterBackend(config)
        >>> result = backend.compute(frame_image)
        >>> print(f"Temporal depth: {result.depth_map.shape}")
    """

    # Backend protocol attributes
    name = "depthcrafter"
    license_type = LicenseType.COMMERCIAL  # Apache 2.0
    requires_checkpoint = True
    _MODEL_INFERENCE_IMPLEMENTED = False

    def __init__(
        self,
        config: Optional["EnhanceConfig"] = None,
        temporal_alpha: float = _DEFAULT_TEMPORAL_ALPHA,
        max_buffer_size: int = _MAX_TEMPORAL_BUFFER,
    ):
        """Initialize DepthCrafter backend.

        Args:
            config: EnhanceConfig for device settings and checkpoint path.
            temporal_alpha: EMA smoothing factor (0.0-1.0). Lower = smoother.
            max_buffer_size: Maximum frames to keep in temporal buffer.
        """
        self._config = config
        self._temporal_alpha = max(0.0, min(1.0, temporal_alpha))
        self._max_buffer_size = max_buffer_size

        # Temporal state
        self._temporal_buffer: Deque[np.ndarray] = deque(maxlen=max_buffer_size)
        self._ema_state: Optional[np.ndarray] = None

        # Checkpoint availability (lazy-checked)
        self._checkpoint_available: Optional[bool] = None
        self._model_inference_warning_emitted = False
        self._model = None

        logger.debug(
            "DepthCrafterBackend initialized (temporal_alpha=%.2f, buffer=%d)",
            self._temporal_alpha,
            self._max_buffer_size,
        )

    def ensure_available(self) -> None:
        """Ensure backend dependencies are available.

        DepthCrafter falls back to synthetic+temporal when checkpoint is
        unavailable, so this method does not raise when the checkpoint is
        missing. It logs a warning instead.

        Raises:
            ImportError: If numpy or Pillow are missing (should never happen).
        """
        if self._checkpoint_available is None:
            self._checkpoint_available = self._check_checkpoint()

        if not self._checkpoint_available:
            logger.warning(
                "DepthCrafter checkpoint unavailable. "
                "Falling back to synthetic depth with temporal smoothing. "
                "To enable full DepthCrafter, place checkpoint at "
                "checkpoints/depthcrafter_v1.pt"
            )

    @classmethod
    def required_packages(cls) -> list[str]:
        """Return required import module names for DepthCrafter.

        The DepthCrafter fallback mode requires only numpy and Pillow
        (already project dependencies). Full model inference would
        additionally require torch and diffusers.

        Returns:
            Empty list (fallback mode has no extra deps beyond numpy/PIL).
        """
        return []

    @staticmethod
    def _normalize_numpy_input(image: np.ndarray) -> np.ndarray:
        """Normalize numpy input to uint8 image space."""
        if image.dtype in (np.float16, np.float32, np.float64):
            image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
            if image.size and float(np.max(image)) <= 1.0:
                return (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
            return np.clip(image, 0.0, 255.0).astype(np.uint8)

        if image.dtype != np.uint8:
            return np.clip(image, 0, 255).astype(np.uint8)

        return image

    @classmethod
    def _to_rgb_uint8_array(cls, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """Convert PIL/numpy image to an RGB uint8 numpy array."""
        if isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            pil_image = Image.fromarray(cls._normalize_numpy_input(image))
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
        return np.asarray(pil_image, dtype=np.uint8)

    def _checkpoint_path(self) -> str:
        """Return checkpoint path from config or default location."""
        checkpoint_path = None
        if self._config is not None:
            checkpoint_path = getattr(self._config, "depthcrafter_checkpoint_path", None)

        if checkpoint_path is None:
            checkpoint_path = "checkpoints/depthcrafter_v1.pt"

        return str(checkpoint_path)

    def compute(
        self,
        image: Union[Image.Image, np.ndarray],
        device: Optional[str] = None,
    ) -> DepthResult:
        """Estimate temporally-coherent depth from image.

        For video sequences, call this method once per frame in order.
        The temporal EMA filter smooths depth across frames to reduce
        inter-frame flicker.

        Args:
            image: Input image as PIL Image or numpy array (H, W, 3).
            device: Optional device override (cpu, cuda, mps).

        Returns:
            DepthResult with temporally-smoothed depth map.
        """
        # Normalize input image to RGB uint8.
        img_array = self._to_rgb_uint8_array(image)
        pil_image = Image.fromarray(img_array)

        # Compute raw depth (checkpoint model or synthetic fallback)
        raw_depth = self._compute_raw_depth(pil_image, device)

        # Apply temporal EMA filter
        smoothed_depth = self._apply_temporal_filter(raw_depth)

        # Track in temporal buffer
        self._temporal_buffer.append(smoothed_depth.copy())

        checkpoint_used = self._checkpoint_available and self._model is not None
        is_synthetic = not checkpoint_used
        metadata = {
            "backend": self.name,
            "version": __version__,
            "temporal_alpha": self._temporal_alpha,
            "temporal_buffer_length": len(self._temporal_buffer),
            "checkpoint_used": checkpoint_used,
            "fallback_mode": is_synthetic,
            "synthetic": is_synthetic,
            "confidence": 0.0 if is_synthetic else 1.0,
            "availability": "missing_checkpoint" if is_synthetic else "available",
        }

        return DepthResult(
            depth_map=smoothed_depth,
            original_image=img_array,
            metadata=metadata,
            depth_units="relative",
            backend_id=self.name,
            device=device or "cpu",
            dtype="float32",
            input_size=(img_array.shape[0], img_array.shape[1]),
        )

    def _compute_raw_depth(
        self,
        pil_image: Image.Image,
        device: Optional[str],
    ) -> np.ndarray:
        """Compute raw depth map using model or synthetic fallback.

        Args:
            pil_image: Input image as PIL.
            device: Device for inference.

        Returns:
            Raw depth map (H, W) float32, values in [0, 1].
        """
        # Check checkpoint availability (lazy)
        if self._checkpoint_available is None:
            self._checkpoint_available = self._check_checkpoint()

        if self._checkpoint_available:
            if not self._MODEL_INFERENCE_IMPLEMENTED:
                if not self._model_inference_warning_emitted:
                    logger.info(
                        "DepthCrafter checkpoint found at %s, but model inference "
                        "is not implemented yet. Falling back to synthetic depth.",
                        self._checkpoint_path(),
                    )
                    self._model_inference_warning_emitted = True
                return self._compute_synthetic_depth(pil_image)
            try:
                return self._infer_model(pil_image, device)
            except Exception as e:
                self._checkpoint_available = False
                logger.warning(
                    "DepthCrafter model inference failed: %s. " "Falling back to synthetic depth.",
                    e,
                )

        # Synthetic fallback: luminance-based depth
        return self._compute_synthetic_depth(pil_image)

    def _compute_synthetic_depth(self, pil_image: Image.Image) -> np.ndarray:
        """Compute synthetic depth from luminance (fallback mode).

        Uses luminance as a proxy for depth: darker pixels = farther.
        This is consistent with SyntheticDepthBackend but wrapped with
        temporal filtering.

        Args:
            pil_image: Input PIL image (RGB).

        Returns:
            Synthetic depth map (H, W), float32, values in [0, 1].
        """
        gray = pil_image.convert("L")
        gray_array = np.asarray(gray, dtype=np.float32)
        # Invert: brighter pixels = closer (smaller depth value)
        depth = 1.0 - (gray_array / 255.0)
        return depth

    def _infer_model(
        self,
        pil_image: Image.Image,
        device: Optional[str],
    ) -> np.ndarray:
        """Run DepthCrafter model inference.

        This is a placeholder for when the DepthCrafter checkpoint becomes
        available. The implementation loads the model lazily and runs
        inference to produce a temporally-consistent depth map.

        Args:
            pil_image: Input image.
            device: Device for inference.

        Returns:
            Model depth map (H, W), float32, values in [0, 1].

        Raises:
            RuntimeError: If model loading or inference fails.
        """
        raise RuntimeError(
            "DepthCrafter model inference not yet available. "
            "Checkpoint required at checkpoints/depthcrafter_v1.pt. "
            "See ADR-026 for integration roadmap."
        )

    def _apply_temporal_filter(self, raw_depth: np.ndarray) -> np.ndarray:
        """Apply exponential moving average temporal filter.

        Smooths depth maps across video frames to eliminate inter-frame
        flicker. For the first frame, returns the raw depth unmodified.

        Algorithm:
            smoothed_t = alpha * raw_t + (1 - alpha) * smoothed_{t-1}

        Args:
            raw_depth: Current frame's raw depth map (H, W).

        Returns:
            Temporally-smoothed depth map (H, W), float32.
        """
        depth = raw_depth.astype(np.float32)

        if self._ema_state is None or self._ema_state.shape != depth.shape:
            # First frame or resolution change: initialize EMA state
            self._ema_state = depth.copy()
            return depth

        # EMA update
        alpha = self._temporal_alpha
        self._ema_state = alpha * depth + (1.0 - alpha) * self._ema_state
        return self._ema_state.copy()

    def _check_checkpoint(self) -> bool:
        """Check if DepthCrafter checkpoint is available.

        Returns:
            True if checkpoint exists at expected path.
        """
        from pathlib import Path

        checkpoint_path = self._checkpoint_path()

        exists = Path(checkpoint_path).exists()
        if not exists:
            logger.debug(
                "DepthCrafter checkpoint not found at %s",
                checkpoint_path,
            )
        return exists

    def get_cache_key(self, image: Union[Image.Image, np.ndarray]) -> str:
        """Generate deterministic cache key.

        Note: Temporal state is NOT included in the cache key because
        caching temporally-filtered results would be incorrect (each
        frame depends on prior frames).

        Args:
            image: Input image.

        Returns:
            Cache key string.
        """
        img_array = self._to_rgb_uint8_array(image)

        content_hash = hashlib.sha256(img_array.tobytes()).hexdigest()[:16]
        config_hash = hashlib.sha256(
            (
                f"alpha={self._temporal_alpha:.6f}|"
                f"buffer={self._max_buffer_size}|"
                f"checkpoint={self._checkpoint_path()}|"
                f"version={__version__}"
            ).encode()
        ).hexdigest()[:12]
        return f"depthcrafter-v{__version__}-{config_hash}-{content_hash}"

    def reset_temporal_state(self) -> None:
        """Reset temporal filter state.

        Call this when starting a new video sequence to prevent
        temporal blending between unrelated sequences.
        """
        self._ema_state = None
        self._temporal_buffer.clear()
        logger.debug("DepthCrafter temporal state reset")

    def reset_state(self, sequence_id: Optional[str] = None) -> None:
        """Reset internal state for a new sequence (StatefulBackend protocol).

        Delegates to reset_temporal_state(). The orchestrator calls this
        at sequence boundaries to prevent cross-sequence contamination.

        Args:
            sequence_id: Optional identifier for the new sequence.
        """
        logger.debug(
            "DepthCrafter reset_state called (sequence_id=%s)",
            sequence_id,
        )
        self.reset_temporal_state()

    def has_state(self) -> bool:
        """Whether EMA temporal state is initialized."""
        return self._ema_state is not None

    @property
    def temporal_buffer_length(self) -> int:
        """Return current temporal buffer length."""
        return len(self._temporal_buffer)

    def __repr__(self) -> str:
        mode = "model" if (self._checkpoint_available and self._model) else "fallback"
        return f"DepthCrafterBackend(name={self.name!r}, " f"mode={mode!r}, " f"temporal_alpha={self._temporal_alpha})"
