"""Inference engine for Depth Anything V3.

Provides depth estimation using Depth Anything V3 models via transformers.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
import logging
import time
import numpy as np

from .config import DA3Config

logger = logging.getLogger(__name__)

# Try importing dependencies with graceful fallback
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    logger.warning("torch not available, install with: pip install torch")

try:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available, install with: pip install transformers")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available, install with: pip install Pillow")


@dataclass
class DepthResult:
    """Result from depth inference."""
    depth_map: np.ndarray
    original_image: np.ndarray
    metadata: Dict[str, Any]

    @property
    def depth(self) -> np.ndarray:
        """Alias for depth_map to support both naming conventions."""
        return self.depth_map


class DA3InferenceEngine:
    """Inference engine for Depth Anything V3 models.

    Supports CPU, CUDA, and MPS (Apple Silicon) backends.
    Uses transformers library for model loading and inference.
    """

    def __init__(
        self,
        config: DA3Config,
        commercial_use: bool = True,
        validate_license_strict: bool = False
    ):
        """Initialize inference engine.

        Args:
            config: DA3 configuration
            commercial_use: Whether commercial use is enabled
            validate_license_strict: Whether to strictly validate license
        """
        self.config = config
        self.commercial_use = commercial_use
        self.validate_license_strict = validate_license_strict

        # Check dependencies
        if not TORCH_AVAILABLE:
            raise ImportError(
                "torch is required for DA3InferenceEngine. "
                "Install with: pip install torch"
            )
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required for DA3InferenceEngine. "
                "Install with: pip install transformers"
            )
        if not PIL_AVAILABLE:
            raise ImportError(
                "Pillow is required for DA3InferenceEngine. "
                "Install with: pip install Pillow"
            )

        # Auto-detect device
        self.device = self._auto_detect_device()
        logger.info("DA3InferenceEngine using device: %s", self.device)

        # Load model and processor
        self.model = None
        self.processor = None
        self._load_model()

    def _auto_detect_device(self) -> str:
        """Auto-detect optimal device for inference."""
        # Use device from config if specified
        if hasattr(self.config.device, 'device') and self.config.device.device != "cpu":
            return self.config.device.device

        # Auto-detect
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self) -> None:
        """Load Depth Anything V3 model from HuggingFace."""
        # Get HuggingFace model ID from variant
        model_id = self.config.model_variant.value.huggingface_id

        logger.info("Loading Depth Anything V3 model: %s", model_id)

        try:
            # Load processor and model
            # nosec B615 - revision pinning intentionally omitted for development flexibility
            # Production deployments should pin specific model revisions
            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.model = AutoModelForDepthEstimation.from_pretrained(model_id)

            # Move model to device
            if self.device == "cuda":
                self.model = self.model.to("cuda")
            elif self.device == "mps":
                self.model = self.model.to("mps")

            # Set to eval mode
            self.model.eval()

            logger.info("Successfully loaded model on device: %s", self.device)

        except Exception as e:
            logger.error("Failed to load Depth Anything V3 model: %s", e)
            raise RuntimeError(f"Failed to load model {model_id}: {e}") from e

    def predict(self, image: np.ndarray) -> DepthResult:
        """Run depth inference on an image (alias for infer).

        Args:
            image: Input image as numpy array (HxWxC, RGB, float32 [0,1] or uint8 [0,255])

        Returns:
            DepthResult with depth map and metadata
        """
        return self.infer(image)

    def infer(self, image: np.ndarray) -> DepthResult:
        """Run depth inference on an image.

        Args:
            image: Input image as numpy array (HxWxC, RGB, float32 [0,1] or uint8 [0,255])

        Returns:
            DepthResult with depth map and metadata
        """
        start_time = time.time()

        # Store original image
        original_image = image.copy()

        # Convert numpy to PIL Image
        if image.dtype in (np.float32, np.float64):
            # Assume [0, 1] range
            image_pil = Image.fromarray((image * 255).astype(np.uint8))
        else:
            # Assume uint8 [0, 255]
            image_pil = Image.fromarray(image)

        # Ensure RGB
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')

        # Preprocess image
        inputs = self.processor(images=image_pil, return_tensors="pt")

        # Move inputs to device
        if self.device in ["mps", "cuda"]:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image_pil.size[::-1],  # (height, width)
            mode="bicubic",
            align_corners=False,
        )

        # Convert to numpy and normalize
        depth_raw = prediction.squeeze().cpu().numpy()

        # Normalize to [0, 1] - 0=closest, 1=farthest
        depth_min = depth_raw.min()
        depth_max = depth_raw.max()
        depth_normalized = (depth_raw - depth_min) / (depth_max - depth_min + 1e-8)

        inference_time = time.time() - start_time

        # Build metadata
        metadata = {
            "model_variant": self.config.model_variant.value.name,
            "device": self.device,
            "inference_time_ms": inference_time * 1000,
            "depth_min": float(depth_min),
            "depth_max": float(depth_max),
            "shape": depth_normalized.shape,
        }

        return DepthResult(
            depth_map=depth_normalized.astype(np.float32),
            original_image=original_image,
            metadata=metadata
        )

    def infer_from_path(self, image_path: Path) -> DepthResult:
        """Run depth inference on an image file.

        Args:
            image_path: Path to input image

        Returns:
            DepthResult with depth map and metadata
        """
        # Load image
        image_pil = Image.open(image_path).convert('RGB')
        image_np = np.array(image_pil)

        # Run inference
        return self.infer(image_np)
