"""Inference engine for Depth Anything V3.

Multi-backend support following V2 architecture patterns:
- PyTorch (CPU/MPS for development)
- CoreML (ANE optimization if V3 models exist)
- Auto-detection of optimal backend for hardware
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, Union, TYPE_CHECKING
import logging
import time
import numpy as np
from PIL import Image

from .config import DA3Config, ModelVariant

if TYPE_CHECKING:
    from .input_manager import ImageInput

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    logging.warning("torch not available, install with: pip install torch")

try:
    from transformers import pipeline
    from transformers.pipelines.depth_estimation import DepthEstimationPipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    DepthEstimationPipeline = Any  # type: ignore
    logging.warning("transformers not available, install with: pip install transformers")

try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False
    logging.warning("coremltools not available, install with: pip install coremltools")


logger = logging.getLogger(__name__)


class ModelBackend(Enum):
    """Supported inference backends."""
    PYTORCH_CPU = "pytorch_cpu"
    PYTORCH_MPS = "pytorch_mps"
    PYTORCH_CUDA = "pytorch_cuda"
    COREML = "coreml"


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

    Follows V2 architecture patterns with multi-backend support.
    Auto-detects optimal backend/device for current hardware.

    Performance (expected, based on V2 Small):
    - Small (518x518): ~24-65ms on Apple Neural Engine
    - Small (1024x1024): ~65-90ms on Apple Neural Engine

    Example:
        >>> config = DA3Config()
        >>> engine = DA3InferenceEngine(config)
        >>> result = engine.predict(image)
        >>> depth_map = result.depth_map  # HxW normalized to [0, 1]
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

        # Auto-detect backend and device
        self.backend = self._auto_detect_backend()
        self.device = self._resolve_device()

        # Initialize model (lazy loading)
        self.model: Optional[Union[DepthEstimationPipeline, Any]] = None
        self._model_loaded = False
        self._using_fallback_model = False
        self._fallback_model_id: Optional[str] = None

        logger.info(
            "Initialized DA3InferenceEngine (variant=%s, backend=%s, device=%s)",
            config.model_variant.name,
            self.backend.name,
            self.device,
        )

    def _auto_detect_backend(self) -> ModelBackend:
        """Auto-detect optimal backend for current hardware."""
        device_spec = self.config.device.device.lower()

        # Explicit device override
        if device_spec == "cuda" and TORCH_AVAILABLE and torch.cuda.is_available():
            return ModelBackend.PYTORCH_CUDA
        if device_spec == "mps" and TORCH_AVAILABLE and torch.backends.mps.is_available():
            return ModelBackend.PYTORCH_MPS
        if device_spec == "coreml" and COREML_AVAILABLE:
            return ModelBackend.COREML
        if device_spec == "cpu":
            return ModelBackend.PYTORCH_CPU

        # Auto-detect (device_spec == "auto" or default)
        # Prefer MPS on Apple Silicon for V3 (CoreML support TBD)
        if TORCH_AVAILABLE and torch.backends.mps.is_available():
            return ModelBackend.PYTORCH_MPS

        # CUDA if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return ModelBackend.PYTORCH_CUDA

        # CPU fallback
        if TORCH_AVAILABLE:
            return ModelBackend.PYTORCH_CPU

        raise RuntimeError(
            "No backend available. Install torch with: pip install torch"
        )

    def _resolve_device(self) -> str:
        """Resolve device string for PyTorch."""
        if self.backend == ModelBackend.PYTORCH_CUDA:
            return "cuda"
        if self.backend == ModelBackend.PYTORCH_MPS:
            return "mps"
        if self.backend == ModelBackend.COREML:
            return "coreml"
        return "cpu"

    def _load_model(self) -> None:
        """Load model based on backend (lazy loading)."""
        if self._model_loaded:
            return

        if self.backend == ModelBackend.COREML:
            self._load_coreml_model()
        else:
            self._load_pytorch_model()

        self._model_loaded = True

    def _load_pytorch_model(self) -> None:
        """Load PyTorch model using transformers pipeline.

        Note: Depth Anything V3 models may not be available in transformers format yet.
        Falls back to V2 metric models if V3 models are not found.
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "torch required for PyTorch backend. Install with: pip install torch"
            )
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers required for PyTorch backend. "
                "Install with: pip install transformers"
            )

        # Get HuggingFace model ID from config
        model_id = self.config.model_variant.value.huggingface_id

        # Fallback mapping to V2 metric models (which exist on HuggingFace)
        v3_to_v2_fallback = {
            "depth-anything/Depth-Anything-V3-Metric-Large-hf": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
            "depth-anything/Depth-Anything-V3-Metric-Base-hf": "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
            "depth-anything/Depth-Anything-V3-Metric-Small-hf": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        }

        try:
            # Use transformers pipeline for simplicity
            device_arg = self.device if self.device != "mps" else 0
            self.model = pipeline(
                task="depth-estimation",
                model=model_id,
                device=device_arg,
            )
            logger.info("Loaded PyTorch model: %s", model_id)

        except Exception as e:
            # Try fallback to V2 metric model
            fallback_model = v3_to_v2_fallback.get(model_id)
            if fallback_model:
                logger.warning(
                    "V3 model %s not available, falling back to V2: %s",
                    model_id,
                    fallback_model
                )
                try:
                    # Try with device first
                    device_arg = self.device if self.device != "mps" else 0
                    try:
                        self.model = pipeline(
                            task="depth-estimation",
                            model=fallback_model,
                            device=device_arg,
                        )
                    except (RuntimeError, ValueError, TypeError) as device_error:
                        # If model uses accelerate, device arg not allowed
                        # Check for accelerate-specific error messages
                        msg = str(device_error).lower()
                        if "accelerate" in msg or "cannot be moved" in msg:
                            logger.info("Model uses accelerate, loading without device arg")
                            self.model = pipeline(
                                task="depth-estimation",
                                model=fallback_model,
                            )
                        else:
                            raise

                    logger.info("Loaded fallback V2 model: %s", fallback_model)
                    self._using_fallback_model = True
                    self._fallback_model_id = fallback_model
                    return
                except Exception as fallback_error:
                    logger.error("Fallback model also failed: %s", fallback_error)

            logger.error("Failed to load PyTorch model: %s", e)
            raise RuntimeError(
                f"Failed to load model {model_id} (and fallback): {e}"
            ) from e

    def _load_coreml_model(self) -> None:
        """Load CoreML model for Apple Neural Engine.

        Note: CoreML V3 models may not exist yet. This is a placeholder
        following V2 patterns for future compatibility.
        """
        if not COREML_AVAILABLE:
            raise ImportError(
                "coremltools required for CoreML backend. "
                "Install with: pip install coremltools"
            )

        logger.warning(
            "CoreML support for Depth Anything V3 is not yet available. "
            "Falling back to PyTorch MPS backend."
        )

        # Fallback to MPS
        self.backend = ModelBackend.PYTORCH_MPS
        self.device = "mps"
        self._load_pytorch_model()

    def predict(self, image: Union[np.ndarray, Path, str, "ImageInput"]) -> DepthResult:
        """Run depth inference on an image (main API).

        Accepts multiple input types for flexibility:
        - np.ndarray: Direct numpy array
        - Path/str: File path (delegates to infer_from_path)
        - ImageInput: Path wrapper from input_manager

        Args:
            image: Input image (numpy array, path, or ImageInput)

        Returns:
            DepthResult with depth map and metadata

        Raises:
            TypeError: If image type is not supported
        """
        # Handle ImageInput (path wrapper)
        try:
            from .input_manager import ImageInput
            if isinstance(image, ImageInput):
                return self.infer_from_path(image.path)
        except (ImportError, AttributeError):
            pass  # ImageInput not available or image is not ImageInput

        # Handle Path/str
        if isinstance(image, (Path, str)):
            return self.infer_from_path(Path(image))

        # Handle numpy array (main path)
        if isinstance(image, np.ndarray):
            return self.infer(image)

        raise TypeError(
            f"Expected np.ndarray, Path, str, or ImageInput, got {type(image)}"
        )

    def infer(self, image: np.ndarray) -> DepthResult:
        """Run depth inference on an image.

        Args:
            image: Input image as numpy array (HxWx3)

        Returns:
            DepthResult with depth map and metadata
        """
        # Lazy load model on first inference
        if not self._model_loaded:
            self._load_model()

        # Convert numpy to PIL if needed
        if isinstance(image, np.ndarray):
            original_image = image.copy()
            if image.dtype in (np.float32, np.float64):
                pil_image = Image.fromarray((image * 255).astype(np.uint8))
            else:
                pil_image = Image.fromarray(image)
        else:
            raise TypeError(f"Expected numpy array, got {type(image)}")

        # Run inference
        start_time = time.time()
        result = self._estimate_depth_pytorch(pil_image)
        inference_time_ms = (time.time() - start_time) * 1000

        # Update metadata
        result["metadata"]["inference_time_ms"] = inference_time_ms
        result["metadata"]["model_variant"] = self.config.model_variant.name
        result["metadata"]["backend"] = self.backend.value
        result["metadata"]["device"] = self.device
        if self._using_fallback_model:
            result["metadata"]["using_fallback"] = True
            result["metadata"]["fallback_model"] = self._fallback_model_id

        return DepthResult(
            depth_map=result["depth"],
            original_image=original_image,
            metadata=result["metadata"],
        )

    def infer_from_path(self, image_path: Path) -> DepthResult:
        """Run depth inference on an image file.

        Args:
            image_path: Path to input image

        Returns:
            DepthResult with depth map and metadata
        """
        # Load image from path
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        # Run inference
        return self.infer(image_np)

    def _estimate_depth_pytorch(self, image: Image.Image) -> dict:
        """Estimate depth using PyTorch backend.

        Args:
            image: PIL Image

        Returns:
            Dictionary with depth map and metadata
        """
        if not TORCH_AVAILABLE:
            raise ImportError("torch required for PyTorch inference")

        # Run inference using pipeline
        if hasattr(self.model, "__call__"):
            prediction = self.model(image)
            depth_raw = prediction["depth"]

            # Convert to numpy
            if isinstance(depth_raw, torch.Tensor):
                depth_raw = depth_raw.cpu().numpy()
            elif isinstance(depth_raw, Image.Image):
                depth_raw = np.array(depth_raw)
        else:
            raise RuntimeError("Model is not callable")

        # Normalize to [0, 1]
        depth_min = depth_raw.min()
        depth_max = depth_raw.max()
        depth_normalized = (depth_raw - depth_min) / (depth_max - depth_min + 1e-8)

        return {
            "depth": depth_normalized.astype(np.float32),
            "depth_raw": depth_raw.astype(np.float32),
            "metadata": {
                "shape": depth_normalized.shape,
            },
        }
