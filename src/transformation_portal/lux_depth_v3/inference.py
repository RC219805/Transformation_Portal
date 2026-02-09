"""Inference engine for Depth Anything V3.

Multi-backend support following V2 architecture patterns:
- PyTorch (CPU/MPS for development)
- CoreML (ANE optimization if V3 models exist)
- Auto-detection of optimal backend for hardware

Supports standard image formats (JPG, PNG, TIFF) and RAW camera files (CR2, NEF, ARW, DNG).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import numpy as np
from PIL import Image

from .config import DA3Config, ModelVariant  # noqa: F401 - Used in docstring examples
from .raw_loader import is_raw_file, load_raw_as_pil

if TYPE_CHECKING:
    from .input_manager import ImageInput

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

try:
    from transformers import pipeline
    from transformers.pipelines.depth_estimation import DepthEstimationPipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    DepthEstimationPipeline = Any  # type: ignore

try:
    import coremltools as ct

    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False
    ct = None  # type: ignore


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
        self, config: Union[DA3Config, str] = "cpu", commercial_use: bool = True, validate_license_strict: bool = False
    ):
        """Initialize inference engine.

        Args:
            config: Either DA3Config object for full control, or device string
                   ("cpu", "mps", "cuda", "auto") for simple usage
            commercial_use: Whether commercial use is enabled
            validate_license_strict: Whether to strictly validate license

        Examples:
            Simple usage with device string:
            >>> engine = DA3InferenceEngine("mps")

            Full control with DA3Config:
            >>> config = DA3Config(model_variant=ModelVariant.METRIC_LARGE)
            >>> engine = DA3InferenceEngine(config)
        """
        # Support simple string device for convenience
        if isinstance(config, str):
            from .config import DeviceConfig

            device_str = config
            device_config = DeviceConfig(device=device_str)
            config = DA3Config(device=device_config)
            logger.debug(f"Auto-constructed DA3Config with device={device_str}")

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
        self._requested_model_id: Optional[str] = None
        self._resolved_model_id: Optional[str] = None

        logger.info(
            "Initialized DA3InferenceEngine (variant=%s, backend=%s, device=%s)",
            config.model_variant.name,
            self.backend.name,
            self.device,
        )

    def _auto_detect_backend(self) -> ModelBackend:
        """Auto-detect optimal backend for current hardware."""
        device_spec = self.config.device.device.lower()

        # Phase 3: Check if CoreML is explicitly requested via config
        use_coreml = getattr(self.config.device, "use_coreml", False)
        if use_coreml and self._should_use_coreml():
            logger.info("CoreML backend enabled via config (5x speedup on Apple Silicon)")
            return ModelBackend.COREML

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

        raise RuntimeError("No backend available. Install torch with: pip install torch")

    def _should_use_coreml(self) -> bool:
        """Check if CoreML should be used based on hardware and dependencies."""
        import platform

        # Only on macOS with Apple Silicon
        if platform.system() != "Darwin":
            return False

        if platform.machine() != "arm64":
            return False

        # CoreML tools must be available
        if not COREML_AVAILABLE:
            logger.warning("CoreML requested but coremltools not available. Install: pip install coremltools")
            return False

        if not TORCH_AVAILABLE:
            logger.warning("CoreML requires torch for model conversion")
            return False

        if not TRANSFORMERS_AVAILABLE:
            logger.warning("CoreML requires transformers for model loading")
            return False

        return True

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
        DA3 Nested models require custom depth-anything-3 library.
        Falls back to V2 metric models if V3 models are not found.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("torch required for PyTorch backend. Install with: pip install torch")
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers required for PyTorch backend. " "Install with: pip install transformers")

        # Get HuggingFace model ID from config
        model_id = self.config.model_variant.value.huggingface_id

        # Provenance
        self._requested_model_id = model_id
        self._resolved_model_id = model_id

        # Check if this is a DA3 Nested model (requires custom library)
        if self._is_da3_model(model_id):
            self._load_da3_model(model_id)
            return

        # Fallback mapping to V2 metric models (which exist on HuggingFace)
        v3_to_v2_fallback = {
            "depth-anything/Depth-Anything-V3-Metric-Large-hf": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
            "depth-anything/Depth-Anything-V3-Metric-Base-hf": "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
            "depth-anything/Depth-Anything-V3-Metric-Small-hf": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        }

        try:
            # Determine device argument for transformers pipeline
            device_arg = self.device if self.device != "mps" else 0

            # Determine dtype for FP16 optimization
            use_fp16 = getattr(self.config.device, "use_fp16", True)
            torch_dtype = None
            if use_fp16 and self.device in ("mps", "cuda"):
                torch_dtype = torch.float16
                logger.info("Enabling FP16 for %s (1.3-1.5x speedup, 2x memory reduction)", self.device)

            # Use transformers pipeline for simplicity
            self.model = pipeline(
                task="depth-estimation",
                model=model_id,
                device=device_arg,
                torch_dtype=torch_dtype,
            )

            # Additional FP16 optimization for MPS
            if use_fp16 and self.device == "mps" and hasattr(self.model.model, "half"):
                self.model.model = self.model.model.half()
                logger.debug("Applied half precision to model for MPS backend")

            logger.info("Loaded PyTorch model: %s", model_id)

        except Exception as e:
            # Try fallback to V2 metric model
            fallback_model = v3_to_v2_fallback.get(model_id)
            if fallback_model:
                logger.warning("V3 model %s not available, falling back to V2: %s", model_id, fallback_model)
                try:
                    device_arg = self.device if self.device != "mps" else 0

                    # Determine dtype for FP16 optimization
                    use_fp16 = getattr(self.config.device, "use_fp16", True)
                    torch_dtype = None
                    if use_fp16 and self.device in ("mps", "cuda"):
                        torch_dtype = torch.float16

                    self.model = pipeline(
                        task="depth-estimation",
                        model=fallback_model,
                        device=device_arg,
                        torch_dtype=torch_dtype,
                    )

                    # Additional FP16 optimization for MPS
                    if use_fp16 and self.device == "mps" and hasattr(self.model.model, "half"):
                        self.model.model = self.model.model.half()

                    logger.info("Loaded fallback V2 model: %s", fallback_model)
                    self._using_fallback_model = True
                    self._fallback_model_id = fallback_model
                    self._resolved_model_id = fallback_model

                    return
                except Exception as fallback_error:
                    logger.error("Fallback model also failed: %s", fallback_error)

            logger.error("Failed to load PyTorch model: %s", e)
            raise RuntimeError(f"Failed to load model {model_id} (and fallback): {e}") from e

    def _is_da3_model(self, model_id: str) -> bool:
        """Check if model ID is a DA3 Nested model."""
        return model_id.startswith("depth-anything/da3") or "da3nested" in model_id.lower()

    def _load_da3_model(self, model_id: str) -> None:
        """Load DA3 model using custom depth-anything-3 library.

        DA3 Nested models require custom library installation:
            git clone https://github.com/ByteDance-Seed/depth-anything-3
            cd depth-anything-3
            pip install -e .

        DA3 uses a different API than transformers:
            - DepthAnything3.from_pretrained() instead of AutoModelForDepthEstimation
            - Different inference interface
        """
        try:
            from depth_anything_3.api import DepthAnything3
        except ImportError:
            # Fallback for older/alternate packaging layouts
            try:
                from depth_anything_3 import DepthAnything3
            except ImportError as e:
                error_msg = (
                    f"\n{'='*80}\n"
                    f"ERROR: DA3 model '{model_id}' requires custom library installation.\n\n"
                    f"DA3 Nested models use a different API than transformers and require:\n"
                    f"  1. Clone: git clone https://github.com/ByteDance-Seed/depth-anything-3\n"
                    f"  2. Install: cd depth-anything-3 && pip install -e .\n\n"
                    f"The DA3 Nested Giant model combines:\n"
                    f"  - Giant model for any-view depth\n"
                    f"  - Metric Large model for metric-scale reconstruction\n"
                    f"  - 1.40B parameters\n"
                    f"  - Full capabilities: relative depth, metric depth, pose estimation, 3D Gaussians\n\n"
                    f"FALLBACK OPTIONS:\n"
                    f"  - Use DA2 model: depth-anything/Depth-Anything-V2-Large-hf (compatible with transformers)\n"
                    f"  - Use DA2 metric: depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf\n\n"
                    f"Original error: {e}\n"
                    f"{'='*80}\n"
                )
                logger.error(error_msg)
                raise ImportError(error_msg) from e

        logger.info(f"Loading DA3 model: {model_id} (using depth-anything-3 library)")

        try:
            self.model = DepthAnything3.from_pretrained(model_id)
            self.model.to(self.device)
            self.model.eval()
            logger.info("✓ DA3 model loaded successfully")
            logger.warning("⚠️  DA3 models use different inference API - custom integration required")
        except Exception as e:
            error_msg = (
                f"Failed to load DA3 model '{model_id}': {e}\n"
                f"Verify the model ID is correct. Recommended DA3 model:\n"
                f"  depth-anything/da3nested-giant-large (1.40B params, metric + relative depth)\n"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _load_coreml_model(self) -> None:
        """Load CoreML model for Apple Neural Engine (Phase 3).

        Converts PyTorch DA3 models to CoreML format with ANE optimization.
        Provides 5x inference speedup on Apple Silicon (400ms → 80ms on M4).
        """
        if not COREML_AVAILABLE:
            raise ImportError("coremltools required for CoreML backend. " "Install with: pip install coremltools")

        from .coreml_backend import CoreMLDepthEstimator

        try:
            # Get HuggingFace model ID from config
            model_id = self.config.model_variant.value.huggingface_id

            logger.info(f"Loading CoreML model: {model_id}")
            self.model = CoreMLDepthEstimator(model_id)

            logger.info("✓ CoreML model loaded with ANE acceleration (5x speedup)")

        except Exception as e:
            logger.error(f"CoreML model loading failed: {e}")
            logger.warning("Falling back to PyTorch MPS backend")

            # Fallback to MPS
            self.backend = ModelBackend.PYTORCH_MPS
            self.device = "mps"
            self._load_pytorch_model()

    def predict(self, image: Union[np.ndarray, "Image.Image", Path, str, "ImageInput"]) -> DepthResult:
        """Run depth inference on an image (main API).

        Accepts multiple input types for flexibility:
        - np.ndarray: Direct numpy array (HxWx3/HxWx4/HxW, uint8/uint16/float32/float64)
        - PIL.Image.Image: PIL Image object (any mode)
        - Path/str: File path (delegates to infer_from_path)
        - ImageInput: Path wrapper from input_manager

        Args:
            image: Input image (numpy array, PIL Image, path, or ImageInput)

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

        # Handle PIL Image or numpy array
        if isinstance(image, (np.ndarray, Image.Image)):
            return self.infer(image)

        raise TypeError(f"Expected np.ndarray, PIL.Image, Path, str, or ImageInput, got {type(image)}")

    def infer(self, image: Union[np.ndarray, "Image.Image"]) -> DepthResult:
        """Run depth inference on an image.

        Accepts multiple input types for flexibility:
        - np.ndarray: Direct numpy array (HxWx3/HxWx4/HxW, uint8/uint16/float32/float64)
        - PIL.Image.Image: PIL Image object (any mode)

        Args:
            image: Input image as numpy array or PIL Image

        Returns:
            DepthResult with depth map and metadata

        Raises:
            TypeError: If image type is not supported
            ValueError: If array shape/dtype is invalid
        """
        # Lazy load model on first inference
        if not self._model_loaded:
            self._load_model()

        # Normalize input to canonical uint8 RGB numpy + PIL Image
        if isinstance(image, Image.Image):
            # PIL Image input: convert to RGB (drop alpha, convert grayscale/palette)
            pil_image = image.convert("RGB")
            # Canonical original_image: uint8 RGB numpy
            original_image = np.array(pil_image, dtype=np.uint8)

        elif isinstance(image, np.ndarray):
            # Validate shape
            if image.ndim == 2:
                # Grayscale HxW → RGB HxWx3
                if image.dtype == np.uint8:
                    gray = image
                elif image.dtype in (np.float32, np.float64):
                    gray = np.clip(image, 0, 1) * 255
                    gray = gray.astype(np.uint8)
                elif image.dtype == np.uint16:
                    gray = (image / 256).astype(np.uint8)  # Scale 16-bit → 8-bit
                else:
                    raise ValueError(f"Unsupported grayscale dtype: {image.dtype}")

                # Convert grayscale to RGB by repeating channel
                rgb_uint8 = np.stack([gray, gray, gray], axis=-1)
                pil_image = Image.fromarray(rgb_uint8, mode="RGB")
                original_image = rgb_uint8

            elif image.ndim == 3:
                h, w, c = image.shape

                if c == 3:
                    # RGB
                    if image.dtype == np.uint8:
                        rgb_uint8 = image.copy()
                    elif image.dtype in (np.float32, np.float64):
                        # Clip and scale float [0,1] → uint8 [0,255]
                        rgb_float = np.clip(image, 0, 1)
                        rgb_uint8 = (rgb_float * 255).astype(np.uint8)
                    elif image.dtype == np.uint16:
                        # Scale 16-bit → 8-bit
                        rgb_uint8 = (image / 256).astype(np.uint8)
                    else:
                        raise ValueError(f"Unsupported RGB dtype: {image.dtype}")

                    pil_image = Image.fromarray(rgb_uint8, mode="RGB")
                    original_image = rgb_uint8

                elif c == 4:
                    # RGBA → RGB (drop alpha channel explicitly)
                    if image.dtype == np.uint8:
                        rgba = image
                    elif image.dtype in (np.float32, np.float64):
                        rgba_float = np.clip(image, 0, 1)
                        rgba = (rgba_float * 255).astype(np.uint8)
                    elif image.dtype == np.uint16:
                        rgba = (image / 256).astype(np.uint8)
                    else:
                        raise ValueError(f"Unsupported RGBA dtype: {image.dtype}")

                    # Drop alpha channel
                    rgb_uint8 = rgba[:, :, :3]
                    pil_image = Image.fromarray(rgb_uint8, mode="RGB")
                    original_image = rgb_uint8

                else:
                    raise ValueError(
                        f"Expected 3 or 4 channels, got {c}. " f"Supported: HxWx3 (RGB), HxWx4 (RGBA), HxW (grayscale)"
                    )

            else:
                raise ValueError(
                    f"Expected 2D (grayscale) or 3D (RGB/RGBA) array, got shape {image.shape}. "
                    f"Batched inputs not supported—process images one at a time."
                )

        else:
            raise TypeError(f"Expected numpy array or PIL.Image, got {type(image)}")

        # Run inference based on backend
        start_time = time.time()

        if self.backend == ModelBackend.COREML:
            # CoreML expects numpy float32 [0,1]
            # Convert canonical uint8 → float32 for CoreML
            image_float32 = original_image.astype(np.float32) / 255.0
            result = self._estimate_depth_coreml(image_float32)
        else:
            # PyTorch path: pass PIL Image (DA3 and transformers both accept PIL)
            result = self._estimate_depth_pytorch(pil_image)

        inference_time_ms = (time.time() - start_time) * 1000

        # Update metadata
        result["metadata"]["inference_time_ms"] = inference_time_ms
        result["metadata"]["model_variant"] = self.config.model_variant.name
        result["metadata"]["backend"] = self.backend.value
        result["metadata"]["device"] = self.device
        # Provenance: requested vs resolved model id
        result["metadata"]["requested_model_id"] = self._requested_model_id
        result["metadata"]["resolved_model_id"] = self._resolved_model_id
        result["metadata"]["resolved_model_source"] = "fallback" if self._using_fallback_model else "primary"

        if self._using_fallback_model:
            result["metadata"]["using_fallback"] = True
            result["metadata"]["fallback_model"] = self._fallback_model_id

        # CRITICAL: original_image is now ALWAYS uint8 RGB regardless of input type
        return DepthResult(
            depth_map=result["depth"],
            original_image=original_image,  # Canonical uint8 RGB numpy
            metadata=result["metadata"],
        )

    def infer_from_path(self, image_path: Path) -> DepthResult:
        """Run depth inference on an image file.

        Supports standard image formats (JPG, PNG, TIFF) and RAW camera files (CR2, NEF, ARW, DNG).

        Args:
            image_path: Path to input image (standard or RAW format)

        Returns:
            DepthResult with depth map and metadata
        """
        # Load image from path (handles both standard and RAW formats)
        if is_raw_file(image_path):
            logger.debug(f"Loading RAW file for inference: {image_path.name}")
            image = load_raw_as_pil(image_path, use_camera_wb=True, half_size=False)
        else:
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

        # Check if this is a DA3 model (custom API)
        try:
            from depth_anything_3.api import DepthAnything3

            is_da3_model = isinstance(self.model, DepthAnything3)
        except ImportError:
            is_da3_model = False

        if is_da3_model:
            # DA3 models use inference() method with list of images
            prediction = self.model.inference([image])
            # DA3 returns Prediction object with .depth attribute (1, H, W)
            depth_raw = prediction.depth[0]  # Remove batch dimension

            # Convert to numpy if needed
            if isinstance(depth_raw, torch.Tensor):
                depth_raw = depth_raw.cpu().numpy()
        elif hasattr(self.model, "__call__"):
            # Transformers pipeline models
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

    def _estimate_depth_coreml(self, image: np.ndarray) -> dict:
        """Estimate depth using CoreML backend (Phase 3).

        Args:
            image: Image as numpy array (H, W, 3) in [0, 255] range

        Returns:
            Dictionary with depth map and metadata
        """
        from .coreml_backend import CoreMLDepthEstimator

        if not isinstance(self.model, CoreMLDepthEstimator):
            raise RuntimeError("Model is not a CoreML estimator")

        # Normalize image to [0, 1] for CoreML
        if image.dtype == np.uint8:
            image_normalized = image.astype(np.float32) / 255.0
        else:
            image_normalized = image.astype(np.float32)

        # Run CoreML inference
        depth_raw = self.model.predict(image_normalized)

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
