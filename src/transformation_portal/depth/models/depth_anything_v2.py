"""
Depth Anything V2 Model Wrapper

Provides unified interface for depth estimation with multiple backend support:
- PyTorch (CPU/MPS for development)
- CoreML (ANE optimization for M4 Max production)
- ONNX (cross-platform deployment)

Model Variants:
- Small: 24.8M params, 49.8MB, Apache 2.0 license, 24ms on M4 Max ANE
- Base: 97.5M params, 195MB, CC-BY-NC-4.0, 50ms on GPU
- Large: 335M params, 671MB, CC-BY-NC-4.0, 100ms on GPU
"""

import logging
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image

from transformation_portal.core.ml_dependency_health import (
    OPTIONAL_IMPORT_EXCEPTIONS,
    detect_transformers_torch_runtime_issue,
)
from transformation_portal.core.security.model_lock import resolve_model_lock_revision

torch = None  # type: ignore[assignment]
transformers_module = None  # type: ignore[assignment]
AutoImageProcessor = None  # type: ignore[assignment]
AutoModelForDepthEstimation = None  # type: ignore[assignment]
pipeline = None  # type: ignore[assignment]
DepthEstimationPipeline = Any  # type: ignore[assignment]
ct = None  # type: ignore[assignment]
resize = None  # type: ignore[assignment]
ort = None  # type: ignore[assignment]
TORCH_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False
COREML_AVAILABLE = False
SKIMAGE_AVAILABLE = False
ONNX_AVAILABLE = False
TRANSFORMERS_TORCH_BACKEND_ISSUE = None
_OPTIONAL_IMPORTS_READY = False


logger = logging.getLogger(__name__)


def _ensure_optional_runtime_imports() -> None:
    """Resolve optional ML/runtime dependencies lazily."""
    global _OPTIONAL_IMPORTS_READY
    global torch, transformers_module, AutoImageProcessor, AutoModelForDepthEstimation, pipeline, DepthEstimationPipeline
    global ct, resize, ort
    global TORCH_AVAILABLE, TRANSFORMERS_AVAILABLE, COREML_AVAILABLE, SKIMAGE_AVAILABLE, ONNX_AVAILABLE
    global TRANSFORMERS_TORCH_BACKEND_ISSUE

    if _OPTIONAL_IMPORTS_READY:
        return
    _OPTIONAL_IMPORTS_READY = True

    try:
        import torch as torch_module

        torch = torch_module
        TORCH_AVAILABLE = True
    except OPTIONAL_IMPORT_EXCEPTIONS:
        TORCH_AVAILABLE = False
        torch = None  # type: ignore[assignment]
        logging.warning("torch not available, install with: pip install torch")

    try:
        import transformers as transformers_pkg
        from transformers import AutoImageProcessor as AutoImageProcessorClass
        from transformers import AutoModelForDepthEstimation as AutoModelForDepthEstimationClass
        from transformers import pipeline as transformers_pipeline
        from transformers.pipelines.depth_estimation import DepthEstimationPipeline as DepthEstimationPipelineClass

        transformers_module = transformers_pkg
        AutoImageProcessor = AutoImageProcessorClass
        AutoModelForDepthEstimation = AutoModelForDepthEstimationClass
        pipeline = transformers_pipeline
        DepthEstimationPipeline = DepthEstimationPipelineClass
        TRANSFORMERS_AVAILABLE = True
    except OPTIONAL_IMPORT_EXCEPTIONS:
        TRANSFORMERS_AVAILABLE = False
        transformers_module = None  # type: ignore[assignment]
        AutoImageProcessor = None  # type: ignore[assignment]
        AutoModelForDepthEstimation = None  # type: ignore[assignment]
        pipeline = None  # type: ignore[assignment]
        DepthEstimationPipeline = Any  # type: ignore[assignment]
        logging.warning("transformers not available, install with: pip install transformers")

    try:
        import coremltools as coremltools_module

        ct = coremltools_module
        COREML_AVAILABLE = True
    except ImportError:
        COREML_AVAILABLE = False
        ct = None  # type: ignore[assignment]
        logging.warning("coremltools not available, install with: pip install coremltools")

    try:
        from skimage.transform import resize as resize_function

        resize = resize_function
        SKIMAGE_AVAILABLE = True
    except ImportError:
        SKIMAGE_AVAILABLE = False
        resize = None  # type: ignore[assignment]
        logging.warning("scikit-image not available, install with: pip install scikit-image")

    try:
        import onnxruntime as onnxruntime_module

        ort = onnxruntime_module
        ONNX_AVAILABLE = True
    except ImportError:
        ONNX_AVAILABLE = False
        ort = None  # type: ignore[assignment]
        logging.warning("onnxruntime not available, install with: pip install onnxruntime")

    TRANSFORMERS_TORCH_BACKEND_ISSUE = detect_transformers_torch_runtime_issue(
        torch if TORCH_AVAILABLE else None,
        transformers_module if TRANSFORMERS_AVAILABLE else None,
    )


class ModelBackend(Enum):
    """Supported inference backends."""

    PYTORCH_CPU = "pytorch_cpu"
    PYTORCH_MPS = "pytorch_mps"  # Apple Silicon GPU
    COREML = "coreml"  # Apple Neural Engine
    ONNX = "onnx"


class ModelVariant(Enum):
    """Depth Anything V2 model variants."""

    SMALL = "depth-anything/Depth-Anything-V2-Small-hf"
    BASE = "depth-anything/Depth-Anything-V2-Base-hf"
    LARGE = "depth-anything/Depth-Anything-V2-Large-hf"

    # CoreML optimized versions
    SMALL_COREML = "apple/coreml-depth-anything-v2-small"


# ONNX model repository and filename candidates for HuggingFace Hub downloads.
# File naming can differ across repos; try deterministic candidates in order.
ONNX_MODEL_SOURCES = {
    ModelVariant.SMALL: {
        "repo_id": "onnx-community/depth-anything-v2-small",
        "filenames": ("depth_anything_v2_vits.onnx", "model.onnx"),
    },
    ModelVariant.BASE: {
        "repo_id": "onnx-community/depth-anything-v2-base",
        "filenames": ("depth_anything_v2_vitb.onnx", "model.onnx"),
    },
    ModelVariant.LARGE: {
        "repo_id": "onnx-community/depth-anything-v2-large",
        "filenames": ("depth_anything_v2_vitl.onnx", "model.onnx"),
    },
}


class DepthAnythingV2Model:
    """
    Depth Anything V2 depth estimation model with multi-backend support.

    Performance (M4 Max):
    - Small (518x518): 24ms (ANE), 35ms (MPS)
    - Small (1024x1024): 65ms (ANE), 90ms (MPS)
    - Large (518x518): 90ms (GPU), 100ms (MPS)

    Example:
        >>> model = DepthAnythingV2Model(
        ...     variant=ModelVariant.SMALL,
        ...     backend=ModelBackend.PYTORCH_MPS
        ... )
        >>> depth = model.estimate_depth(image)
        >>> depth_map = depth['depth']  # HxW normalized to [0, 1]
    """

    def __init__(
        self,
        variant: ModelVariant = ModelVariant.SMALL,
        backend: Optional[ModelBackend] = None,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        *,
        precision: str = "fp16",
        model_revision: Optional[str] = None,
        onnx_revision: Optional[str] = None,
        coreml_revision: Optional[str] = None,
        strict_model_lock: Optional[bool] = None,
    ):
        """
        Initialize depth estimation model.

        Args:
            variant: Model variant to use (SMALL recommended for production)
            backend: Inference backend (auto-detected if None)
            model_path: Path to local model file (downloads if None)
            device: Device override ("cpu", "mps", "cuda")
            precision: Model precision ("fp32", "fp16")
            model_revision: Optional immutable HF revision for PyTorch model loads
            onnx_revision: Optional immutable HF revision for ONNX artifact downloads
            coreml_revision: Optional immutable HF revision for CoreML artifact downloads
            strict_model_lock: Enforce pinned revisions for remote model loads.
                If None, uses ``TP_STRICT_MODEL_LOCK`` environment variable.
        """
        _ensure_optional_runtime_imports()

        self.variant = variant
        self.precision = precision
        self.model_path = Path(model_path) if model_path else None
        self.model_revision = model_revision
        self.onnx_revision = onnx_revision
        self.coreml_revision = coreml_revision
        self.strict_model_lock = strict_model_lock

        # Auto-detect backend if not specified
        if backend is None:
            backend = self._auto_detect_backend()
        self.backend = backend

        # Auto-detect device if not specified
        if device is None:
            device = self._auto_detect_device()
        self.device = device
        self._resolve_remote_artifact_revisions()

        # Initialize model with proper types
        self.model: Optional[Union[DepthEstimationPipeline, Any]] = None
        self.processor: Optional[Any] = None
        self._load_model()

        logger.info(
            "Initialized Depth Anything V2 (variant=%s, backend=%s, device=%s)",
            variant.name,
            backend.name,
            device,
        )

    @staticmethod
    def _coreml_repo_for_variant(variant: ModelVariant) -> Optional[str]:
        return {
            ModelVariant.SMALL: "apple/coreml-depth-anything-v2-small",
            ModelVariant.SMALL_COREML: "apple/coreml-depth-anything-v2-small",
        }.get(variant)

    @staticmethod
    def _coreml_filename_for_variant(variant: ModelVariant) -> Optional[str]:
        return {
            ModelVariant.SMALL: "DepthAnythingV2SmallF16.mlpackage",
            ModelVariant.SMALL_COREML: "DepthAnythingV2SmallF16.mlpackage",
        }.get(variant)

    @staticmethod
    def _onnx_source_for_variant(variant: ModelVariant) -> Optional[dict[str, object]]:
        return ONNX_MODEL_SOURCES.get(variant)

    def _resolve_remote_artifact_revisions(self) -> None:
        """Resolve effective revisions from explicit args + model lock manifest."""
        if self.model_path is not None:
            return

        if self.backend in (ModelBackend.PYTORCH_CPU, ModelBackend.PYTORCH_MPS):
            self.model_revision = resolve_model_lock_revision(
                self.variant.value,
                self.model_revision,
                strict=self.strict_model_lock,
                context="DepthAnythingV2Model(PyTorch)",
            )
            if not self.model_revision:
                logger.warning(
                    "DepthAnythingV2 PyTorch model '%s' is unpinned (no revision). "
                    "Set model_revision or config/model_lock_manifest.yaml for deterministic, supply-chain-safe loads.",
                    self.variant.value,
                )

        if self.backend == ModelBackend.ONNX:
            onnx_source = self._onnx_source_for_variant(self.variant)
            if not onnx_source:
                raise ValueError(f"ONNX model not available for variant {self.variant}. " "Use SMALL, BASE, or LARGE.")
            onnx_repo = str(onnx_source["repo_id"])
            self.onnx_revision = resolve_model_lock_revision(
                onnx_repo,
                self.onnx_revision,
                strict=self.strict_model_lock,
                context="DepthAnythingV2Model(ONNX)",
            )
            if not self.onnx_revision:
                logger.warning(
                    "DepthAnythingV2 ONNX artifact for '%s' is unpinned (no onnx_revision). "
                    "Set onnx_revision or config/model_lock_manifest.yaml for deterministic, supply-chain-safe downloads.",
                    self.variant.value,
                )

        if self.backend == ModelBackend.COREML:
            coreml_repo = self._coreml_repo_for_variant(self.variant)
            if not coreml_repo:
                raise ValueError(
                    f"CoreML model not available for variant {self.variant}. "
                    "Use SMALL/SMALL_COREML only; Base CoreML repo is not published on Hugging Face."
                )
            self.coreml_revision = resolve_model_lock_revision(
                coreml_repo,
                self.coreml_revision,
                strict=self.strict_model_lock,
                context="DepthAnythingV2Model(CoreML)",
            )
            if not self.coreml_revision:
                logger.warning(
                    "DepthAnythingV2 CoreML artifact for '%s' is unpinned (no coreml_revision). "
                    "Set coreml_revision or config/model_lock_manifest.yaml for deterministic, supply-chain-safe downloads.",
                    self.variant.value,
                )

    def _auto_detect_backend(self) -> ModelBackend:
        """Auto-detect optimal backend for current hardware."""
        # Prefer CoreML on Apple Silicon for best performance
        if TORCH_AVAILABLE and COREML_AVAILABLE:
            try:
                if hasattr(torch, "backends") and torch.backends.mps.is_available():
                    return ModelBackend.COREML
            except AttributeError:
                pass

        # Fallback to PyTorch with MPS acceleration
        if TORCH_AVAILABLE:
            try:
                if hasattr(torch, "backends") and torch.backends.mps.is_available():
                    return ModelBackend.PYTORCH_MPS
            except AttributeError:
                pass

        # CPU fallback (or ONNX if torch not available)
        if TORCH_AVAILABLE:
            return ModelBackend.PYTORCH_CPU
        if ONNX_AVAILABLE:
            return ModelBackend.ONNX
        raise RuntimeError("No backend available. Install torch or onnxruntime.")

    def _auto_detect_device(self) -> str:
        """Auto-detect optimal device for PyTorch."""
        if self.backend == ModelBackend.COREML:
            return "coreml"
        if self.backend == ModelBackend.ONNX:
            return "onnx"
        if self.backend == ModelBackend.PYTORCH_MPS:
            return "mps"
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load_model(self) -> None:
        """Load model based on backend."""
        if self.backend == ModelBackend.COREML:
            self._load_coreml_model()
        elif self.backend == ModelBackend.ONNX:
            self._load_onnx_model()
        elif self.backend in [ModelBackend.PYTORCH_CPU, ModelBackend.PYTORCH_MPS]:
            self._load_pytorch_model()
        else:
            raise NotImplementedError(f"Backend {self.backend} not implemented")

    def _load_pytorch_model(self) -> None:
        """Load PyTorch model using transformers."""
        if not TORCH_AVAILABLE:
            raise ImportError("torch required for PyTorch backend. " "Install with: pip install torch")
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers required for PyTorch backend. " "Install with: pip install transformers")
        if TRANSFORMERS_TORCH_BACKEND_ISSUE:
            raise RuntimeError(TRANSFORMERS_TORCH_BACKEND_ISSUE)

        try:
            # Use transformers pipeline for simplicity
            pipeline_kwargs = {
                "task": "depth-estimation",
                "model": self.variant.value,
                "device": self.device if self.device != "mps" else 0,  # MPS uses device 0
            }
            if self.model_revision:
                pipeline_kwargs["revision"] = self.model_revision
            self.model = pipeline(
                task="depth-estimation",
                model=pipeline_kwargs["model"],
                device=pipeline_kwargs["device"],
                revision=pipeline_kwargs.get("revision"),
            )
            logger.info("Loaded PyTorch model: %s", self.variant.value)

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to load PyTorch model: %s", e)
            # Fallback: manual loading
            # Production deployments should pin specific model revisions
            self.processor = AutoImageProcessor.from_pretrained(  # nosec B615
                self.variant.value,
                revision=self.model_revision,
                use_fast=False,
            )
            self.model = AutoModelForDepthEstimation.from_pretrained(  # nosec B615
                self.variant.value,
                revision=self.model_revision,
            )

            if self.device == "mps":
                self.model = self.model.to("mps")
            elif self.device == "cuda":
                self.model = self.model.to("cuda")

            logger.info("Loaded PyTorch model manually: %s", self.variant.value)

    def _load_onnx_model(self) -> None:
        """
        Load ONNX model for cross-platform inference.

        Expected ONNX model format:
        - Input tensor name: first input from model.get_inputs()
        - Input tensor shape: (1, 3, H, W) in NCHW format
        - Input normalization: ImageNet mean/std
            - mean: [0.485, 0.456, 0.406]
            - std:  [0.229, 0.224, 0.225]
            - Input pixel values should be float32 in range [0, 1],
              normalized as (x - mean) / std per channel
        - Output: depth map as float32, shape varies by model

        If providing a custom ONNX model via `model_path`, ensure it
        matches these specifications. Incompatible models may fail to
        load or produce incorrect results.
        """
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime required for ONNX backend. " "Install with: pip install onnxruntime")

        # Check if local model exists
        if self.model_path and self.model_path.exists():
            model_path = self.model_path
        else:
            # Download from HuggingFace Hub
            model_path = self._download_onnx_model()

        try:
            # Configure session options for optimal performance
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            # Select execution providers based on available hardware
            providers = self._get_onnx_providers()

            self.model = ort.InferenceSession(str(model_path), sess_options=sess_options, providers=providers)
            logger.info("Loaded ONNX model: %s with providers: %s", model_path, providers)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to load ONNX model: %s", e)
            logger.info("Falling back to PyTorch CPU backend")
            self.backend = ModelBackend.PYTORCH_CPU
            self.device = "cpu"
            self._load_pytorch_model()

    def _get_onnx_providers(self) -> list:
        """Get available ONNX execution providers in priority order."""
        available_providers = ort.get_available_providers()
        preferred_order = [
            "CUDAExecutionProvider",
            "CoreMLExecutionProvider",
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ]
        providers = []
        for provider in preferred_order:
            if provider in available_providers:
                providers.append(provider)
        # Ensure at least CPU provider is included
        if not providers:
            providers = ["CPUExecutionProvider"]
        return providers

    def _download_onnx_model(self) -> Path:
        """Download ONNX model from HuggingFace Hub."""
        from huggingface_hub import hf_hub_download  # pylint: disable=import-outside-toplevel

        onnx_source = self._onnx_source_for_variant(self.variant)
        if not onnx_source:
            raise ValueError(f"ONNX model not available for variant {self.variant}. " "Use SMALL, BASE, or LARGE.")

        onnx_repo = str(onnx_source["repo_id"])
        filenames = tuple(str(name) for name in onnx_source["filenames"])
        last_error: Optional[Exception] = None

        for onnx_filename in filenames:
            try:
                # Production deployments should pin specific model revisions
                model_path = hf_hub_download(  # nosec B615
                    repo_id=onnx_repo,
                    filename=onnx_filename,
                    cache_dir=Path.home() / ".cache" / "depth_anything_v2",
                    revision=self.onnx_revision,
                )
                return Path(model_path)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                last_error = exc

        raise RuntimeError(
            f"Failed to download ONNX model for variant {self.variant} from '{onnx_repo}' "
            f"using filenames {filenames}: {last_error}"
        ) from last_error

    def _load_coreml_model(self) -> None:
        """Load CoreML model for Apple Neural Engine."""
        if not COREML_AVAILABLE:
            raise ImportError("coremltools required for CoreML backend. " "Install with: pip install coremltools")

        # Check if local model exists
        if self.model_path and self.model_path.exists():
            model_path = self.model_path
        else:
            # Download from HuggingFace Hub
            model_path = self._download_coreml_model()

        try:
            self.model = ct.models.MLModel(str(model_path))
            logger.info("Loaded CoreML model: %s", model_path)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to load CoreML model: %s", e)
            logger.info("Falling back to PyTorch MPS backend")
            self.backend = ModelBackend.PYTORCH_MPS
            self.device = "mps"
            self._load_pytorch_model()

    def _download_coreml_model(self) -> Path:
        """Download CoreML model from HuggingFace Hub."""
        from huggingface_hub import hf_hub_download  # pylint: disable=import-outside-toplevel

        # Map variant to CoreML repo
        coreml_variant = self._coreml_repo_for_variant(self.variant)

        if not coreml_variant:
            raise ValueError(
                f"CoreML model not available for variant {self.variant}. "
                "Use SMALL/SMALL_COREML only; Base CoreML repo is not published on Hugging Face."
            )

        filename = self._coreml_filename_for_variant(self.variant)
        if not filename:
            raise ValueError(
                f"CoreML filename not available for variant {self.variant}. "
                "Use SMALL/SMALL_COREML only; Base CoreML repo is not published on Hugging Face."
            )

        # Download model package
        # Production deployments should pin specific model revisions
        model_path = hf_hub_download(  # nosec B615
            repo_id=coreml_variant,
            filename=filename,
            cache_dir=Path.home() / ".cache" / "depth_anything_v2",
            revision=self.coreml_revision,
        )

        return Path(model_path)

    def estimate_depth(
        self,
        image: Union[np.ndarray, Image.Image, str, Path],
        output_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """
        Estimate depth map from input image.

        Args:
            image: Input image (numpy array, PIL Image, or path)
            output_size: Resize output to (height, width). None keeps input size.

        Returns:
            Dictionary with:
                - 'depth': Depth map as numpy array (HxW), normalized to [0, 1]
                           where 0=closest, 1=farthest
                - 'depth_raw': Raw depth predictions before normalization
                - 'metadata': Model information and timing
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            # Convert numpy to PIL
            if image.dtype in (np.float32, np.float64):
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)

        # Estimate depth based on backend
        if self.backend == ModelBackend.COREML:
            result = self._estimate_depth_coreml(image)
        elif self.backend == ModelBackend.ONNX:
            result = self._estimate_depth_onnx(image)
        else:
            result = self._estimate_depth_pytorch(image)

        # Resize output if requested
        if output_size is not None:
            if not SKIMAGE_AVAILABLE:
                raise ImportError("scikit-image is required for resizing. Install with: pip install scikit-image")

            result["depth"] = resize(
                result["depth"],
                output_size,
                order=1,  # Bilinear
                preserve_range=True,
                anti_aliasing=True,
            )

        return result

    def _estimate_depth_pytorch(self, image: Image.Image) -> dict:
        """Estimate depth using PyTorch backend."""
        if not TORCH_AVAILABLE:
            raise ImportError("torch required for PyTorch inference")

        start_time = time.time()

        # Run inference
        if hasattr(self.model, "__call__"):
            # Pipeline API
            prediction = self.model(image)
            depth_raw = prediction["depth"]

            # Convert to numpy
            if TORCH_AVAILABLE and isinstance(depth_raw, torch.Tensor):
                depth_raw = depth_raw.cpu().numpy()
            elif isinstance(depth_raw, Image.Image):
                depth_raw = np.array(depth_raw)
        else:
            # Manual inference
            inputs = self.processor(images=image, return_tensors="pt")

            if self.device in ["mps", "cuda"]:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                depth_raw = outputs.predicted_depth

            # Convert to numpy
            depth_raw = depth_raw.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        depth_min = depth_raw.min()
        depth_max = depth_raw.max()
        depth_normalized = (depth_raw - depth_min) / (depth_max - depth_min + 1e-8)

        inference_time = time.time() - start_time

        return {
            "depth": depth_normalized.astype(np.float32),
            "depth_raw": depth_raw.astype(np.float32),
            "metadata": {
                "backend": self.backend.value,
                "variant": self.variant.name,
                "device": self.device,
                "inference_time_ms": inference_time * 1000,
                "shape": depth_normalized.shape,
            },
        }

    def _estimate_depth_coreml(self, image: Image.Image) -> dict:
        """Estimate depth using CoreML backend."""
        start_time = time.time()

        # Prepare input
        # CoreML expects specific input format
        image_array = np.array(image).astype(np.float32) / 255.0

        # Run inference
        prediction = self.model.predict({"image": image_array})

        # Extract depth from output
        depth_raw = prediction.get("depth", prediction.get("var_1071"))

        # Normalize to [0, 1]
        depth_min = depth_raw.min()
        depth_max = depth_raw.max()
        depth_normalized = (depth_raw - depth_min) / (depth_max - depth_min + 1e-8)

        inference_time = time.time() - start_time

        return {
            "depth": depth_normalized.astype(np.float32),
            "depth_raw": depth_raw.astype(np.float32),
            "metadata": {
                "backend": "coreml",
                "variant": self.variant.name,
                "device": "ane",
                "inference_time_ms": inference_time * 1000,
                "shape": depth_normalized.shape,
            },
        }

    def _estimate_depth_onnx(self, image: Image.Image) -> dict:
        """Estimate depth using ONNX runtime backend."""
        start_time = time.time()

        # Get model input details
        input_name = self.model.get_inputs()[0].name
        input_shape = self.model.get_inputs()[0].shape

        # Prepare input - resize and normalize
        # Default input size for Depth Anything V2 is 518x518
        target_size = (518, 518)
        if input_shape and len(input_shape) == 4 and input_shape[2] is not None and input_shape[3] is not None:
            target_size = (input_shape[2], input_shape[3])

        image_resized = image.resize(target_size, Image.Resampling.BILINEAR)
        image_array = np.array(image_resized).astype(np.float32) / 255.0

        # Normalize with ImageNet mean and std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_array = (image_array - mean) / std

        # Convert to NCHW format (batch, channels, height, width)
        image_array = image_array.transpose(2, 0, 1)
        image_array = np.expand_dims(image_array, axis=0)

        # Run inference
        outputs = self.model.run(None, {input_name: image_array})
        depth_raw = outputs[0]

        # Remove batch dimension and squeeze
        depth_raw = np.squeeze(depth_raw)

        # Normalize to [0, 1]
        depth_min = depth_raw.min()
        depth_max = depth_raw.max()
        depth_normalized = (depth_raw - depth_min) / (depth_max - depth_min + 1e-8)

        inference_time = time.time() - start_time

        return {
            "depth": depth_normalized.astype(np.float32),
            "depth_raw": depth_raw.astype(np.float32),
            "metadata": {
                "backend": "onnx",
                "variant": self.variant.name,
                "device": self.device,
                "inference_time_ms": inference_time * 1000,
                "shape": depth_normalized.shape,
            },
        }

    def estimate_depth_batch(
        self,
        images: list,
        batch_size: int = 4,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> list:
        """
        Estimate depth for multiple images in batches.

        Args:
            images: List of images (numpy arrays, PIL Images, or paths)
            batch_size: Number of images to process simultaneously
            output_size: Resize output to (height, width)

        Returns:
            List of depth estimation results (same format as estimate_depth)
        """
        results = []

        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]

            for image in batch:
                result = self.estimate_depth(image, output_size)
                results.append(result)

        return results

    def __repr__(self) -> str:
        return (
            "DepthAnythingV2Model(" f"variant={self.variant.name}, " f"backend={self.backend.name}, " f"device={self.device})"
        )


def safe_depth_estimation(
    image: Union[np.ndarray, Image.Image],
    model: DepthAnythingV2Model,
    fallback_size: int = 512,
) -> dict:
    """
    Robust depth estimation with automatic fallback on OOM errors.

    Args:
        image: Input image
        model: Depth estimation model
        fallback_size: Downscale size if OOM occurs

    Returns:
        Depth estimation result
    """
    try:
        return model.estimate_depth(image)

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.warning("OOM error, falling back to %spx resolution", fallback_size)

            # Downscale image
            if isinstance(image, np.ndarray):
                image_pil = Image.fromarray(image)
            else:
                image_pil = image

            image_pil.thumbnail((fallback_size, fallback_size), Image.Resampling.LANCZOS)

            # Retry with smaller size
            result = model.estimate_depth(image_pil)

            # Upscale depth map to original size
            original_size = image.shape[:2] if isinstance(image, np.ndarray) else image.size[::-1]
            if not SKIMAGE_AVAILABLE:
                raise ImportError("scikit-image is required for resizing. " "Install with: pip install scikit-image") from None

            result["depth"] = resize(
                result["depth"],
                original_size,
                order=1,
                preserve_range=True,
                anti_aliasing=True,
            )

            result["metadata"]["fallback_resolution"] = fallback_size
            return result
        raise
