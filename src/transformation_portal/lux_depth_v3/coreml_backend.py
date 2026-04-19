"""CoreML backend for published Depth Anything CoreML artifacts on Apple Silicon.

This release only supports registry-listed published CoreML packages. DA3-to-CoreML
conversion is intentionally not part of the supported runtime surface.
"""

from __future__ import annotations

import logging
import platform
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from transformation_portal.depth.models.coreml_wrapper import CoreMLDepthModel

logger = logging.getLogger(__name__)
_SUPPORTED_PUBLISHED_COREML_MODEL_IDS = {"apple/coreml-depth-anything-v2-small"}

# Lazy imports to avoid dependency errors
try:
    import coremltools as ct

    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False
    ct = None  # type: ignore

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

try:
    from transformers import pipeline as hf_pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    hf_pipeline = None  # type: ignore


class CoreMLDepthEstimator:
    """CoreML depth estimation for published Apple Silicon artifacts.

    Example:
        >>> estimator = CoreMLDepthEstimator("apple/coreml-depth-anything-v2-small")
        >>> depth = estimator.predict(image)
    """

    def __init__(
        self,
        model_id: str,
        cache_dir: Optional[Path] = None,
        force_reconvert: bool = False,
        revision: Optional[str] = None,
    ):
        """Initialize CoreML depth estimator.

        Args:
            model_id: Published Hugging Face CoreML model ID.
            cache_dir: Directory for cached CoreML models (default: ~/.cache/transformation_portal/coreml/)
            force_reconvert: Force reconversion even if cached model exists

        Raises:
            RuntimeError: If coremltools unavailable
            ValueError: If the model is not a supported published CoreML artifact
        """
        if not COREML_AVAILABLE:
            raise RuntimeError("coremltools not available. Install: pip install coremltools")

        if model_id not in _SUPPORTED_PUBLISHED_COREML_MODEL_IDS:
            raise ValueError(
                f"CoreML backend is only supported for published CoreML artifacts in this release. "
                f"Unsupported model_id={model_id!r}."
            )

        self.model_id = model_id
        self.revision = revision
        self.cache_dir = cache_dir or Path.home() / ".cache" / "transformation_portal" / "coreml"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load or convert model
        self.coreml_model = self._load_or_convert(force_reconvert=force_reconvert)

        logger.info(f"CoreML model ready: {model_id}")

    def _get_cache_path(self) -> Path:
        """Get cache path for converted CoreML model.

        Returns:
            Path to .mlpackage directory
        """
        # Sanitize model ID for filesystem
        model_name = self.model_id.replace("/", "_").replace("-", "_")
        if not self.revision:
            return self.cache_dir / f"{model_name}.mlpackage"
        revision_token = self.revision.replace("/", "_").replace("-", "_")
        return self.cache_dir / f"{model_name}_{revision_token}.mlpackage"

    def _load_or_convert(self, force_reconvert: bool = False) -> Any:
        """Load cached CoreML model or convert from PyTorch.

        Args:
            force_reconvert: Force reconversion even if cache exists

        Returns:
            CoreML model instance
        """
        cache_path = self._get_cache_path()

        if self.model_id in _SUPPORTED_PUBLISHED_COREML_MODEL_IDS:
            return self._load_published_coreml_model(cache_path)

        if cache_path.exists() and not force_reconvert:
            logger.info(f"Loading cached CoreML model: {cache_path}")
            try:
                return ct.models.MLModel(str(cache_path))
            except Exception as e:
                logger.warning(f"Failed to load cached model, reconverting: {e}")

        logger.info(f"Converting {self.model_id} to CoreML (this may take 5-10 minutes)...")
        start_time = time.time()

        try:
            model = self._convert_pytorch_to_coreml(cache_path)
            elapsed = time.time() - start_time
            logger.info(f"CoreML conversion completed in {elapsed:.1f}s")
            return model
        except Exception as e:
            logger.error(f"CoreML conversion failed: {e}")
            raise ValueError(f"Failed to convert {self.model_id} to CoreML: {e}") from e

    def _load_published_coreml_model(self, cache_path: Path) -> Any:
        """Load a published CoreML artifact from Hugging Face."""
        from huggingface_hub import hf_hub_download  # pylint: disable=import-outside-toplevel

        if cache_path.exists():
            return CoreMLDepthModel(cache_path)

        model_path = Path(
            hf_hub_download(  # nosec B615
                repo_id=self.model_id,
                filename="DepthAnythingV2SmallF16.mlpackage",
                cache_dir=self.cache_dir,
                revision=self.revision,
            )
        )
        if not cache_path.exists():
            try:
                cache_path.symlink_to(model_path, target_is_directory=model_path.is_dir())
            except OSError:
                logger.debug(
                    "CoreML cache alias unavailable for %s@%s; using Hub cache path directly",
                    self.model_id,
                    self.revision or "unpinned",
                )
        stable_path = cache_path if cache_path.exists() else model_path
        return CoreMLDepthModel(stable_path)

    def _convert_pytorch_to_coreml(self, output_path: Path) -> Any:
        """Convert PyTorch depth model to CoreML with ANE optimization.

        Conversion strategy:
        1. Load PyTorch model from HuggingFace
        2. Trace with example input (1024×1024 RGB)
        3. Convert to CoreML with FP16 precision
        4. Set compute units to ALL (CPU + GPU + ANE)
        5. Cache for future use

        Args:
            output_path: Path to save .mlpackage

        Returns:
            CoreML model instance
        """
        # Load PyTorch model
        logger.info(f"Loading PyTorch model: {self.model_id}")
        pipe = hf_pipeline(
            "depth-estimation",
            model=self.model_id,
            device="cpu",  # CPU for conversion
        )

        # Extract underlying model
        model = pipe.model
        model.eval()

        # Create example input (1024×1024 RGB)
        # Note: Adjust size based on model requirements
        example_input = torch.randn(1, 3, 1024, 1024)

        logger.info("Tracing PyTorch model...")
        # Trace model
        with torch.no_grad():
            traced_model = torch.jit.trace(model, example_input)

        logger.info("Converting to CoreML...")
        # Convert to CoreML with ANE optimization
        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="input", shape=(1, 3, 1024, 1024))],
            outputs=[ct.TensorType(name="depth")],
            compute_precision=ct.precision.FLOAT16,  # FP16 for ANE
            compute_units=ct.ComputeUnit.ALL,  # Use ANE when possible
            minimum_deployment_target=ct.target.macOS14,  # M4 support
        )

        # Save to cache
        logger.info(f"Saving CoreML model to {output_path}")
        mlmodel.save(str(output_path))

        return mlmodel

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run depth estimation on CoreML model.

        Args:
            image: RGB image array (H, W, 3) in [0, 1] range

        Returns:
            Depth map (H, W) in [0, 1] range
        """
        # Preprocess: HWC → CHW, add batch dimension
        if image.ndim == 3:
            image = np.transpose(image, (2, 0, 1))  # HWC → CHW

        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)  # CHW → BCHW

        # Ensure float32 (CoreML input requirement)
        image = image.astype(np.float32)

        depth = self.coreml_model.predict(image)
        if depth.ndim == 4:
            depth = depth[0, 0]
        elif depth.ndim == 3:
            depth = depth[0]

        return depth


def should_use_coreml(config: Any, force: bool = False) -> bool:
    """Check if CoreML should be used for inference.

    Requirements:
    - macOS with Apple Silicon (arm64)
    - Config flag enabled (use_coreml=True)
    - coremltools available

    Args:
        config: Configuration object with use_coreml attribute
        force: Force enable even if not optimal (for testing)

    Returns:
        True if CoreML should be used
    """
    # User must opt-in
    if not getattr(config, "use_coreml", False) and not force:
        return False

    # Only on macOS with Apple Silicon
    if platform.system() != "Darwin":
        logger.debug("CoreML disabled: not macOS")
        return False

    if platform.machine() != "arm64":
        logger.debug("CoreML disabled: not Apple Silicon")
        return False

    # CoreML tools must be available
    if not COREML_AVAILABLE:
        logger.warning("CoreML requested but coremltools not available. Install: pip install coremltools")
        return False

    if not TORCH_AVAILABLE:
        logger.warning("CoreML requires torch for conversion")
        return False

    if not TRANSFORMERS_AVAILABLE:
        logger.warning("CoreML requires transformers for model loading")
        return False

    return True


def get_coreml_cache_stats(cache_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Get statistics about CoreML model cache.

    Args:
        cache_dir: Cache directory (default: ~/.cache/transformation_portal/coreml/)

    Returns:
        Dictionary with cache statistics
    """
    cache_dir = cache_dir or Path.home() / ".cache" / "transformation_portal" / "coreml"

    if not cache_dir.exists():
        return {
            "cache_dir": str(cache_dir),
            "exists": False,
            "model_count": 0,
            "total_size_mb": 0,
        }

    # Count .mlpackage directories
    models = list(cache_dir.glob("*.mlpackage"))

    # Calculate total size
    total_size = 0
    for model_path in models:
        for file_path in model_path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size

    return {
        "cache_dir": str(cache_dir),
        "exists": True,
        "model_count": len(models),
        "models": [m.name for m in models],
        "total_size_mb": total_size / (1024 * 1024),
    }


def clear_coreml_cache(cache_dir: Optional[Path] = None) -> int:
    """Clear CoreML model cache.

    Args:
        cache_dir: Cache directory (default: ~/.cache/transformation_portal/coreml/)

    Returns:
        Number of models deleted
    """
    import shutil

    cache_dir = cache_dir or Path.home() / ".cache" / "transformation_portal" / "coreml"

    if not cache_dir.exists():
        return 0

    models = list(cache_dir.glob("*.mlpackage"))
    count = 0

    for model_path in models:
        try:
            shutil.rmtree(model_path)
            count += 1
            logger.info(f"Deleted cached model: {model_path.name}")
        except Exception as e:
            logger.warning(f"Failed to delete {model_path.name}: {e}")

    return count
