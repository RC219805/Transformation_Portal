"""
CoreML wrapper for Depth Anything V2 optimized for Apple Neural Engine.

Provides optimized inference on M4 Max with ANE (Apple Neural Engine) acceleration.
Expected performance: 24ms @ 518x518, 65ms @ 1024x1024
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False

logger = logging.getLogger(__name__)


class CoreMLDepthModel:
    """
    CoreML-optimized depth estimation model for Apple Silicon.

    Features:
    - Apple Neural Engine (ANE) acceleration
    - FP16 precision for optimal ANE performance
    - Minimal memory overhead
    - 24-65ms inference on M4 Max

    Example:
        >>> model = CoreMLDepthModel("DepthAnythingV2SmallF16.mlpackage")
        >>> depth = model.predict(image)
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        compute_units: str = "ALL",
    ):
        """
        Initialize CoreML depth model.

        Args:
            model_path: Path to .mlpackage or .mlmodel file
            compute_units: Compute unit selection ("ALL", "CPU_AND_NE", "CPU_ONLY")
                          "ALL" auto-selects optimal (ANE > GPU > CPU)
        """
        if not COREML_AVAILABLE:
            raise ImportError(
                "coremltools required for CoreML models. "
                "Install with: pip install coremltools"
            )

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Configure compute units
        compute_unit_map = {
            "ALL": ct.ComputeUnit.ALL,
            "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
            "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        }
        self.compute_unit = compute_unit_map.get(compute_units, ct.ComputeUnit.ALL)

        # Load model
        self.model = self._load_model()

        logger.info(f"Loaded CoreML model from {model_path} (compute={compute_units})")

    def _load_model(self):
        """Load CoreML model with error handling."""
        try:
            model = ct.models.MLModel(
                str(self.model_path),
                compute_units=self.compute_unit
            )
            return model
        except Exception as e:
            logger.error(f"Failed to load CoreML model: {e}")
            raise

    def predict(
        self,
        image: Union[np.ndarray, Image.Image],
        normalize_input: bool = True,
    ) -> np.ndarray:
        """
        Run depth prediction on input image.

        Args:
            image: Input RGB image
            normalize_input: Normalize to [0, 1] if True

        Returns:
            Depth map as numpy array (HxW, float32)
        """
        # Prepare input
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Ensure float32
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        # Normalize if needed
        if normalize_input and image.max() > 1.0:
            image = image / 255.0

        # Run inference
        try:
            prediction = self.model.predict({'image': image})

            # Extract depth (output name may vary)
            if 'depth' in prediction:
                depth = prediction['depth']
            else:
                # Fallback to first output
                depth = next(iter(prediction.values()))

            return depth.astype(np.float32)

        except Exception as e:
            logger.error(f"CoreML prediction failed: {e}")
            raise

    def predict_batch(
        self,
        images: list,
        normalize_input: bool = True,
    ) -> list:
        """
        Predict depth for batch of images.

        Note: CoreML processes sequentially, no batch optimization.

        Args:
            images: List of input images
            normalize_input: Normalize inputs to [0, 1]

        Returns:
            List of depth maps
        """
        return [
            self.predict(img, normalize_input=normalize_input)
            for img in images
        ]

    def get_model_info(self) -> dict:
        """Get model metadata."""
        spec = self.model.get_spec()

        return {
            'input_description': spec.description.input,
            'output_description': spec.description.output,
            'compute_unit': str(self.compute_unit),
            'model_path': str(self.model_path),
        }

    def __repr__(self) -> str:
        return f"CoreMLDepthModel(path={self.model_path.name}, compute={self.compute_unit})"
