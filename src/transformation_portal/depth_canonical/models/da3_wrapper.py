"""Depth Anything V3 model wrapper for canonical depth pipeline.

Provides unified interface for DA3 models with metric depth estimation.
"""

import logging
import time
from typing import Any, Dict, Optional, Union, Tuple
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import DeviceType

logger = logging.getLogger(__name__)


class DA3ModelWrapper:
    """Wrapper for Depth Anything V3 models.

    Features:
    - PyTorch backend (CPU, CUDA, MPS)
    - Metric depth estimation
    - Automatic device detection
    - Lazy loading (model loaded on first inference)

    Example:
        >>> wrapper = DA3ModelWrapper(
        ...     model_id="depth-anything/Depth-Anything-V2-Metric-Hypersim-Large",
        ...     device=DeviceType.MPS
        ... )
        >>> result = wrapper.estimate(image)
        >>> depth_map = result['depth']
    """

    def __init__(
        self,
        model_id: str,
        device: DeviceType,
        dtype: str = "float32"
    ):
        """Initialize DA3 model wrapper.

        Args:
            model_id: HuggingFace model ID
            device: Target device
            dtype: Data type for inference
        """
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self._pipeline = None

    def _load_model(self) -> None:
        """Lazy load the model on first use."""
        if self._pipeline is not None:
            return

        try:
            from transformers import pipeline as hf_pipeline

            # Map device to transformers device
            if self.device == DeviceType.COREML:
                # CoreML not directly supported, fall back to MPS
                device_str = "mps"
            elif self.device == DeviceType.MPS:
                device_str = "mps"
            elif self.device == DeviceType.CUDA:
                device_str = "cuda"
            else:
                device_str = "cpu"

            # Load via transformers pipeline
            self._pipeline = hf_pipeline(
                task="depth-estimation",
                model=self.model_id,
                device=device_str if device_str != "mps" else 0
            )

            logger.info(
                "Loaded DA3 model: %s on device: %s",
                self.model_id,
                device_str
            )

        except ImportError as e:
            raise ImportError(
                "transformers required for DA3 models. "
                "Install with: pip install transformers torch"
            ) from e

    def estimate(
        self,
        image: Union[np.ndarray, Image.Image, str, Path],
        output_size: Optional[Tuple[int, int]] = None
    ) -> Dict[str, Any]:
        """Estimate depth from image.

        Args:
            image: Input image (PIL Image, numpy array, or path)
            output_size: Optional output size (height, width)

        Returns:
            Dictionary with:
                - 'depth': Normalized depth map [0, 1]
                - 'depth_raw': Raw depth predictions (metric)
                - 'metadata': Model info and timing
        """
        # Load model if needed
        self._load_model()

        # Convert input to PIL Image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            if image.dtype in (np.float32, np.float64):
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)

        # Run inference
        start_time = time.time()

        prediction = self._pipeline(image)
        depth_raw = prediction["depth"]

        # Convert to numpy
        if hasattr(depth_raw, "cpu"):
            # PyTorch tensor
            depth_raw = depth_raw.cpu().numpy()
        elif isinstance(depth_raw, Image.Image):
            depth_raw = np.array(depth_raw, dtype=np.float32)
        else:
            depth_raw = np.array(depth_raw, dtype=np.float32)

        # Normalize to [0, 1]
        depth_min = depth_raw.min()
        depth_max = depth_raw.max()
        depth_normalized = (depth_raw - depth_min) / (depth_max - depth_min + 1e-8)

        # Resize if requested
        if output_size is not None:
            from skimage.transform import resize
            depth_normalized = resize(
                depth_normalized,
                output_size,
                order=1,
                preserve_range=True,
                anti_aliasing=True
            )

        inference_time = time.time() - start_time

        return {
            "depth": depth_normalized.astype(np.float32),
            "depth_raw": depth_raw.astype(np.float32),
            "metadata": {
                "model_id": self.model_id,
                "device": self.device.value,
                "inference_time_ms": inference_time * 1000,
                "shape": depth_normalized.shape,
            }
        }
