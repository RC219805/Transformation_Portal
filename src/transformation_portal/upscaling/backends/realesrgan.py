"""Real-ESRGAN upscaler backend (optional ML tier).

Commercial-safe ML upscaling with superior quality over bicubic.
Local implementation using BasicSR (realesrgan package is unmaintained/banned).

License:
    - Real-ESRGAN Model: BSD-3-Clause (commercial-safe)
    - BasicSR: Apache 2.0 (commercial-safe)

Model Variants:
    - RealESRGAN_x2plus: 2x upscaling, best quality for 2x
    - RealESRGAN_x4plus: 4x upscaling, best quality for 4x

References:
    - Paper: https://arxiv.org/abs/2107.10833
    - Code: https://github.com/xinntao/Real-ESRGAN
    - License: https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE
    - Model Weights: https://github.com/xinntao/Real-ESRGAN/releases
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class RealESRGANUpscaler:
    """Real-ESRGAN upscaling backend (local implementation).

    Requires ML dependencies (torch, basicsr).
    Commercial-safe (BSD-3-Clause license).

    Performance:
        - ~10-30 images/hour for 4K upscaling to 8K (GPU)
        - ~2-5 images/hour on CPU
        - Memory: ~2-4GB GPU memory per image
        - Quality: Excellent, preserves fine details and textures

    Usage:
        >>> upscaler = RealESRGANUpscaler(device="cuda", model="RealESRGAN_x2plus")
        >>> image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
        >>> upscaled = upscaler.upscale(image, scale_factor=2.0)
        >>> upscaled.shape
        (2000, 2000, 3)
    """

    # Backend ID for registry (class-level constant)
    BACKEND_ID = "realesrgan"

    def __init__(
        self,
        device: str = "cpu",
        model: str = "RealESRGAN_x2plus",
        half_precision: bool = False,
    ):
        """Initialize Real-ESRGAN upscaler.

        Args:
            device: Device to use (cpu, cuda, mps).
            model: Model name (RealESRGAN_x2plus, RealESRGAN_x4plus).
            half_precision: Use FP16 for faster inference (GPU only).

        Raises:
            ImportError: If torch or basicsr not installed.
            ValueError: If model name is invalid.
        """
        # SECURITY: BasicSR is blocked due to CVE-2024-27763
        # Real-ESRGAN backend is currently unavailable until a safe alternative is implemented
        raise ImportError(
            "Real-ESRGAN backend is currently unavailable due to CVE-2024-27763 in BasicSR. "
            "Use 'bicubic' backend instead. "
            "A vendored safe implementation will be added in a future update."
        )

        # The code below is disabled until BasicSR security issue is resolved
        # - PyTorch dependency check (torch import)
        # - BasicSR RRDBNet model import (BLOCKED)
        # - Device/model/precision configuration
        # - Model name validation

    @property
    def name(self) -> str:
        """Backend name."""
        return self.BACKEND_ID

    @property
    def requires_ml(self) -> bool:
        """Requires ML dependencies."""
        return True

    def _download_model_weights(self, model_path: Path) -> None:
        """Download model weights from GitHub releases.

        Args:
            model_path: Path to save model weights.
        """
        import urllib.request

        # Model URLs from official Real-ESRGAN releases
        model_urls = {
            "RealESRGAN_x2plus": ("https://github.com/xinntao/Real-ESRGAN/releases/download/" "v0.2.1/RealESRGAN_x2plus.pth"),
            "RealESRGAN_x4plus": ("https://github.com/xinntao/Real-ESRGAN/releases/download/" "v0.1.0/RealESRGAN_x4plus.pth"),
        }

        url = model_urls[self._model_name]
        logger.info(f"Downloading model weights from {url}")
        logger.info(f"Saving to {model_path}")

        model_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            urllib.request.urlretrieve(url, model_path)
            logger.info("Model weights downloaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to download model weights: {e}") from e

    def _load_model(self):
        """Lazy load Real-ESRGAN model.

        Downloads model weights automatically on first use.
        Weights are cached in `weights/` directory.
        """
        # SECURITY: This method is disabled due to CVE-2024-27763 in BasicSR
        raise ImportError(
            "Real-ESRGAN backend is currently unavailable due to CVE-2024-27763. "
            "A safe alternative implementation is required."
        )

        # The code below is disabled until BasicSR security issue is resolved:
        # - Model singleton check
        # - PyTorch import
        # - RRDBNet model class import (BLOCKED)

        logger.info(f"Loading Real-ESRGAN model: {self._model_name}")

        # Model configurations
        if self._model_name == "RealESRGAN_x2plus":
            # Best for 2x upscaling
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=2,
            )
            self._netscale = 2

        elif self._model_name == "RealESRGAN_x4plus":
            # Best for 4x upscaling
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            )
            self._netscale = 4

        else:
            raise ValueError(f"Unknown model: {self._model_name}")

        # Model weights path
        weights_dir = Path("weights")
        weights_dir.mkdir(exist_ok=True)
        model_path = weights_dir / f"{self._model_name}.pth"

        # Download if not cached
        if not model_path.exists():
            self._download_model_weights(model_path)

        # Load weights
        try:
            loadnet = torch.load(model_path, map_location=torch.device(self._device))

            # Handle different checkpoint formats
            if "params_ema" in loadnet:
                keyname = "params_ema"
            elif "params" in loadnet:
                keyname = "params"
            else:
                keyname = "model"

            model.load_state_dict(loadnet[keyname], strict=True)

        except Exception as e:
            raise RuntimeError(f"Failed to load model weights from {model_path}: {e}") from e

        # Move to device and set to eval mode
        model.eval()
        model = model.to(self._device)

        if self._half_precision:
            model = model.half()

        self._model = model

        logger.info(f"Real-ESRGAN loaded: {self._model_name} on {self._device} " f"(half={self._half_precision})")

    def _preprocess_image(self, image: np.ndarray) -> "torch.Tensor":
        """Convert numpy image to torch tensor.

        Args:
            image: Input image (H, W, 3), RGB uint8 [0-255].

        Returns:
            Torch tensor (1, 3, H, W), float32 [0-1].
        """
        import torch

        # Input is RGB (protocol standard), use directly
        # Real-ESRGAN model works with RGB internally
        image_rgb = image.copy()

        # Convert to float32 [0, 1]
        image_float = image_rgb.astype(np.float32) / 255.0

        # HWC to CHW
        image_chw = np.transpose(image_float, (2, 0, 1))

        # Add batch dimension
        image_tensor = torch.from_numpy(image_chw).unsqueeze(0)

        # Move to device
        image_tensor = image_tensor.to(self._device)

        if self._half_precision:
            image_tensor = image_tensor.half()

        return image_tensor

    def _postprocess_output(self, output: "torch.Tensor") -> np.ndarray:
        """Convert torch tensor to numpy image.

        Args:
            output: Torch tensor (1, 3, H, W), float32 [0-1].

        Returns:
            Numpy image (H, W, 3), RGB uint8 [0-255].
        """
        # Remove batch dimension and move to CPU
        output = output.squeeze(0).cpu().float()

        # CHW to HWC
        output_np = output.numpy().transpose(1, 2, 0)

        # Clip to [0, 1] and convert to uint8
        output_np = np.clip(output_np, 0, 1)
        output_uint8 = (output_np * 255.0).round().astype(np.uint8)

        # Output is RGB (protocol standard), return directly
        return output_uint8

    def upscale(
        self,
        image: np.ndarray,
        scale_factor: float,
    ) -> np.ndarray:
        """Upscale using Real-ESRGAN.

        Args:
            image: Input image (H, W, 3), uint8 or float32.
            scale_factor: Upscaling factor.
                         For RealESRGAN_x2plus: 2.0 recommended.
                         For RealESRGAN_x4plus: 4.0 recommended.
                         Note: Non-integer scales will use post-resize.

        Returns:
            Upscaled image with same dtype as input.

        Raises:
            RuntimeError: If upscaling fails.
        """
        import torch
        import torch.nn.functional as F

        # Lazy load model
        self._load_model()

        # Handle dtype conversion
        input_dtype = image.dtype
        if image.dtype == np.float32:
            # Convert float32 [0, 1] to uint8 [0, 255]
            image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            image_uint8 = image

        try:
            # Preprocess to tensor
            input_tensor = self._preprocess_image(image_uint8)

            # Run inference
            with torch.no_grad():
                output_tensor = self._model(input_tensor)

            # Postprocess to numpy
            output_rgb = self._postprocess_output(output_tensor)

            # Handle non-native scale factors (e.g., 1.5x, 3x)
            if abs(scale_factor - self._netscale) > 0.01:
                # Post-resize to desired scale
                h, w = output_rgb.shape[:2]
                target_h = int(image_uint8.shape[0] * scale_factor)
                target_w = int(image_uint8.shape[1] * scale_factor)

                from PIL import Image

                pil_img = Image.fromarray(output_rgb[..., ::-1])  # BGR to RGB
                pil_resized = pil_img.resize((target_w, target_h), Image.BICUBIC)
                output_rgb = np.array(pil_resized)[..., ::-1]  # RGB to BGR

            # Convert back to original dtype
            if input_dtype == np.float32:
                return output_rgb.astype(np.float32) / 255.0
            else:
                return output_rgb

        except Exception as e:
            raise RuntimeError(f"Real-ESRGAN upscaling failed: {e}") from e
