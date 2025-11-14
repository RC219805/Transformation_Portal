"""LPIPS (Learned Perceptual Image Patch Similarity) metric.

LPIPS uses deep networks trained on human perceptual similarity judgments.
Significantly more aligned with human perception than PSNR or SSIM.

Key properties:
- Lower score = more perceptually similar
- Typical range: 0.0 (identical) to ~0.5 (very different)
- Threshold for "similar": < 0.1
- Threshold for "different": > 0.3

Uses pre-trained networks (AlexNet, VGG, SqueezeNet) as perceptual backbones.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    logging.warning(
        "LPIPS not available. Install with: pip install lpips"
    )


logger = logging.getLogger(__name__)


class LPIPSMetric:
    """LPIPS perceptual similarity metric.

    Measures perceptual similarity using deep neural networks trained
    on human judgments. More accurate than traditional metrics for
    assessing enhancement quality.

    Example:
        >>> metric = LPIPSMetric(network='alex')
        >>> distance = metric.calculate("original.jpg", "enhanced.jpg")
        >>> print(f"LPIPS distance: {distance:.4f}")
        >>> if distance < 0.1:
        ...     print("Images are perceptually very similar")
        >>> elif distance < 0.3:
        ...     print("Images are somewhat similar")
        >>> else:
        ...     print("Images are perceptually different")
    """

    # Interpretation thresholds
    VERY_SIMILAR_THRESHOLD = 0.1
    SIMILAR_THRESHOLD = 0.2
    DIFFERENT_THRESHOLD = 0.3

    def __init__(
        self,
        network: str = 'alex',  # 'alex', 'vgg', 'squeeze'
        device: Optional[str] = None,
        spatial: bool = False
    ):
        """Initialize LPIPS metric.

        Args:
            network: Perceptual network ('alex', 'vgg', 'squeeze')
            device: Computation device (auto-detected if None)
            spatial: Return spatial map instead of scalar

        Raises:
            ImportError: If lpips not installed
        """
        if not LPIPS_AVAILABLE:
            raise ImportError(
                "LPIPS required. Install with: pip install lpips"
            )

        self.network = network
        self.device = device or self._detect_device()
        self.spatial = spatial

        logger.info(f"Initializing LPIPS ({network}) on {self.device}")

        # Load LPIPS model
        self.model = lpips.LPIPS(
            net=network,
            spatial=spatial
        ).to(self.device)

        self.model.eval()

        logger.info("LPIPS metric initialized")

    def _detect_device(self) -> str:
        """Auto-detect optimal device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def calculate(
        self,
        image1: Union[str, Path, Image.Image, np.ndarray, torch.Tensor],
        image2: Union[str, Path, Image.Image, np.ndarray, torch.Tensor],
        normalize: bool = True
    ) -> float:
        """Calculate LPIPS distance between two images.

        Args:
            image1: First image
            image2: Second image
            normalize: Whether to normalize inputs to [-1, 1]

        Returns:
            LPIPS distance (lower = more similar)
        """
        # Load and prepare images
        tensor1 = self._prepare_image(image1, normalize)
        tensor2 = self._prepare_image(image2, normalize)

        # Calculate LPIPS
        with torch.no_grad():
            distance = self.model(tensor1, tensor2)

        # Return scalar value
        if self.spatial:
            return distance.mean().item()
        else:
            return distance.item()

    def calculate_batch(
        self,
        images1: List[Union[str, Path, Image.Image, np.ndarray]],
        images2: List[Union[str, Path, Image.Image, np.ndarray]],
        normalize: bool = True
    ) -> np.ndarray:
        """Calculate LPIPS for batch of image pairs.

        Args:
            images1: List of first images
            images2: List of second images
            normalize: Whether to normalize inputs

        Returns:
            Array of LPIPS distances
        """
        assert len(images1) == len(images2), "Lists must have same length"

        distances = []

        for img1, img2 in zip(images1, images2):
            distance = self.calculate(img1, img2, normalize)
            distances.append(distance)

        return np.array(distances)

    def interpret(self, distance: float) -> Dict[str, Any]:
        """Interpret LPIPS distance.

        Args:
            distance: LPIPS distance

        Returns:
            Dictionary with interpretation
        """
        if distance < self.VERY_SIMILAR_THRESHOLD:
            similarity = "very_similar"
            quality = "excellent"
        elif distance < self.SIMILAR_THRESHOLD:
            similarity = "similar"
            quality = "good"
        elif distance < self.DIFFERENT_THRESHOLD:
            similarity = "somewhat_different"
            quality = "acceptable"
        else:
            similarity = "different"
            quality = "poor"

        return {
            "distance": distance,
            "similarity": similarity,
            "quality": quality,
            "preserve_details": distance < self.SIMILAR_THRESHOLD,
            "acceptable_for_enhancement": distance < self.DIFFERENT_THRESHOLD
        }

    def _prepare_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor],
        normalize: bool
    ) -> torch.Tensor:
        """Prepare image as tensor for LPIPS.

        Args:
            image: Input image
            normalize: Normalize to [-1, 1]

        Returns:
            Tensor (1, 3, H, W)
        """
        if isinstance(image, torch.Tensor):
            tensor = image
        else:
            # Load as PIL Image
            if isinstance(image, (str, Path)):
                pil_image = Image.open(image).convert("RGB")
            elif isinstance(image, Image.Image):
                pil_image = image.convert("RGB")
            elif isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")

            # Convert to tensor
            array = np.array(pil_image).astype(np.float32)
            array = array.transpose(2, 0, 1)  # HWC -> CHW
            tensor = torch.from_numpy(array)

        # Ensure 4D (B, C, H, W)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)

        # Normalize to [-1, 1] if requested
        if normalize and tensor.max() > 1.0:
            tensor = tensor / 255.0

        if normalize:
            tensor = tensor * 2.0 - 1.0

        return tensor.to(self.device)

    def __repr__(self) -> str:
        return (
            f"LPIPSMetric(network='{self.network}', "
            f"device='{self.device}', spatial={self.spatial})"
        )
