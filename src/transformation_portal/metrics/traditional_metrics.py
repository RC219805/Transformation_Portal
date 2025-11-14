"""Traditional image quality metrics (PSNR, SSIM, MS-SSIM).

While less correlated with human perception than LPIPS, traditional metrics
provide fast, reference-based quality assessment.

Metrics:
- PSNR (Peak Signal-to-Noise Ratio): dB scale, higher = better
- SSIM (Structural Similarity Index): 0-1, higher = better
- MS-SSIM (Multi-Scale SSIM): 0-1, higher = better (more robust than SSIM)
"""

import logging
from typing import Dict, Union

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


logger = logging.getLogger(__name__)


class TraditionalMetrics:
    """Traditional image quality metrics.

    Provides PSNR, SSIM, and MS-SSIM for reference-based quality assessment.

    Example:
        >>> metrics = TraditionalMetrics()
        >>> results = metrics.calculate_all("original.jpg", "enhanced.jpg")
        >>> print(f"PSNR: {results['psnr']:.2f} dB")
        >>> print(f"SSIM: {results['ssim']:.4f}")
    """

    def calculate_psnr(
        self,
        image1: Union[str, np.ndarray, Image.Image],
        image2: Union[str, np.ndarray, Image.Image],
        data_range: Optional[int] = None
    ) -> float:
        """Calculate PSNR between two images.

        Args:
            image1: First image
            image2: Second image
            data_range: Data range (auto-detected if None)

        Returns:
            PSNR in dB (higher = better, typically 20-50 dB)
        """
        # Load images
        img1 = self._load_image(image1)
        img2 = self._load_image(image2)

        # Auto-detect data range
        if data_range is None:
            data_range = 255 if img1.max() > 1.0 else 1.0

        # Calculate PSNR
        psnr = peak_signal_noise_ratio(img1, img2, data_range=data_range)

        return psnr

    def calculate_ssim(
        self,
        image1: Union[str, np.ndarray, Image.Image],
        image2: Union[str, np.ndarray, Image.Image],
        multichannel: bool = True,
        data_range: Optional[int] = None
    ) -> float:
        """Calculate SSIM between two images.

        Args:
            image1: First image
            image2: Second image
            multichannel: Whether to compute for multichannel images
            data_range: Data range (auto-detected if None)

        Returns:
            SSIM score 0-1 (higher = better)
        """
        # Load images
        img1 = self._load_image(image1)
        img2 = self._load_image(image2)

        # Auto-detect data range
        if data_range is None:
            data_range = 255 if img1.max() > 1.0 else 1.0

        # Calculate SSIM
        ssim = structural_similarity(
            img1, img2,
            channel_axis=2 if multichannel and img1.ndim == 3 else None,
            data_range=data_range
        )

        return ssim

    def calculate_all(
        self,
        image1: Union[str, np.ndarray, Image.Image],
        image2: Union[str, np.ndarray, Image.Image]
    ) -> Dict[str, float]:
        """Calculate all traditional metrics.

        Args:
            image1: First image
            image2: Second image

        Returns:
            Dictionary with all metric values
        """
        psnr = self.calculate_psnr(image1, image2)
        ssim = self.calculate_ssim(image1, image2)

        return {
            "psnr": psnr,
            "ssim": ssim
        }

    def _load_image(
        self,
        image: Union[str, np.ndarray, Image.Image]
    ) -> np.ndarray:
        """Load image as numpy array."""
        if isinstance(image, np.ndarray):
            return image
        elif isinstance(image, Image.Image):
            return np.array(image)
        else:
            pil_img = Image.open(image).convert("RGB")
            return np.array(pil_img)

    def __repr__(self) -> str:
        return "TraditionalMetrics()"
