"""FID (Fréchet Inception Distance) metric.

FID measures distribution matching between real and generated images using
Inception-v3 features. Lower FID = generated distribution closer to real distribution.

Key properties:
- Lower score = better (0 = identical distributions)
- Typical ranges:
  - FID < 10: Excellent (nearly indistinguishable)
  - FID 10-20: Very good
  - FID 20-50: Good
  - FID > 50: Poor (noticeable distribution mismatch)

For luxury real estate:
- Ensures enhanced images remain within authentic photography manifold
- Detects drift toward synthetic/CGI appearance
- Validates enhancement doesn't introduce unrealistic artifacts
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from scipy import linalg

try:
    from torchvision.models import inception_v3, Inception_V3_Weights
    INCEPTION_AVAILABLE = True
except ImportError:
    INCEPTION_AVAILABLE = False
    logging.warning("torchvision not available")


logger = logging.getLogger(__name__)


class FIDMetric:
    """Fréchet Inception Distance metric.

    Measures distribution similarity between two sets of images using
    Inception-v3 feature statistics.

    Example:
        >>> metric = FIDMetric()
        >>> real_images = ["real1.jpg", "real2.jpg", "real3.jpg", ...]
        >>> enhanced_images = ["enh1.jpg", "enh2.jpg", "enh3.jpg", ...]
        >>> fid_score = metric.calculate(real_images, enhanced_images)
        >>> print(f"FID score: {fid_score:.2f}")
        >>> if fid_score < 10:
        ...     print("Excellent - enhanced images match real distribution")
    """

    # Interpretation thresholds
    EXCELLENT_THRESHOLD = 10.0
    VERY_GOOD_THRESHOLD = 20.0
    GOOD_THRESHOLD = 50.0

    def __init__(
        self,
        device: Optional[str] = None,
        dims: int = 2048
    ):
        """Initialize FID metric.

        Args:
            device: Computation device (auto-detected if None)
            dims: Dimensionality of Inception features (2048 for pool3 layer)

        Raises:
            ImportError: If torchvision not available
        """
        if not INCEPTION_AVAILABLE:
            raise ImportError(
                "torchvision required for FID. "
                "Install with: pip install torchvision"
            )

        self.device = device or self._detect_device()
        self.dims = dims

        logger.info(f"Initializing FID metric on {self.device}")

        # Load Inception-v3
        self.model = inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1,
            transform_input=False
        ).to(self.device)

        self.model.eval()

        # Remove final layers to get features
        self.model.fc = torch.nn.Identity()

        logger.info("FID metric initialized")

    def _detect_device(self) -> str:
        """Auto-detect optimal device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def calculate(
        self,
        images1: List[Union[str, Path, Image.Image, np.ndarray]],
        images2: List[Union[str, Path, Image.Image, np.ndarray]],
        batch_size: int = 32
    ) -> float:
        """Calculate FID between two sets of images.

        Args:
            images1: First set of images (e.g., real images)
            images2: Second set of images (e.g., enhanced images)
            batch_size: Batch size for feature extraction

        Returns:
            FID score (lower = better)
        """
        logger.info(f"Calculating FID for {len(images1)} vs {len(images2)} images")

        # Extract features for both sets
        features1 = self._extract_features(images1, batch_size)
        features2 = self._extract_features(images2, batch_size)

        # Calculate statistics
        mu1, sigma1 = self._calculate_statistics(features1)
        mu2, sigma2 = self._calculate_statistics(features2)

        # Calculate FID
        fid = self._calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

        logger.info(f"FID score: {fid:.4f}")

        return fid

    def _extract_features(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        batch_size: int
    ) -> np.ndarray:
        """Extract Inception features for images.

        Args:
            images: List of images
            batch_size: Batch size

        Returns:
            Feature array (N, dims)
        """
        features = []

        # Process in batches
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]

            # Prepare batch
            batch_tensors = [self._prepare_image(img) for img in batch]
            batch_tensor = torch.cat(batch_tensors, dim=0)

            # Extract features
            with torch.no_grad():
                batch_features = self.model(batch_tensor)

            features.append(batch_features.cpu().numpy())

        # Concatenate all batches
        features = np.concatenate(features, axis=0)

        return features

    def _prepare_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray]
    ) -> torch.Tensor:
        """Prepare image for Inception.

        Args:
            image: Input image

        Returns:
            Tensor (1, 3, 299, 299)
        """
        # Load as PIL Image
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Resize to 299x299 (Inception input size)
        pil_image = pil_image.resize((299, 299), Image.Resampling.BILINEAR)

        # Convert to tensor
        array = np.array(pil_image).astype(np.float32)
        array = array.transpose(2, 0, 1)  # HWC -> CHW
        tensor = torch.from_numpy(array).unsqueeze(0)

        # Normalize to [0, 1]
        tensor = tensor / 255.0

        # Normalize to Inception range [-1, 1]
        tensor = tensor * 2.0 - 1.0

        return tensor.to(self.device)

    def _calculate_statistics(
        self,
        features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate mean and covariance of features.

        Args:
            features: Feature array (N, dims)

        Returns:
            Tuple of (mean, covariance)
        """
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)

        return mu, sigma

    def _calculate_frechet_distance(
        self,
        mu1: np.ndarray,
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray,
        eps: float = 1e-6
    ) -> float:
        """Calculate Fréchet distance between two Gaussians.

        Args:
            mu1: Mean of first distribution
            sigma1: Covariance of first distribution
            mu2: Mean of second distribution
            sigma2: Covariance of second distribution
            eps: Epsilon for numerical stability

        Returns:
            Fréchet distance
        """
        # Calculate mean difference
        diff = mu1 - mu2

        # Calculate sqrt of product of covariances
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

        # Handle numerical errors
        if not np.isfinite(covmean).all():
            logger.warning("FID calculation produced singular product; adding epsilon")
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Handle complex numbers from sqrtm
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f"Imaginary component {m}")
            covmean = covmean.real

        # Calculate FID
        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)

        return float(fid)

    def interpret(self, fid_score: float) -> Dict[str, Any]:
        """Interpret FID score.

        Args:
            fid_score: FID score

        Returns:
            Dictionary with interpretation
        """
        if fid_score < self.EXCELLENT_THRESHOLD:
            quality = "excellent"
            description = "Nearly indistinguishable from real distribution"
        elif fid_score < self.VERY_GOOD_THRESHOLD:
            quality = "very_good"
            description = "Very close to real distribution"
        elif fid_score < self.GOOD_THRESHOLD:
            quality = "good"
            description = "Similar to real distribution"
        else:
            quality = "poor"
            description = "Noticeable distribution mismatch"

        return {
            "fid_score": fid_score,
            "quality": quality,
            "description": description,
            "photorealistic": fid_score < self.VERY_GOOD_THRESHOLD,
            "acceptable_for_enhancement": fid_score < self.GOOD_THRESHOLD
        }

    def __repr__(self) -> str:
        return f"FIDMetric(device='{self.device}', dims={self.dims})"
