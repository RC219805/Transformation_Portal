"""
Quality metrics computation with categorization.

Provides standardized metrics for image quality assessment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


@dataclass
class QualityMetrics:
    """Container for quality metrics."""
    ssim: Optional[float] = None
    psnr: Optional[float] = None
    lpips: Optional[float] = None
    nima: Optional[float] = None
    mae: Optional[float] = None
    mse: Optional[float] = None

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary, excluding None values."""
        return {
            k: v for k, v in {
                "ssim": self.ssim,
                "psnr": self.psnr,
                "lpips": self.lpips,
                "nima": self.nima,
                "mae": self.mae,
                "mse": self.mse,
            }.items() if v is not None
        }

    def __str__(self) -> str:
        """Format metrics as string."""
        lines = []
        if self.ssim is not None:
            lines.append(f"SSIM: {self.ssim:.4f}")
        if self.psnr is not None:
            lines.append(f"PSNR: {self.psnr:.2f} dB")
        if self.lpips is not None:
            lines.append(f"LPIPS: {self.lpips:.4f}")
        if self.nima is not None:
            lines.append(f"NIMA: {self.nima:.4f}")
        if self.mae is not None:
            lines.append(f"MAE: {self.mae:.4f}")
        if self.mse is not None:
            lines.append(f"MSE: {self.mse:.4f}")
        return "\n".join(lines) if lines else "No metrics available"


class MetricsComputer:
    """
    Compute quality metrics with categorization.

    Supports multiple metrics:
    - SSIM: Structural similarity (higher is better, 0-1)
    - PSNR: Peak signal-to-noise ratio (higher is better, dB)
    - LPIPS: Perceptual similarity (lower is better, 0-1)
    - NIMA: Neural image assessment (higher is better, 1-10)
    - MAE: Mean absolute error (lower is better)
    - MSE: Mean squared error (lower is better)
    """

    def __init__(self):
        """Initialize metrics computer."""
        if not NUMPY_AVAILABLE:
            raise ImportError("MetricsComputer requires numpy")

        # Metric weights for weighted score
        self.weights = {
            "ssim": 0.3,
            "psnr": 0.2,
            "lpips": 0.3,
            "nima": 0.2
        }

        # Cache for heavy models (LPIPS, NIMA)
        self._lpips_model = None
        self._nima_model = None

    def compute(
        self,
        reference: np.ndarray,
        processed: np.ndarray,
        metrics: Optional[list[str]] = None
    ) -> QualityMetrics:
        """
        Compute quality metrics.

        Args:
            reference: Reference image (H, W, C) or (C, H, W), values 0-255 or 0-1
            processed: Processed image (same format as reference)
            metrics: List of metrics to compute (None = all available)

        Returns:
            QualityMetrics with computed values
        """
        # Normalize inputs
        ref_norm = self._normalize_image(reference)
        proc_norm = self._normalize_image(processed)

        if metrics is None:
            metrics = ["ssim", "psnr", "mae", "mse"]  # Fast metrics by default

        result = QualityMetrics()

        if "ssim" in metrics:
            result.ssim = self._compute_ssim(ref_norm, proc_norm)

        if "psnr" in metrics:
            result.psnr = self._compute_psnr(ref_norm, proc_norm)

        if "mae" in metrics:
            result.mae = self._compute_mae(ref_norm, proc_norm)

        if "mse" in metrics:
            result.mse = self._compute_mse(ref_norm, proc_norm)

        if "lpips" in metrics:
            try:
                result.lpips = self._compute_lpips(ref_norm, proc_norm)
            except Exception as e:
                logger.warning(f"Failed to compute LPIPS: {e}")

        if "nima" in metrics:
            try:
                result.nima = self._compute_nima(proc_norm)
            except Exception as e:
                logger.warning(f"Failed to compute NIMA: {e}")

        return result

    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 1] float32 (H, W, C) format.

        Args:
            img: Input image

        Returns:
            Normalized image
        """
        img = np.asarray(img, dtype=np.float32)

        # Convert (C, H, W) to (H, W, C) if needed
        if img.ndim == 3 and img.shape[0] in (1, 3, 4):
            if img.shape[0] < img.shape[2]:  # Likely (C, H, W)
                img = np.transpose(img, (1, 2, 0))

        # Scale to [0, 1] if needed
        if img.max() > 1.0:
            img = img / 255.0

        return np.clip(img, 0.0, 1.0)

    def _compute_ssim(self, ref: np.ndarray, proc: np.ndarray) -> float:
        """Compute structural similarity index."""
        try:
            from skimage.metrics import structural_similarity

            # Convert to grayscale if needed for faster computation
            if ref.ndim == 3:
                ref_gray = np.mean(ref, axis=2)
                proc_gray = np.mean(proc, axis=2)
            else:
                ref_gray = ref
                proc_gray = proc

            ssim = structural_similarity(ref_gray, proc_gray, data_range=1.0)
            return float(ssim)

        except ImportError:
            logger.warning("scikit-image not available, using simple SSIM approximation")
            return self._compute_simple_ssim(ref, proc)

    def _compute_simple_ssim(self, ref: np.ndarray, proc: np.ndarray) -> float:
        """Simple SSIM approximation without scikit-image."""
        # Simplified SSIM using correlation
        ref_flat = ref.flatten()
        proc_flat = proc.flatten()

        mean_ref = np.mean(ref_flat)
        mean_proc = np.mean(proc_flat)

        var_ref = np.var(ref_flat)
        var_proc = np.var(proc_flat)

        covar = np.mean((ref_flat - mean_ref) * (proc_flat - mean_proc))

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        ssim = ((2 * mean_ref * mean_proc + c1) * (2 * covar + c2)) / \
               ((mean_ref**2 + mean_proc**2 + c1) * (var_ref + var_proc + c2))

        return float(ssim)

    def _compute_psnr(self, ref: np.ndarray, proc: np.ndarray) -> float:
        """Compute peak signal-to-noise ratio."""
        mse = np.mean((ref - proc) ** 2)

        if mse < 1e-10:
            return 100.0  # Identical images

        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
        return float(psnr)

    def _compute_mae(self, ref: np.ndarray, proc: np.ndarray) -> float:
        """Compute mean absolute error."""
        mae = np.mean(np.abs(ref - proc))
        return float(mae)

    def _compute_mse(self, ref: np.ndarray, proc: np.ndarray) -> float:
        """Compute mean squared error."""
        mse = np.mean((ref - proc) ** 2)
        return float(mse)

    def _compute_lpips(self, ref: np.ndarray, proc: np.ndarray) -> float:
        """
        Compute perceptual similarity using LPIPS.

        Requires lpips package: pip install lpips
        """
        if not TORCH_AVAILABLE:
            raise ImportError("LPIPS requires torch")

        import lpips

        # Load model (cached)
        if self._lpips_model is None:
            self._lpips_model = lpips.LPIPS(net='alex', verbose=False)
            if torch.cuda.is_available():
                self._lpips_model = self._lpips_model.cuda()

        # Convert to torch tensors (C, H, W) in [-1, 1]
        ref_tensor = torch.from_numpy(np.transpose(ref, (2, 0, 1))).unsqueeze(0)
        proc_tensor = torch.from_numpy(np.transpose(proc, (2, 0, 1))).unsqueeze(0)

        ref_tensor = ref_tensor * 2 - 1
        proc_tensor = proc_tensor * 2 - 1

        if torch.cuda.is_available():
            ref_tensor = ref_tensor.cuda()
            proc_tensor = proc_tensor.cuda()

        # Compute LPIPS
        with torch.inference_mode():
            distance = self._lpips_model(ref_tensor, proc_tensor)

        return float(distance.item())

    def _compute_nima(self, img: np.ndarray) -> float:
        """
        Compute neural image assessment score.

        Placeholder - would require NIMA model.
        Returns random score for now.
        """
        logger.debug("NIMA not implemented, using placeholder")
        # Would load NIMA model and compute aesthetic score
        # For now, return midpoint score
        return 5.0

    def compute_weighted_score(self, metrics: QualityMetrics) -> float:
        """
        Compute weighted quality score.

        Args:
            metrics: Quality metrics

        Returns:
            Weighted score (0-1, higher is better)
        """
        score = 0.0
        total_weight = 0.0

        metrics_dict = metrics.to_dict()

        for key, weight in self.weights.items():
            if key in metrics_dict:
                value = metrics_dict[key]

                # Normalize to 0-1 (higher is better)
                if key == "ssim":
                    normalized = value
                elif key == "psnr":
                    normalized = min(1.0, value / 50.0)  # 50 dB = perfect
                elif key == "lpips":
                    normalized = 1.0 - value  # LPIPS: lower is better
                elif key == "nima":
                    normalized = value / 10.0  # NIMA: 1-10 scale
                else:
                    normalized = value

                score += normalized * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0

        return score / total_weight
