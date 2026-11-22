"""
Image Quality Metrics.

Computes objective quality scores (PSNR, SSIM) to validate enhancement results.
"""

import math
from dataclasses import dataclass
from typing import Dict, Union

import numpy as np

try:
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim

    SKIMAGE_AVAIL = True
except ImportError:
    SKIMAGE_AVAIL = False


@dataclass
class QualityMetrics:
    psnr: float
    ssim: float
    mse: float


class MetricsComputer:
    """Calculates image quality metrics against a reference."""

    @staticmethod
    def compute(prediction: np.ndarray, target: np.ndarray) -> QualityMetrics:
        """
        Compare prediction against target (ground truth or original).

        Args:
            prediction: Image array (H, W, C), 0-255 uint8 or 0-1 float.
            target: Reference image array (must match shape/type).
        """
        # Ensure consistency
        if prediction.shape != target.shape:
            # Resize prediction to match target if needed (simple center crop or resize)
            # For strict metrics, shapes must match. returning zeros.
            return QualityMetrics(0.0, 0.0, 0.0)

        # Cast before arithmetic so uint8 differences and squares cannot wrap.
        mse = float(np.mean((target.astype(float) - prediction.astype(float)) ** 2))

        # Handle data range: the full span of an integer type, 0-1 for floats.
        # Assuming 1.0 for every non-uint8 input rated 16-bit images absurdly low.
        if np.issubdtype(prediction.dtype, np.integer):
            dtype_info = np.iinfo(prediction.dtype)
            data_range = float(dtype_info.max) - float(dtype_info.min)
        else:
            data_range = 1.0

        # Use skimage if available (Robust)
        if SKIMAGE_AVAIL:
            # Multichannel must be explicitly set for SSIM in new skimage versions
            channel_axis = -1 if prediction.ndim == 3 else None

            p_val = psnr(target, prediction, data_range=data_range)
            s_val = ssim(target, prediction, data_range=data_range, channel_axis=channel_axis)
            return QualityMetrics(p_val, s_val, mse)

        # Fallback (Manual Calculation)
        if mse == 0:
            return QualityMetrics(100.0, 1.0, 0.0)

        psnr_val = 20 * math.log10(data_range / math.sqrt(mse))

        return QualityMetrics(psnr_val, 0.0, mse)  # SSIM hard to implement manually efficiently
