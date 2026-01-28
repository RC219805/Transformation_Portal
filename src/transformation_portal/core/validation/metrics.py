"""
Image Quality Metrics.

Computes objective quality scores (PSNR, SSIM) to validate enhancement results.
"""

import numpy as np
import math
from dataclasses import dataclass
from typing import Union, Dict

try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
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
    def compute(
        prediction: np.ndarray, 
        target: np.ndarray
    ) -> QualityMetrics:
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

        # Use skimage if available (Robust)
        if SKIMAGE_AVAIL:
            # Handle data range
            data_range = 255 if prediction.dtype == np.uint8 else 1.0
            
            # Multichannel must be explicitly set for SSIM in new skimage versions
            channel_axis = -1 if prediction.ndim == 3 else None
            
            p_val = psnr(target, prediction, data_range=data_range)
            s_val = ssim(target, prediction, data_range=data_range, channel_axis=channel_axis)
            mse_val = np.mean((target - prediction) ** 2)
            
            return QualityMetrics(p_val, s_val, mse_val)
            
        # Fallback (Manual Calculation)
        mse = np.mean((target.astype(float) - prediction.astype(float)) ** 2)
        if mse == 0:
            return QualityMetrics(100.0, 1.0, 0.0)
            
        max_pixel = 255.0 if prediction.dtype == np.uint8 else 1.0
        psnr_val = 20 * math.log10(max_pixel / math.sqrt(mse))
        
        return QualityMetrics(psnr_val, 0.0, mse) # SSIM hard to implement manually efficiently
