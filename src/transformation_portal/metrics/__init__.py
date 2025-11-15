"""Quality metrics for image enhancement validation.

Provides perceptual and distribution-based metrics:

1. **LPIPS (Learned Perceptual Image Patch Similarity)**
   - Uses deep networks trained on human perceptual judgments
   - Surpasses PSNR and SSIM for perceptual quality assessment
   - Lower score = more similar (perceptually)

2. **FID (Fréchet Inception Distance)**
   - Measures distribution matching between real and generated images
   - Ensures enhanced images remain within authentic photography manifold
   - Lower score = better distribution match

3. **Traditional Metrics** (PSNR, SSIM, MS-SSIM)
   - Provided for completeness
   - Less correlated with human perception than LPIPS

For luxury real estate:
- Validate enhancements maintain photographic realism
- Ensure no distribution drift into obvious synthetic appearance
- Quantify perceptual quality improvements
"""

from transformation_portal.metrics.lpips_metric import LPIPSMetric
from transformation_portal.metrics.fid_metric import FIDMetric
from transformation_portal.metrics.traditional_metrics import TraditionalMetrics

__all__ = [
    'LPIPSMetric',
    'FIDMetric',
    'TraditionalMetrics',
]
