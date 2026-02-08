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

4. **Performance Metrics** (PerformanceCapsule, ledger)
   - Scene-dependent performance tracking
   - Regression detection with bucketing
   - Zero-overhead timing instrumentation

For luxury real estate:
- Validate enhancements maintain photographic realism
- Ensure no distribution drift into obvious synthetic appearance
- Quantify perceptual quality improvements
- Track and enforce performance characteristics
"""

from transformation_portal.metrics.ledger import PerformanceLedger

# Performance metrics (no heavy dependencies)
from transformation_portal.metrics.performance_capsule import (
    DEFAULT_BUCKETS,
    PerformanceBucket,
    PerformanceCapsule,
    compute_config_hash,
    compute_dimension_adjustment,
    compute_specificity,
    get_bucket_for_capsule,
)
from transformation_portal.metrics.timing import TimingContext, compute_overhead, merge_timings, timing_context

__all__ = [
    "PerformanceCapsule",
    "PerformanceBucket",
    "PerformanceLedger",
    "TimingContext",
    "timing_context",
    "compute_config_hash",
    "compute_dimension_adjustment",
    "compute_specificity",
    "get_bucket_for_capsule",
    "compute_overhead",
    "merge_timings",
    "DEFAULT_BUCKETS",
]

# Optional ML-based metrics (lazy import to avoid dependency issues)
try:
    from transformation_portal.metrics.fid_metric import FIDMetric  # noqa: F401

    __all__.append("FIDMetric")
except ImportError:
    # Optional dependency: FIDMetric unavailable when torch-fidelity not installed
    pass

try:
    from transformation_portal.metrics.lpips_metric import LPIPSMetric  # noqa: F401

    __all__.append("LPIPSMetric")
except ImportError:
    # Optional dependency: LPIPSMetric unavailable when lpips package not installed
    pass

try:
    from transformation_portal.metrics.traditional_metrics import TraditionalMetrics  # noqa: F401

    __all__.append("TraditionalMetrics")
except ImportError:
    # Optional dependency: TraditionalMetrics unavailable when skimage not installed
    pass
