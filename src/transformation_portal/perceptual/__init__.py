"""Phase 2 perceptual baseline calibration public exports.

Several perceptual modules depend on optional ML packages such as torch.  Keep
the package import itself light so pure-Python submodules can run in core CI.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .synthetic_viewer import ACUScore, JourneyMoment, SyntheticViewer

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from .analyzer import AnalysisResult, PerceptualAnalyzer
    from .baseline import BaselineConfig, PerceptualBaseline
    from .image_loader import ImageLoader, ImageMetadata
    from .metrics import (
        MetricType,
        PerceptualScore,
        QualityMetrics,
        compute_brisque,
        compute_fid,
        compute_lpips,
        compute_niqe,
        compute_psnr,
        compute_ssim,
    )
    from .tracker import EnhancementTracker, TrajectoryPoint

_LAZY_EXPORTS = {
    "AnalysisResult": ".analyzer",
    "PerceptualAnalyzer": ".analyzer",
    "BaselineConfig": ".baseline",
    "PerceptualBaseline": ".baseline",
    "ImageLoader": ".image_loader",
    "ImageMetadata": ".image_loader",
    "MetricType": ".metrics",
    "PerceptualScore": ".metrics",
    "QualityMetrics": ".metrics",
    "compute_brisque": ".metrics",
    "compute_fid": ".metrics",
    "compute_lpips": ".metrics",
    "compute_niqe": ".metrics",
    "compute_psnr": ".metrics",
    "compute_ssim": ".metrics",
    "EnhancementTracker": ".tracker",
    "TrajectoryPoint": ".tracker",
}


def __getattr__(name: str) -> Any:
    """Lazily resolve optional perceptual exports."""

    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EXPORTS})


__all__ = [
    "PerceptualBaseline",
    "BaselineConfig",
    "QualityMetrics",
    "MetricType",
    "PerceptualScore",
    "ImageLoader",
    "ImageMetadata",
    "PerceptualAnalyzer",
    "AnalysisResult",
    "SyntheticViewer",
    "ACUScore",
    "JourneyMoment",
    "EnhancementTracker",
    "TrajectoryPoint",
    "compute_lpips",
    "compute_fid",
    "compute_brisque",
    "compute_niqe",
    "compute_psnr",
    "compute_ssim",
]

__version__ = "1.0.0"
