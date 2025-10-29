"""Luxury TIFF batch processing toolkit.

The package is organised into focused modules:

- :mod:`luxury_tiff_batch_processor.adjustments` encapsulates the color science
  primitives, presets, and mathematical helpers for manipulating image arrays.
- :mod:`luxury_tiff_batch_processor.io_utils` hosts capability detection,
  floating-point conversion helpers, and low-level TIFF I/O routines.
- :mod:`luxury_tiff_batch_processor.cli` wires these pieces together for the
  command-line interface while remaining lightweight for library consumers.
- :mod:`luxury_tiff_batch_processor.pipeline` provides orchestration helpers
  used by both the CLI and programmatic integrations.
"""
from __future__ import annotations

import logging

from .adjustments import (
    AdjustmentSettings,
    LUXURY_PRESETS,
    apply_adjustments,
    gaussian_blur,
    gaussian_kernel,
    gaussian_kernel_cached,
)
from .cli import build_adjustments, main, parse_args, run_pipeline, default_output_folder
from .io_utils import (
    FloatDynamicRange,
    ImageToFloatResult,
    LuxuryGradeException,
    ProcessingCapabilities,
    ProcessingContext,
    float_to_dtype_array,
    image_to_float,
    save_image,
)
from .profiles import DEFAULT_PROFILE_NAME, PROCESSING_PROFILES, ProcessingProfile
from .pipeline import (
    collect_images,
    process_image,
    process_single_image,
)

LOGGER = logging.getLogger("luxury_tiff_batch_processor")

__all__ = [
    "AdjustmentSettings",
    "FloatDynamicRange",
    "ImageToFloatResult",
    "LUXURY_PRESETS",
    "LuxuryGradeException",
    "ProcessingCapabilities",
    "ProcessingContext",
    "ProcessingProfile",
    "apply_adjustments",
    "build_adjustments",
    "collect_images",
    "default_output_folder",
    "float_to_dtype_array",
    "gaussian_blur",
    "gaussian_kernel",
    "gaussian_kernel_cached",
    "image_to_float",
    "main",
    "parse_args",
    "process_image",
    "process_single_image",
    "PROCESSING_PROFILES",
    "DEFAULT_PROFILE_NAME",
    "run_pipeline",
    "save_image",
]

__all__.extend(["_PROGRESS_WRAPPER", "_wrap_with_progress", "ensure_output_path"])
