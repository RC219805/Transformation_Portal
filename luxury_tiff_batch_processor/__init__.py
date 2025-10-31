"""Luxury TIFF batch processing toolkit for high-fidelity image enhancement.

This package provides professional-grade image processing for luxury real estate,
architectural visualization, and editorial post-production. It supports 16-bit TIFF
workflows with comprehensive color science and Material Response technology.

Module Organization
-------------------

adjustments
    Color science primitives, presets, and mathematical image transformations.
    Includes exposure, white balance, tone curves, clarity, and specialty effects.

io_utils
    High-fidelity I/O operations with 16-bit TIFF support, capability detection,
    and metadata preservation for professional workflows.

cli
    Command-line interface providing preset-driven batch processing with
    extensive customization options.

pipeline
    Processing orchestration for single images and batch operations with
    progress tracking and parallel execution support.

profiles
    Processing profiles balancing quality and performance for different use cases.

Key Features
------------

- 16-bit TIFF processing with Material Response surface enhancement
- Preset-driven workflows (Signature, Architectural, Golden Hour, Twilight)
- Intelligent adjustments: vibrance, clarity, glow, chroma denoising
- Batch processing with recursive directory support
- HDR-capable with metadata preservation
- Parallel processing for high-throughput operations

Example Usage
-------------

    from luxury_tiff_batch_processor import (
        AdjustmentSettings,
        LUXURY_PRESETS,
        process_single_image,
    )

    # Use a preset for quick processing
    settings = LUXURY_PRESETS["signature"]
    process_single_image(
        source=Path("input.tif"),
        destination=Path("output.tif"),
        adjustments=settings,
        compression="tiff_lzw",
    )

    # Or create custom adjustments
    custom = AdjustmentSettings(
        exposure=0.15,
        clarity=0.20,
        vibrance=0.25,
    )
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
    ensure_output_path,
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
    "ensure_output_path",
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
