#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Type, configuration, and result models for the 4K rendering pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


class ToneMappingMethod(Enum):
    """Supported HDR tone mapping methods."""

    AGX = "agx"
    FILMIC = "filmic"
    REINHARD = "reinhard"
    ACES = "aces"


class QualityLevel(Enum):
    """Quality presets for processing."""

    PREVIEW = "preview"  # Fast, lower resolution
    STANDARD = "standard"  # Balanced quality/speed
    HIGH = "high"  # High quality
    ULTRA = "ultra"  # Maximum quality, 4K output


class DeviceType(Enum):
    """Compute device types."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Metal


# Processing stage names for metrics and feedback
STAGE_NAMES = [
    "input_validation",
    "depth_estimation",
    "tone_mapping",
    "material_response",
    "color_grading",
    "ai_enhancement",
    "upscaling",
    "quality_assessment",
    "output_generation",
]


@dataclass
class DepthConfig:
    """Depth estimation configuration."""

    enabled: bool = True
    model_variant: str = "small"  # small, base, large
    backend: str = "auto"  # auto, pytorch_mps, pytorch_cpu, coreml
    num_zones: int = 3  # Foreground, midground, background
    cache_enabled: bool = True
    cache_max_size: int = 50


@dataclass
class ToneMappingConfig:
    """HDR tone mapping configuration."""

    enabled: bool = True
    method: ToneMappingMethod = ToneMappingMethod.AGX
    exposure: float = 0.0
    contrast: float = 1.0
    white_point: float = 11.2
    preserve_highlights: bool = True


@dataclass
class MaterialResponseConfig:
    """Material Response Technology configuration."""

    enabled: bool = True
    strength: float = 0.7
    texture_boost: float = 0.25
    surface_types: List[str] = field(default_factory=lambda: ["wood", "metal", "glass", "stone", "fabric"])
    preserve_highlights: bool = True
    micro_contrast: float = 0.15


@dataclass
class ColorGradingConfig:
    """Color grading and LUT configuration."""

    enabled: bool = True
    lut_paths: List[str] = field(default_factory=list)
    lut_strengths: List[float] = field(default_factory=list)
    saturation: float = 1.05
    vibrance: float = 1.08
    temperature_shift: Tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass
class AIEnhancementConfig:
    """AI enhancement configuration (ControlNet guidance)."""

    enabled: bool = False  # Requires optional ML dependencies
    use_controlnet: bool = True
    use_depth_guidance: bool = True
    prompt: str = "photorealistic luxury architectural rendering, professional lighting"
    negative_prompt: str = "blurry, artifacts, cartoon, oversaturated"
    strength: float = 0.3
    guidance_scale: float = 7.5
    num_steps: int = 25
    seed: int = 42  # For reproducibility


@dataclass
class UpscalingConfig:
    """Upscaling configuration."""

    enabled: bool = True
    target_resolution: Tuple[int, int] = (3840, 2160)  # 4K UHD
    method: str = "lanczos"  # lanczos, esrgan (requires optional deps)
    scale_factor: int = 4
    preserve_sharpness: bool = True


@dataclass
class QualityFeedbackConfig:
    """RAG-based quality feedback loop configuration."""

    enabled: bool = True
    min_quality_threshold: float = 0.75
    max_iterations: int = 3
    metrics: List[str] = field(default_factory=lambda: ["sharpness", "contrast", "colorfulness", "exposure"])
    auto_adjust: bool = True
    # LPIPS integration settings
    use_lpips: bool = False  # Enable LPIPS perceptual scoring (requires torch/lpips)
    lpips_network: str = "alex"  # Network for LPIPS ('alex', 'vgg', 'squeeze')
    perceptual_percentile_target: float = 95.0  # Target percentile for perceptual quality
    material_fidelity_target: float = 0.98  # 98% material fidelity target
    # Hybrid mode settings
    hybrid_mode: bool = True  # Compute both LPIPS and heuristic metrics simultaneously
    enable_material_fidelity: bool = True  # Compute per-material fidelity scores
    # RAG indexing settings
    rag_indexing_enabled: bool = False  # Enable RAG quality metric indexing
    rag_index_path: Optional[str] = None  # Path to RAG index (if None, uses default)


@dataclass
class OutputConfig:
    """Output configuration."""

    master_tiff_16bit: bool = True
    delivery_jpeg: bool = True
    jpeg_quality: int = 95
    jpeg_progressive: bool = True
    save_intermediate: bool = False
    save_depth_visualization: bool = True
    save_quality_report: bool = True
    preserve_metadata: bool = True


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""

    name: str = "default"
    description: str = ""
    quality_level: QualityLevel = QualityLevel.HIGH
    depth: DepthConfig = field(default_factory=DepthConfig)
    tone_mapping: ToneMappingConfig = field(default_factory=ToneMappingConfig)
    material_response: MaterialResponseConfig = field(default_factory=MaterialResponseConfig)
    color_grading: ColorGradingConfig = field(default_factory=ColorGradingConfig)
    ai_enhancement: AIEnhancementConfig = field(default_factory=AIEnhancementConfig)
    upscaling: UpscalingConfig = field(default_factory=UpscalingConfig)
    quality_feedback: QualityFeedbackConfig = field(default_factory=QualityFeedbackConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


@dataclass
class StageMetrics:
    """Metrics for a single processing stage."""

    name: str
    duration_ms: float
    success: bool
    quality_delta: float = 0.0
    notes: str = ""


@dataclass
class QualityMetrics:
    """Image quality assessment metrics."""

    sharpness: float = 0.0  # 0-1
    contrast: float = 0.0  # 0-1
    colorfulness: float = 0.0  # 0-1
    exposure_balance: float = 0.0  # 0-1
    noise_level: float = 0.0  # 0-1 (lower is better)
    overall_score: float = 0.0  # 0-1
    # LPIPS perceptual metrics (when available)
    lpips_score: float = 0.0  # 0-1 (lower is better, 0 = identical)
    lpips_percentile: float = 0.0  # Percentile rank against benchmark
    material_fidelity: float = 0.0  # 0-1 (higher is better)
    perceptual_quality: float = 0.0  # Composite perceptual score (0-100)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ProcessingResult:
    """Complete processing result with image and metadata."""

    image: Image.Image
    depth_map: Optional[np.ndarray] = None
    quality_metrics: Optional[QualityMetrics] = None
    stage_metrics: List[StageMetrics] = field(default_factory=list)
    total_duration_ms: float = 0.0
    iterations: int = 1
    output_paths: Dict[str, Path] = field(default_factory=dict)
    config_used: Optional[PipelineConfig] = None

    @property
    def quality_score(self) -> float:
        """Get overall quality score."""
        if self.quality_metrics:
            return self.quality_metrics.overall_score
        return 0.0


__all__ = [
    "AIEnhancementConfig",
    "ColorGradingConfig",
    "DepthConfig",
    "DeviceType",
    "MaterialResponseConfig",
    "OutputConfig",
    "PipelineConfig",
    "ProcessingResult",
    "QualityFeedbackConfig",
    "QualityLevel",
    "QualityMetrics",
    "STAGE_NAMES",
    "StageMetrics",
    "ToneMappingConfig",
    "ToneMappingMethod",
    "UpscalingConfig",
]
