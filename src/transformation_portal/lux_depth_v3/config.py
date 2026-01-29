"""Configuration module for lux_depth_v3 pipeline.

STUB IMPLEMENTATION - Critical types to enable package imports.
Full implementation pending.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any
from .security import HashMode


class ModelVariant(Enum):
    """Depth Anything V3 model variants."""
    METRIC_LARGE = type('ModelVariantValue', (), {
        'name': 'depth-anything-v3-metric-large',
        'display_name': 'Depth Anything V3 Metric Large',
        'huggingface_id': 'depth-anything/Depth-Anything-V3-Metric-Large-hf',
    })()
    METRIC_BASE = type('ModelVariantValue', (), {
        'name': 'depth-anything-v3-metric-base',
        'display_name': 'Depth Anything V3 Metric Base',
        'huggingface_id': 'depth-anything/Depth-Anything-V3-Metric-Base-hf',
    })()
    METRIC_SMALL = type('ModelVariantValue', (), {
        'name': 'depth-anything-v3-metric-small',
        'display_name': 'Depth Anything V3 Metric Small',
        'huggingface_id': 'depth-anything/Depth-Anything-V3-Metric-Small-hf',
    })()


class Preset(Enum):
    """Pipeline presets for different use cases."""
    ARCHITECTURAL_INTERIOR = "architectural_interior"
    ARCHITECTURAL_EXTERIOR = "architectural_exterior"
    LUXURY_ESTATE = "luxury_estate"
    DEFAULT = "default"


@dataclass
class DeviceConfig:
    """Device configuration for inference."""
    device: str = "cpu"
    dtype: str = "float32"


@dataclass
class PostprocessingConfig:
    """Postprocessing configuration for depth maps."""
    apply_metric_scaling: bool = True
    scale_factor: float = 1.0
    apply_median_filter: bool = False
    median_kernel_size: int = 3
    apply_bilateral_filter: bool = False
    bilateral_sigma_color: float = 0.0
    bilateral_sigma_space: float = 0.0
    preserve_edges: bool = True
    edge_threshold: float = 0.1
    fusion_mode: str = "weighted"
    refinement: Optional[Any] = None


@dataclass
class DA3Config:
    """Depth Anything V3 configuration."""
    model_variant: ModelVariant = ModelVariant.METRIC_LARGE
    device: DeviceConfig = field(default_factory=DeviceConfig)
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)

    @classmethod
    def from_preset(cls, preset: Preset) -> DA3Config:
        """Create configuration from preset.

        STUB: Returns default configuration.
        """
        return cls()


@dataclass
class EnhanceConfig:
    """Configuration for the enhancement orchestrator."""
    # Depth configuration
    model_variant: Optional[ModelVariant] = None
    preset: Optional[Preset] = None
    depth_device: str = "cpu"
    depth_quantization: str = "none"

    # V2 configuration
    v2_preset: str = "default"
    v2_device: str = "cpu"
    v2_upscaler_backend: str = "default"

    # Flags
    force_depth: bool = False
    force_v2: bool = False
    non_commercial_ok: bool = False
    verify_depth_writes: bool = True

    # Fallback configuration
    depth_fallback: str = "fail"  # Options: "fail", "skip", "v2-auto"
    v2_timeout: int = 300

    # Hash mode
    hash_mode: HashMode = HashMode.IF_MANIFEST_EXISTS
