"""Unified configuration for canonical depth processing.

This module provides the single source of truth for all depth-related configuration.
"""

from dataclasses import dataclass, field
from enum import Enum


class DeviceType(str, Enum):
    """Canonical device enumeration for depth processing."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    COREML = "coreml"


class ModelVariant(Enum):
    """Supported depth estimation models."""
    # Depth Anything V2 models (via HuggingFace transformers)
    DA2_LARGE = "depth-anything-v2-large"
    DA2_BASE = "depth-anything-v2-base"
    DA2_SMALL = "depth-anything-v2-small"

    # DA3 aliases for V2 models (V3 uses same V2 models)
    DA3_LARGE = "depth-anything-v2-large"
    DA3_BASE = "depth-anything-v2-base"
    DA3_SMALL = "depth-anything-v2-small"

    # Depth Pro (metric depth)
    DEPTH_PRO = "depth-pro"


@dataclass(frozen=True)
class PBRConfig:
    """PBR map generation configuration (immutable for cache-ability).

    Args:
        enabled: Enable PBR map generation
        normal_strength: Gradient multiplier for normal maps (higher = more pronounced)
        normal_blur_radius: Pre-blur depth before gradient computation (0 = disabled)
        roughness_strength: Detail multiplier for roughness maps
        roughness_blur_radius: Smoothing kernel size for roughness
        ao_strength: Darkness multiplier for ambient occlusion
        ao_blur_radius: Occlusion spread radius
        ao_bias: Brightness offset (0.0-1.0) - higher values prevent dark occlusion
    """
    enabled: bool = False
    normal_strength: float = 1.0
    normal_blur_radius: int = 0
    roughness_strength: float = 1.0
    roughness_blur_radius: int = 3
    ao_strength: float = 1.0
    ao_blur_radius: int = 5
    ao_bias: float = 0.5


@dataclass
class ModelConfig:
    """Model selection and device configuration."""
    variant: ModelVariant = ModelVariant.DA3_SMALL
    device: DeviceType = DeviceType.CPU
    dtype: str = "float32"


@dataclass
class ProcessingConfig:
    """Depth processing and enhancement configuration."""
    # Postprocessing
    apply_bilateral: bool = True
    bilateral_sigma_color: float = 10.0
    bilateral_sigma_space: float = 10.0

    # Zone mapping
    enable_zone_mapping: bool = False
    num_zones: int = 3
    tone_map_method: str = "agx"

    # Atmospheric effects
    enable_atmospheric: bool = False
    haze_strength: float = 0.0

    # Denoising
    enable_denoise: bool = False
    denoise_strength: float = 0.5

    # PBR map generation
    pbr: PBRConfig = field(default_factory=PBRConfig)


@dataclass
class IOConfig:
    """I/O and caching configuration."""
    cache_enabled: bool = True
    cache_size: int = 128
    output_format: str = "png"  # png, tiff
    depth_bit_depth: int = 16  # 8 or 16 bit output


@dataclass
class SecurityConfig:
    """Security and validation configuration."""
    validate_paths: bool = True
    max_image_size: int = 8192  # Maximum dimension in pixels
    allowed_extensions: tuple = (".jpg", ".jpeg", ".png", ".tiff", ".tif")


@dataclass
class UnifiedDepthConfig:
    """Unified depth estimation and processing configuration.

    This is the single source of truth for all depth pipeline configuration.

    Example:
        >>> config = UnifiedDepthConfig(
        ...     model=ModelConfig(variant=ModelVariant.DA3_LARGE),
        ...     processing=ProcessingConfig(
        ...         pbr=PBRConfig(enabled=True, normal_strength=1.2)
        ...     )
        ... )
        >>> pipeline = DepthPipeline(config)
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    io: IOConfig = field(default_factory=IOConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)

    @classmethod
    def from_preset(cls, preset_name: str) -> "UnifiedDepthConfig":
        """Load configuration from YAML preset.

        Args:
            preset_name: Name of preset file (without .yaml extension)

        Returns:
            UnifiedDepthConfig loaded from preset

        Note:
            This is a stub implementation. Full YAML loading will be
            implemented in Phase 2.
        """
        # Stub: Return default for now
        # TODO: Implement YAML loading in Phase 2
        return cls()
