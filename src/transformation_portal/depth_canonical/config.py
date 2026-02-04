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
                        or full path to YAML file

        Returns:
            UnifiedDepthConfig loaded from preset

        Raises:
            FileNotFoundError: If preset file not found
            ValueError: If YAML is invalid or missing required fields

        Example:
            >>> config = UnifiedDepthConfig.from_preset("depth_pro_example")
            >>> config = UnifiedDepthConfig.from_preset("config/presets/my_preset.yaml")
        """
        from pathlib import Path
        import yaml
        
        # Determine preset path
        if preset_name.endswith(('.yaml', '.yml')):
            preset_path = Path(preset_name)
        else:
            # Look in config/presets/ directory
            preset_path = Path(f"config/presets/{preset_name}.yaml")
            if not preset_path.exists():
                # Try without config/ prefix (in case called from config dir)
                preset_path = Path(f"presets/{preset_name}.yaml")
        
        if not preset_path.exists():
            raise FileNotFoundError(
                f"Preset file not found: {preset_path}\n"
                f"Looked in: config/presets/{preset_name}.yaml"
            )
        
        # Load YAML
        with open(preset_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if not isinstance(data, dict):
            raise ValueError(f"Preset must be a dictionary, got {type(data).__name__}")
        
        # Parse configuration sections
        model_data = data.get('model', {})
        processing_data = data.get('processing', {})
        io_data = data.get('io', {})
        security_data = data.get('security', {})
        
        # Parse nested PBR config if present
        pbr_data = processing_data.pop('pbr', {}) if 'pbr' in processing_data else {}
        
        # Build config objects
        model_config = ModelConfig(**model_data) if model_data else ModelConfig()
        
        pbr_config = PBRConfig(**pbr_data) if pbr_data else PBRConfig()
        processing_config = ProcessingConfig(pbr=pbr_config, **processing_data) if processing_data else ProcessingConfig()
        
        io_config = IOConfig(**io_data) if io_data else IOConfig()
        security_config = SecurityConfig(**security_data) if security_data else SecurityConfig()
        
        return cls(
            model=model_config,
            processing=processing_config,
            io=io_config,
            security=security_config
        )

    def to_yaml(self, output_path: str = None) -> str:
        """Export configuration to YAML format.

        Args:
            output_path: Optional path to write YAML file. If None, returns YAML string.

        Returns:
            YAML string representation of configuration

        Example:
            >>> config = UnifiedDepthConfig.from_preset("depth_pro_example")
            >>> yaml_str = config.to_yaml()
            >>> config.to_yaml("my_config.yaml")  # Write to file
        """
        import yaml
        from pathlib import Path
        from dataclasses import asdict
        
        # Convert to dictionary
        config_dict = {
            'model': asdict(self.model),
            'processing': {
                **{k: v for k, v in asdict(self.processing).items() if k != 'pbr'},
                'pbr': asdict(self.processing.pbr)
            },
            'io': asdict(self.io),
            'security': asdict(self.security)
        }
        
        # Convert enums to strings
        def _convert_enums(obj):
            if isinstance(obj, dict):
                return {k: _convert_enums(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_convert_enums(item) for item in obj]
            elif isinstance(obj, Enum):
                return obj.value
            return obj
        
        config_dict = _convert_enums(config_dict)
        
        # Generate YAML
        yaml_str = yaml.safe_dump(config_dict, default_flow_style=False, sort_keys=False)
        
        # Write to file if path provided
        if output_path:
            Path(output_path).write_text(yaml_str, encoding='utf-8')
        
        return yaml_str
