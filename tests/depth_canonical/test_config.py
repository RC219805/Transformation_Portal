"""Tests for depth_canonical configuration module."""

import pytest
from transformation_portal.depth_canonical.config import (
    UnifiedDepthConfig,
    ModelConfig,
    ProcessingConfig,
    PBRConfig,
    IOConfig,
    SecurityConfig,
    DeviceType,
    ModelVariant,
)


def test_device_type_enum_values():
    """Test DeviceType enum has expected values."""
    assert DeviceType.CPU == "cpu"
    assert DeviceType.CUDA == "cuda"
    assert DeviceType.MPS == "mps"
    assert DeviceType.COREML == "coreml"


def test_model_variant_enum_has_da2_and_da3():
    """Test ModelVariant enum includes both DA2 and DA3 variants."""
    variants = set(ModelVariant)

    # Check DA3 variants
    assert ModelVariant.DA3_LARGE in variants
    assert ModelVariant.DA3_BASE in variants
    assert ModelVariant.DA3_SMALL in variants

    # Check DA2 variants
    assert ModelVariant.DA2_LARGE in variants
    assert ModelVariant.DA2_BASE in variants
    assert ModelVariant.DA2_SMALL in variants


def test_pbr_config_is_frozen():
    """Test PBRConfig is immutable (frozen dataclass)."""
    config = PBRConfig(normal_strength=2.0)

    with pytest.raises(Exception):  # FrozenInstanceError
        config.normal_strength = 1.0


def test_pbr_config_defaults():
    """Test PBRConfig has sensible defaults."""
    config = PBRConfig()

    assert config.enabled is False
    assert config.normal_strength == 1.0
    assert config.normal_blur_radius == 0
    assert config.roughness_strength == 1.0
    assert config.roughness_blur_radius == 3
    assert config.ao_strength == 1.0
    assert config.ao_blur_radius == 5
    assert config.ao_bias == 0.5


def test_pbr_config_custom_values():
    """Test PBRConfig accepts custom values."""
    config = PBRConfig(
        enabled=True,
        normal_strength=1.5,
        normal_blur_radius=2,
        roughness_strength=0.8,
        roughness_blur_radius=5,
        ao_strength=1.2,
        ao_blur_radius=7,
        ao_bias=0.6,
    )

    assert config.enabled is True
    assert config.normal_strength == 1.5
    assert config.normal_blur_radius == 2
    assert config.roughness_strength == 0.8
    assert config.roughness_blur_radius == 5
    assert config.ao_strength == 1.2
    assert config.ao_blur_radius == 7
    assert config.ao_bias == 0.6


def test_model_config_defaults():
    """Test ModelConfig has sensible defaults."""
    config = ModelConfig()

    assert config.variant == ModelVariant.DA3_SMALL
    assert config.device == DeviceType.CPU
    assert config.dtype == "float32"


def test_processing_config_defaults():
    """Test ProcessingConfig has sensible defaults."""
    config = ProcessingConfig()

    # Postprocessing defaults
    assert config.apply_bilateral is True
    assert config.bilateral_sigma_color == 10.0
    assert config.bilateral_sigma_space == 10.0

    # Zone mapping defaults
    assert config.enable_zone_mapping is False
    assert config.num_zones == 3
    assert config.tone_map_method == "agx"

    # Atmospheric defaults
    assert config.enable_atmospheric is False
    assert config.haze_strength == 0.0

    # Denoising defaults
    assert config.enable_denoise is False
    assert config.denoise_strength == 0.5

    # PBR defaults
    assert isinstance(config.pbr, PBRConfig)
    assert config.pbr.enabled is False


def test_io_config_defaults():
    """Test IOConfig has sensible defaults."""
    config = IOConfig()

    assert config.cache_enabled is True
    assert config.cache_size == 128
    assert config.output_format == "png"
    assert config.depth_bit_depth == 16


def test_security_config_defaults():
    """Test SecurityConfig has sensible defaults."""
    config = SecurityConfig()

    assert config.validate_paths is True
    assert config.max_image_size == 8192
    assert config.allowed_extensions == (".jpg", ".jpeg", ".png", ".tiff", ".tif")


def test_unified_depth_config_defaults():
    """Test UnifiedDepthConfig creates with default sub-configs."""
    config = UnifiedDepthConfig()

    assert isinstance(config.model, ModelConfig)
    assert isinstance(config.processing, ProcessingConfig)
    assert isinstance(config.io, IOConfig)
    assert isinstance(config.security, SecurityConfig)


def test_unified_depth_config_custom_subconfigs():
    """Test UnifiedDepthConfig accepts custom sub-configurations."""
    model_config = ModelConfig(
        variant=ModelVariant.DA3_BASE,
        device=DeviceType.CUDA
    )
    processing_config = ProcessingConfig(
        pbr=PBRConfig(enabled=True, normal_strength=1.5)
    )
    io_config = IOConfig(cache_size=256)
    security_config = SecurityConfig(max_image_size=4096)

    config = UnifiedDepthConfig(
        model=model_config,
        processing=processing_config,
        io=io_config,
        security=security_config,
    )

    assert config.model.variant == ModelVariant.DA3_BASE
    assert config.model.device == DeviceType.CUDA
    assert config.processing.pbr.enabled is True
    assert config.processing.pbr.normal_strength == 1.5
    assert config.io.cache_size == 256
    assert config.security.max_image_size == 4096


def test_unified_depth_config_from_preset_stub():
    """Test UnifiedDepthConfig.from_preset returns default (stub implementation)."""
    config = UnifiedDepthConfig.from_preset("architectural_interior")

    # Stub implementation returns default
    assert isinstance(config, UnifiedDepthConfig)
    assert isinstance(config.model, ModelConfig)
    assert isinstance(config.processing, ProcessingConfig)


def test_pbr_config_enabled_flag():
    """Test PBRConfig enabled flag controls PBR generation."""
    # Disabled by default
    config_disabled = PBRConfig()
    assert config_disabled.enabled is False

    # Can be explicitly enabled
    config_enabled = PBRConfig(enabled=True)
    assert config_enabled.enabled is True
