"""Tests for YAML loading functionality (PERF-001)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from transformation_portal.depth_canonical.config import (
    DeviceType,
    IOConfig,
    ModelConfig,
    ModelVariant,
    PBRConfig,
    ProcessingConfig,
    UnifiedDepthConfig,
)


def test_from_preset_basic(temp_workspace):
    """Test loading a basic YAML preset."""
    preset_dir = temp_workspace["root"] / "config" / "presets"
    preset_dir.mkdir(parents=True)

    preset_path = preset_dir / "test_preset.yaml"
    preset_content = """
model:
  variant: depth-anything-v2-large
  device: cpu

processing:
  apply_bilateral: true
  pbr:
    enabled: true
    normal_strength: 1.5

io:
  cache_enabled: false
  output_format: tiff
"""
    preset_path.write_text(preset_content)

    # Change to temp dir so relative paths work
    orig_dir = os.getcwd()
    try:
        os.chdir(temp_workspace["root"])
        config = UnifiedDepthConfig.from_preset("test_preset")

        assert config.model.variant == ModelVariant.DA2_LARGE
        assert config.model.device == DeviceType.CPU
        assert config.processing.apply_bilateral is True
        assert config.processing.pbr.enabled is True
        assert config.processing.pbr.normal_strength == 1.5
        assert config.io.cache_enabled is False
        assert config.io.output_format == "tiff"
    finally:
        os.chdir(orig_dir)


def test_from_preset_full_path(temp_workspace):
    """Test loading preset with full path."""
    preset_path = temp_workspace["root"] / "my_preset.yaml"
    preset_content = """
model:
  variant: depth-anything-v2-base
  device: cuda

processing:
  pbr:
    enabled: false
"""
    preset_path.write_text(preset_content)

    config = UnifiedDepthConfig.from_preset(str(preset_path))

    assert config.model.variant == ModelVariant.DA2_BASE
    assert config.model.device == DeviceType.CUDA
    assert config.processing.pbr.enabled is False


def test_from_preset_not_found():
    """Test that FileNotFoundError is raised for missing preset."""
    with pytest.raises(FileNotFoundError, match="Preset file not found"):
        UnifiedDepthConfig.from_preset("nonexistent_preset")


def test_from_preset_invalid_yaml(temp_workspace):
    """Test that ValueError is raised for invalid YAML."""
    preset_path = temp_workspace["root"] / "invalid.yaml"
    preset_path.write_text("- item1\n- item2\n")  # List instead of dict

    with pytest.raises(ValueError, match="Preset must be a dictionary"):
        UnifiedDepthConfig.from_preset(str(preset_path))


def test_to_yaml_basic():
    """Test exporting config to YAML."""
    config = UnifiedDepthConfig(
        model=ModelConfig(variant=ModelVariant.DA2_LARGE, device=DeviceType.CPU),
        processing=ProcessingConfig(apply_bilateral=True, pbr=PBRConfig(enabled=True, normal_strength=1.5)),
        io=IOConfig(cache_enabled=False, output_format="tiff"),
    )

    yaml_str = config.to_yaml()

    assert "depth-anything-v2-large" in yaml_str
    assert "cpu" in yaml_str
    assert "apply_bilateral: true" in yaml_str
    assert "enabled: true" in yaml_str
    assert "normal_strength: 1.5" in yaml_str
    assert "cache_enabled: false" in yaml_str


def test_to_yaml_file(temp_workspace):
    """Test writing config to YAML file."""
    output_path = temp_workspace["root"] / "output_config.yaml"

    config = UnifiedDepthConfig(model=ModelConfig(variant=ModelVariant.DA2_BASE))

    config.to_yaml(str(output_path))

    assert output_path.exists()
    content = output_path.read_text()
    assert "depth-anything-v2-base" in content


def test_yaml_roundtrip(temp_workspace):
    """Test that config can be saved and loaded back."""
    # Create original config
    original = UnifiedDepthConfig(
        model=ModelConfig(variant=ModelVariant.DA2_LARGE, device=DeviceType.CUDA),
        processing=ProcessingConfig(apply_bilateral=True, pbr=PBRConfig(enabled=True, normal_strength=2.0)),
    )

    # Save to YAML
    yaml_path = temp_workspace["root"] / "config.yaml"
    original.to_yaml(str(yaml_path))

    # Load back
    loaded = UnifiedDepthConfig.from_preset(str(yaml_path))

    # Verify values match
    assert loaded.model.variant == original.model.variant
    assert loaded.model.device == original.model.device
    assert loaded.processing.apply_bilateral == original.processing.apply_bilateral
    assert loaded.processing.pbr.enabled == original.processing.pbr.enabled
    assert loaded.processing.pbr.normal_strength == original.processing.pbr.normal_strength
