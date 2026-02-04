"""Tests for YAML loading functionality (PERF-001)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from transformation_portal.depth_canonical.config import (
    UnifiedDepthConfig,
    ModelConfig,
    ProcessingConfig,
    PBRConfig,
    IOConfig,
    ModelVariant,
    DeviceType,
)


def test_from_preset_basic():
    """Test loading a basic YAML preset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        preset_dir = Path(tmpdir) / 'config' / 'presets'
        preset_dir.mkdir(parents=True)

        preset_path = preset_dir / 'test_preset.yaml'
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

        # Change to tmpdir so relative paths work
        import os
        orig_dir = os.getcwd()
        try:
            os.chdir(tmpdir)
            config = UnifiedDepthConfig.from_preset('test_preset')

            assert config.model.variant == ModelVariant.DA2_LARGE
            assert config.model.device == DeviceType.CPU
            assert config.processing.apply_bilateral is True
            assert config.processing.pbr.enabled is True
            assert config.processing.pbr.normal_strength == 1.5
            assert config.io.cache_enabled is False
            assert config.io.output_format == 'tiff'
        finally:
            os.chdir(orig_dir)


def test_from_preset_full_path():
    """Test loading preset with full path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        preset_path = Path(tmpdir) / 'my_preset.yaml'
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
        UnifiedDepthConfig.from_preset('nonexistent_preset')


def test_from_preset_invalid_yaml():
    """Test that ValueError is raised for invalid YAML."""
    with tempfile.TemporaryDirectory() as tmpdir:
        preset_path = Path(tmpdir) / 'invalid.yaml'
        preset_path.write_text('- item1\n- item2\n')  # List instead of dict

        with pytest.raises(ValueError, match="Preset must be a dictionary"):
            UnifiedDepthConfig.from_preset(str(preset_path))


def test_to_yaml_basic():
    """Test exporting config to YAML."""
    config = UnifiedDepthConfig(
        model=ModelConfig(variant=ModelVariant.DA2_LARGE, device=DeviceType.CPU),
        processing=ProcessingConfig(
            apply_bilateral=True,
            pbr=PBRConfig(enabled=True, normal_strength=1.5)
        ),
        io=IOConfig(cache_enabled=False, output_format='tiff')
    )

    yaml_str = config.to_yaml()

    assert 'depth-anything-v2-large' in yaml_str
    assert 'cpu' in yaml_str
    assert 'apply_bilateral: true' in yaml_str
    assert 'enabled: true' in yaml_str
    assert 'normal_strength: 1.5' in yaml_str
    assert 'cache_enabled: false' in yaml_str


def test_to_yaml_file():
    """Test writing config to YAML file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / 'output_config.yaml'

        config = UnifiedDepthConfig(
            model=ModelConfig(variant=ModelVariant.DA2_BASE)
        )

        config.to_yaml(str(output_path))

        assert output_path.exists()
        content = output_path.read_text()
        assert 'depth-anything-v2-base' in content


def test_yaml_roundtrip():
    """Test that config can be saved and loaded back."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create original config
        original = UnifiedDepthConfig(
            model=ModelConfig(variant=ModelVariant.DA2_LARGE, device=DeviceType.CUDA),
            processing=ProcessingConfig(
                apply_bilateral=True,
                pbr=PBRConfig(enabled=True, normal_strength=2.0)
            )
        )

        # Save to YAML
        yaml_path = Path(tmpdir) / 'config.yaml'
        original.to_yaml(str(yaml_path))

        # Load back
        loaded = UnifiedDepthConfig.from_preset(str(yaml_path))

        # Verify values match
        assert loaded.model.variant == original.model.variant
        assert loaded.model.device == original.model.device
        assert loaded.processing.apply_bilateral == original.processing.apply_bilateral
        assert loaded.processing.pbr.enabled == original.processing.pbr.enabled
        assert loaded.processing.pbr.normal_strength == original.processing.pbr.normal_strength
