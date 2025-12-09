"""Test Platform Core integration in lux_depth_v2."""

import pytest
from pathlib import Path

from lux_depth_v2.config import PipelineConfig, Preset, CORE_CONFIG_AVAILABLE
from lux_depth_v2 import torch_ops


@pytest.mark.skipif(not CORE_CONFIG_AVAILABLE, reason="Platform Core not available")
class TestCoreConfigIntegration:
    """Test integration with Platform Core config module."""
    
    def test_device_config_generation(self):
        """Test DeviceConfig can be generated from legacy fields."""
        cfg = PipelineConfig(
            device="cuda",
            precision="fp16",
            cudnn_benchmark=True,
            warn_float_gb=6.0
        )
        
        device_config = cfg.get_device_config()
        assert device_config is not None
        assert device_config.device.value == "cuda"
        assert device_config.precision.value == "fp16"
        assert device_config.enable_cudnn_benchmark is True
        assert 0.1 <= device_config.memory_fraction <= 0.95
    
    def test_device_config_auto_mode(self):
        """Test DeviceConfig with auto device selection."""
        cfg = PipelineConfig(device="auto", precision="fp32")
        
        device_config = cfg.get_device_config()
        assert device_config is not None
        assert device_config.device.value == "auto"
        assert device_config.precision.value == "fp32"
    
    def test_device_config_mps_mode(self):
        """Test DeviceConfig with MPS device."""
        cfg = PipelineConfig(device="mps", precision="fp16")
        
        device_config = cfg.get_device_config()
        assert device_config is not None
        assert device_config.device.value == "mps"
        assert device_config.prefer_neural_engine is True
    
    def test_paths_config_generation(self):
        """Test PathsConfig can be generated from legacy fields."""
        cfg = PipelineConfig(
            input_dir=Path("/tmp/input"),
            output_dir=Path("/tmp/output")
        )
        
        paths_config = cfg.get_paths_config()
        assert paths_config is not None
        # Note: Paths are expanded/resolved, so /tmp may become /private/tmp on macOS
        assert paths_config.input_dir.name == "input"
        assert paths_config.output_dir.name == "output"
        assert paths_config.cache_dir == Path(".cache").resolve()
        assert paths_config.checkpoint_dir == Path(".checkpoints").resolve()
    
    def test_paths_config_with_orchestrator(self):
        """Test PathsConfig uses orchestrator checkpoint_dir."""
        cfg = PipelineConfig()
        cfg.orchestrator.checkpoint_dir = "/custom/checkpoints"
        
        paths_config = cfg.get_paths_config()
        assert paths_config is not None
        assert paths_config.checkpoint_dir == Path("/custom/checkpoints")
    
    def test_paths_config_none_paths(self):
        """Test PathsConfig handles None paths gracefully."""
        cfg = PipelineConfig(input_dir=None, output_dir=None)
        
        paths_config = cfg.get_paths_config()
        assert paths_config is not None
        assert paths_config.input_dir is None
        assert paths_config.output_dir is None
    
    def test_preset_with_core_config(self):
        """Test preset application works with core config integration."""
        cfg = PipelineConfig(preset=Preset.INTERIOR_LUXURY)
        cfg.apply_preset()
        
        # Verify preset applied
        assert cfg.material_strength == 0.90
        
        # Verify core config still works
        device_config = cfg.get_device_config()
        assert device_config is not None
        
        paths_config = cfg.get_paths_config()
        assert paths_config is not None
    
    def test_all_presets_with_core_config(self):
        """Test all presets work with core config integration."""
        for preset in Preset:
            cfg = PipelineConfig(preset=preset)
            cfg.apply_preset()
            
            # Verify core configs can be generated
            device_config = cfg.get_device_config()
            assert device_config is not None
            
            paths_config = cfg.get_paths_config()
            assert paths_config is not None


@pytest.mark.skipif(CORE_CONFIG_AVAILABLE, reason="Test fallback when core not available")
class TestCoreConfigFallback:
    """Test graceful fallback when Platform Core not available."""
    
    def test_get_device_config_returns_none(self):
        """Test get_device_config returns None when core not available."""
        cfg = PipelineConfig()
        assert cfg.get_device_config() is None
    
    def test_get_paths_config_returns_none(self):
        """Test get_paths_config returns None when core not available."""
        cfg = PipelineConfig()
        assert cfg.get_paths_config() is None
    
    def test_config_still_works_without_core(self):
        """Test config still works normally without core module."""
        cfg = PipelineConfig(preset=Preset.PHOTO_REALISTIC)
        cfg.apply_preset()
        
        # Verify config works as expected
        assert cfg.device == "auto"
        assert cfg.precision == "fp16"
        assert cfg.material_strength == 0.70


@pytest.mark.skipif(not torch_ops.CORE_DEVICE_AVAILABLE, reason="Platform Core device not available")
class TestCoreDeviceIntegration:
    """Test integration with Platform Core device detection."""
    
    def test_pick_device_auto_uses_core(self):
        """Test pick_device uses core detector in auto mode."""
        device = torch_ops.pick_device("auto")
        assert device is not None
        # Should return a valid torch device
        assert hasattr(device, 'type')
    
    def test_pick_device_explicit_still_works(self):
        """Test explicit device selection still works."""
        device = torch_ops.pick_device("cpu")
        assert device.type == "cpu"
    
    def test_get_device_info_returns_capabilities(self):
        """Test get_device_info returns device capabilities."""
        info = torch_ops.get_device_info()
        assert info is not None
        assert hasattr(info, 'device')
        assert hasattr(info, 'capabilities')
        
        # Check capabilities structure
        caps = info.capabilities
        assert hasattr(caps, 'device_type')
        assert hasattr(caps, 'total_memory_gb')
        assert hasattr(caps, 'available_memory_gb')
        assert hasattr(caps, 'neural_engine_available')
    
    def test_device_detector_caching(self):
        """Test device detector is cached between calls."""
        info1 = torch_ops.get_device_info()
        info2 = torch_ops.get_device_info()
        # Both should use same cached detector
        assert info1 is not None
        assert info2 is not None
    
    def test_pick_device_with_memory_fraction(self):
        """Test pick_device accepts memory_fraction parameter."""
        device = torch_ops.pick_device("auto", memory_fraction=0.75)
        assert device is not None


@pytest.mark.skipif(torch_ops.CORE_DEVICE_AVAILABLE, reason="Test fallback when core not available")
class TestDeviceFallback:
    """Test device detection fallback when Platform Core not available."""
    
    def test_get_device_info_returns_none(self):
        """Test get_device_info returns None when core not available."""
        assert torch_ops.get_device_info() is None
    
    def test_pick_device_fallback_works(self):
        """Test legacy pick_device fallback works."""
        device = torch_ops.pick_device("cpu")
        assert device.type == "cpu"


@pytest.mark.skipif(not CORE_CONFIG_AVAILABLE, reason="Platform Core not available")
class TestCoreSecurityIntegration:
    """Test integration with Platform Core security/validation."""
    
    def test_validate_image_file_with_core(self, tmp_path):
        """Test validate_image_file uses core validator when available."""
        from lux_depth_v2.hardening.safe_io import validate_image_file, CORE_VALIDATION_AVAILABLE
        from lux_depth_v2.hardening.policy import HardeningPolicy
        
        if not CORE_VALIDATION_AVAILABLE:
            pytest.skip("Core validation not available")
        
        # Create a valid test image
        test_img = tmp_path / "test.png"
        # Write PNG header
        test_img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        
        policy = HardeningPolicy(
            allowed_input_exts=(".png",),
            max_input_bytes=1024 * 1024,  # 1MB
            enforce_output_within=str(tmp_path)
        )
        
        # Should pass validation with core enabled
        validate_image_file(test_img, policy, use_core=True)
    
    def test_validate_image_file_legacy_fallback(self, tmp_path):
        """Test validate_image_file works without core validator."""
        from lux_depth_v2.hardening.safe_io import validate_image_file
        from lux_depth_v2.hardening.policy import HardeningPolicy
        
        # Create a valid test image
        test_img = tmp_path / "test.png"
        test_img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        
        policy = HardeningPolicy(
            allowed_input_exts=(".png",),
            max_input_bytes=1024 * 1024,
            enforce_output_within=str(tmp_path)
        )
        
        # Should pass validation with core disabled (legacy mode)
        validate_image_file(test_img, policy, use_core=False)
    
    def test_validate_image_file_rejects_invalid(self, tmp_path):
        """Test validate_image_file rejects invalid files."""
        from lux_depth_v2.hardening.safe_io import validate_image_file, InputValidationError
        from lux_depth_v2.hardening.policy import HardeningPolicy
        
        # Create a file with wrong extension
        test_img = tmp_path / "test.png"
        test_img.write_bytes(b"\xFF\xD8\xFF" + b"\x00" * 100)  # JPEG magic bytes, but .png extension
        
        policy = HardeningPolicy(
            allowed_input_exts=(".png",),
            max_input_bytes=1024 * 1024,
            enforce_output_within=str(tmp_path)
        )
        
        # Should raise InputValidationError
        with pytest.raises(InputValidationError):
            validate_image_file(test_img, policy, use_core=True)
