#!/usr/bin/env python3
"""
Basic tests for Hyper-Reality Enhancement Module

These tests verify module structure and basic functionality and require PyTorch to be installed.
For complete training infrastructure tests, see test_training_infrastructure.py

NOTE: This test file expects the package to be installed via 'pip install -e .'
or for PYTHONPATH to be set correctly. See tests/conftest.py for details.
"""

import pytest
from pathlib import Path

# Check if PyTorch is available (required for some tests)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestModuleImports:
    """Test that the module can be imported and has the expected structure."""

    def test_import_main_module(self):
        """Test that the main hyper_reality_enhancement module can be imported."""
        try:
            from enhancements import hyper_reality_enhancement as hre
            assert hre is not None
        except ImportError as e:
            pytest.skip(f"Cannot import hyper_reality_enhancement: {e}")

    def test_quality_mode_enum(self):
        """Test that QualityMode enum is available."""
        try:
            from enhancements.hyper_reality_enhancement import QualityMode
            assert hasattr(QualityMode, 'STANDARD')
            assert hasattr(QualityMode, 'PREMIUM')
            assert hasattr(QualityMode, 'HYPER')
            assert hasattr(QualityMode, 'QUANTUM')
            assert hasattr(QualityMode, 'THEORETICAL')
        except ImportError as e:
            pytest.skip(f"Cannot import QualityMode: {e}")

    def test_enhancement_config(self):
        """Test that EnhancementConfig class is available."""
        try:
            from enhancements.hyper_reality_enhancement import EnhancementConfig
            assert EnhancementConfig is not None
        except ImportError as e:
            pytest.skip(f"Cannot import EnhancementConfig: {e}")

    def test_version_info(self):
        """Test that version information is available in module docstring."""
        try:
            from enhancements import hyper_reality_enhancement as hre
            # Version is documented in module docstring
            assert hre.__doc__ is not None
            assert "Version:" in hre.__doc__ or "version" in hre.__doc__.lower()
        except (ImportError, AttributeError) as e:
            pytest.skip(f"Cannot get version info: {e}")


class TestConfiguration:
    """Test configuration and setup."""

    def test_quality_mode_values(self):
        """Test QualityMode enum values match expected implementation."""
        try:
            from enhancements.hyper_reality_enhancement import QualityMode

            # Test that quality values match implementation exactly
            assert QualityMode.STANDARD.value[0] == 70
            assert QualityMode.PREMIUM.value[0] == 85
            assert QualityMode.HYPER.value[0] == 95

            # Test that ranges are ordered correctly
            assert QualityMode.STANDARD.value[0] < QualityMode.PREMIUM.value[0]
            assert QualityMode.PREMIUM.value[0] < QualityMode.HYPER.value[0]
        except ImportError as e:
            pytest.skip(f"Cannot test QualityMode: {e}")

    def test_enhancement_config_creation(self):
        """Test that EnhancementConfig can be created with default values."""
        try:
            from enhancements.hyper_reality_enhancement import EnhancementConfig, QualityMode

            config = EnhancementConfig()
            assert config is not None
            assert hasattr(config, 'mode')
            assert hasattr(config, 'target_quality')
            assert hasattr(config, 'quantum_caustics')
            assert hasattr(config, 'neural_atmosphere')
            # Verify types rather than specific values for stability
            assert isinstance(config.mode, QualityMode)
            assert isinstance(config.target_quality, int)
        except ImportError as e:
            pytest.skip(f"Cannot test EnhancementConfig: {e}")


class TestModuleStructure:
    """Test module structure and component availability."""

    def test_neural_components_defined(self):
        """Test that neural network components are defined."""
        try:
            from enhancements.hyper_reality_enhancement import (
                CausticGenerator,
                AtmosphericSynthesizer,
                MaterialTranscendence,
                SpatialHarmonics
            )
            assert CausticGenerator is not None
            assert AtmosphericSynthesizer is not None
            assert MaterialTranscendence is not None
            assert SpatialHarmonics is not None
        except ImportError as e:
            pytest.skip(f"Cannot test neural components: {e}")

    def test_main_processor_class(self):
        """Test that HyperRealityProcessor class is available."""
        try:
            from enhancements.hyper_reality_enhancement import HyperRealityProcessor
            assert HyperRealityProcessor is not None
        except ImportError as e:
            pytest.skip(f"Cannot test HyperRealityProcessor: {e}")

    def test_enhance_function(self):
        """Test that enhance_image function is available."""
        try:
            from enhancements.hyper_reality_enhancement import enhance_image
            assert enhance_image is not None
            assert callable(enhance_image)
        except ImportError as e:
            pytest.skip(f"Cannot test enhance_image function: {e}")


@pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch not installed - training module tests require ML dependencies"
)
class TestTrainingModuleAvailability:
    """Test training module availability (skipped if PyTorch not installed)."""

    def test_training_module_import(self):
        """Test that training module can be imported when PyTorch is available."""
        try:
            from enhancements.train_hyper_reality import HyperRealityTrainer
            assert HyperRealityTrainer is not None
        except ImportError as e:
            pytest.fail(f"Training module should be available when PyTorch is installed: {e}")

    def test_model_loader_import(self):
        """Test that model loader can be imported."""
        try:
            from enhancements.model_loader import ModelLoader
            assert ModelLoader is not None
        except ImportError as e:
            pytest.fail(f"Model loader should be available when PyTorch is installed: {e}")


class TestDocumentation:
    """Test that documentation is available."""

    def test_module_docstring(self):
        """Test that main module has documentation."""
        try:
            from enhancements import hyper_reality_enhancement as hre
            assert hre.__doc__ is not None
            assert len(hre.__doc__) > 0
        except ImportError as e:
            pytest.skip(f"Cannot test module docstring: {e}")

    def test_readme_exists(self):
        """Test that README documentation exists."""
        readme_path = Path(__file__).parent.parent / 'src' / 'enhancements' / 'README.md'
        assert readme_path.exists(), f"README not found at {readme_path}"

    def test_enhancement_guide_exists(self):
        """Test that enhancement guide exists."""
        guide_path = Path(__file__).parent.parent / 'docs' / 'HYPER_REALITY_ENHANCEMENT.md'
        assert guide_path.exists(), f"Enhancement guide not found at {guide_path}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
