#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for Depth Anything V2 ONNX backend implementation.

These tests validate the ONNX backend implementation without requiring
actual ONNX runtime or models to be installed.
"""
# pylint: disable=redefined-outer-name  # pytest fixtures

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from scripts.utilities.depth_anything_v2 import (
    ModelBackend,
    ModelVariant,
    DepthAnythingV2Model,
    ONNX_AVAILABLE,
    ONNX_MODEL_FILENAMES,
)


@pytest.fixture
def sample_image():
    """Create a sample RGB image for testing."""
    return Image.new('RGB', (100, 100), color='red')


class TestONNXBackendAvailability:
    """Test ONNX backend availability detection."""

    def test_onnx_available_flag_exists(self):
        """Test that ONNX_AVAILABLE flag is defined."""
        # The flag should exist (may be True or False depending on environment)
        assert ONNX_AVAILABLE is not None
        assert isinstance(ONNX_AVAILABLE, bool)

    def test_onnx_backend_enum_exists(self):
        """Test that ONNX backend is defined in ModelBackend enum."""
        assert hasattr(ModelBackend, 'ONNX')
        assert ModelBackend.ONNX.value == 'onnx'


class TestModelBackendEnum:
    """Test ModelBackend enum completeness."""

    def test_all_backends_defined(self):
        """Test that all expected backends are defined."""
        expected_backends = ['PYTORCH_CPU', 'PYTORCH_MPS', 'COREML', 'ONNX']
        for backend in expected_backends:
            assert hasattr(ModelBackend, backend), f"Missing backend: {backend}"

    def test_backend_values(self):
        """Test that backend enum values are correct."""
        assert ModelBackend.PYTORCH_CPU.value == 'pytorch_cpu'
        assert ModelBackend.PYTORCH_MPS.value == 'pytorch_mps'
        assert ModelBackend.COREML.value == 'coreml'
        assert ModelBackend.ONNX.value == 'onnx'


class TestModelVariant:
    """Test ModelVariant enum."""

    def test_all_variants_defined(self):
        """Test that all expected model variants are defined."""
        expected_variants = ['SMALL', 'BASE', 'LARGE']
        for variant in expected_variants:
            assert hasattr(ModelVariant, variant), f"Missing variant: {variant}"


class TestONNXInference:
    """Test ONNX inference functionality."""

    def test_estimate_depth_onnx_method_exists(self):
        """Test that _estimate_depth_onnx method exists."""
        assert hasattr(DepthAnythingV2Model, '_estimate_depth_onnx')

    def test_load_onnx_model_method_exists(self):
        """Test that _load_onnx_model method exists."""
        assert hasattr(DepthAnythingV2Model, '_load_onnx_model')

    def test_download_onnx_model_method_exists(self):
        """Test that _download_onnx_model method exists."""
        assert hasattr(DepthAnythingV2Model, '_download_onnx_model')

    def test_get_onnx_providers_method_exists(self):
        """Test that _get_onnx_providers method exists."""
        assert hasattr(DepthAnythingV2Model, '_get_onnx_providers')


class TestAutoDetectDevice:
    """Test auto-detect device functionality includes ONNX."""

    def test_auto_detect_device_returns_onnx_for_onnx_backend(self):
        """Test that auto-detect device returns 'onnx' for ONNX backend."""
        # Create a mock model instance
        model = DepthAnythingV2Model.__new__(DepthAnythingV2Model)
        model.backend = ModelBackend.ONNX

        # Test the device detection
        device = model._auto_detect_device()
        assert device == 'onnx'


class TestONNXDownloadMapping:
    """Test ONNX model download mapping."""

    def test_onnx_filename_mapping_constant_exists(self):
        """Test that ONNX_MODEL_FILENAMES constant is defined correctly."""
        expected_mapping = {
            ModelVariant.SMALL: "depth_anything_v2_vits.onnx",
            ModelVariant.BASE: "depth_anything_v2_vitb.onnx",
            ModelVariant.LARGE: "depth_anything_v2_vitl.onnx",
        }

        # Verify the constant matches expected mapping
        assert ONNX_MODEL_FILENAMES == expected_mapping

    def test_onnx_download_uses_correct_filename(self):
        """Test that _download_onnx_model uses correct filename for each variant."""
        pytest.importorskip('huggingface_hub')

        model = DepthAnythingV2Model.__new__(DepthAnythingV2Model)

        with patch('huggingface_hub.hf_hub_download') as mock_download:
            mock_download.return_value = '/fake/path/model.onnx'

            # Test SMALL variant
            model.variant = ModelVariant.SMALL
            model._download_onnx_model()
            mock_download.assert_called_with(
                repo_id='onnx/Depth-Anything-V2',
                filename='depth_anything_v2_vits.onnx',
                cache_dir=Path.home() / ".cache" / "depth_anything_v2"
            )

            # Test BASE variant
            model.variant = ModelVariant.BASE
            model._download_onnx_model()
            mock_download.assert_called_with(
                repo_id='onnx/Depth-Anything-V2',
                filename='depth_anything_v2_vitb.onnx',
                cache_dir=Path.home() / ".cache" / "depth_anything_v2"
            )

            # Test LARGE variant
            model.variant = ModelVariant.LARGE
            model._download_onnx_model()
            mock_download.assert_called_with(
                repo_id='onnx/Depth-Anything-V2',
                filename='depth_anything_v2_vitl.onnx',
                cache_dir=Path.home() / ".cache" / "depth_anything_v2"
            )


class TestLoadModelRouting:
    """Test that _load_model routes to correct backend loader."""

    def test_load_model_calls_onnx_loader_for_onnx_backend(self):
        """Test that _load_model calls _load_onnx_model for ONNX backend."""
        model = DepthAnythingV2Model.__new__(DepthAnythingV2Model)
        model.backend = ModelBackend.ONNX

        with patch.object(model, '_load_onnx_model') as mock_load:
            model._load_model()
            mock_load.assert_called_once()


class TestEstimateDepthRouting:
    """Test that estimate_depth routes to correct inference method."""

    def test_estimate_depth_routes_to_onnx(self, sample_image):
        """Test that estimate_depth routes to ONNX inference for ONNX backend."""
        model = DepthAnythingV2Model.__new__(DepthAnythingV2Model)
        model.backend = ModelBackend.ONNX
        model.model = MagicMock()

        # Mock the ONNX inference method
        mock_result = {
            'depth': np.zeros((100, 100), dtype=np.float32),
            'depth_raw': np.zeros((100, 100), dtype=np.float32),
            'metadata': {'backend': 'onnx'}
        }

        with patch.object(model, '_estimate_depth_onnx', return_value=mock_result) as mock_onnx:
            result = model.estimate_depth(sample_image)
            mock_onnx.assert_called_once()
            assert result['metadata']['backend'] == 'onnx'


class TestONNXModelLoadingWithMock:
    """Test ONNX model loading with mocked onnxruntime."""

    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="onnxruntime not installed")
    def test_get_onnx_providers_returns_list(self):
        """Test that _get_onnx_providers returns a list when onnxruntime is available."""
        model = DepthAnythingV2Model.__new__(DepthAnythingV2Model)
        providers = model._get_onnx_providers()
        assert isinstance(providers, list)
        assert len(providers) > 0
        assert 'CPUExecutionProvider' in providers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
