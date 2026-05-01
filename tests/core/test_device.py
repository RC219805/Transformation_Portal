"""Tests for core.device — DeviceType, memory estimates, and batch size."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from transformation_portal.core.device.detector import DeviceType
from transformation_portal.core.device.memory import (
    calculate_safe_batch_size,
    estimate_memory_usage,
)

pytestmark = pytest.mark.unit


class TestDeviceType:
    def test_cpu_enum_value(self):
        """DeviceType.CPU has value 'cpu'."""
        assert DeviceType.CPU == "cpu"

    def test_cuda_enum_value(self):
        """DeviceType.CUDA has value 'cuda'."""
        assert DeviceType.CUDA == "cuda"

    def test_mps_enum_value(self):
        """DeviceType.MPS has value 'mps'."""
        assert DeviceType.MPS == "mps"

    def test_all_three_values_distinct(self):
        """All three device types are distinct."""
        types = {DeviceType.CPU, DeviceType.CUDA, DeviceType.MPS}
        assert len(types) == 3


class TestEstimateMemoryUsage:
    def test_basic_calculation(self):
        """Known resolution returns expected MB estimate."""
        # (1920x1080) * 3 channels * 2 bytes (FP16) * 6 overhead / 1024^2
        result = estimate_memory_usage((1080, 1920))
        expected = (1080 * 1920 * 3 * 2 / (1024 ** 2)) * 6.0
        assert result == pytest.approx(expected)

    def test_higher_resolution_higher_memory(self):
        """4K resolution requires more memory than 1080p."""
        mem_1080p = estimate_memory_usage((1080, 1920))
        mem_4k = estimate_memory_usage((2160, 3840))
        assert mem_4k > mem_1080p

    def test_precision_bytes_scales_linearly(self):
        """4-byte precision requires twice the memory of 2-byte."""
        mem_fp16 = estimate_memory_usage((512, 512), precision_bytes=2)
        mem_fp32 = estimate_memory_usage((512, 512), precision_bytes=4)
        assert mem_fp32 == pytest.approx(mem_fp16 * 2.0)

    def test_more_channels_more_memory(self):
        """4 channels require more memory than 3 channels."""
        m3 = estimate_memory_usage((512, 512), channels=3)
        m4 = estimate_memory_usage((512, 512), channels=4)
        assert m4 > m3


class TestCalculateSafeBatchSize:
    def test_returns_positive_int(self):
        """Result is always at least 1."""
        result = calculate_safe_batch_size(
            available_vram_gb=8.0,
            model_weights_gb=2.0,
            image_resolution=(512, 512),
        )
        assert isinstance(result, int)
        assert result >= 1

    def test_large_model_returns_one(self):
        """When model_weights_gb is nearly all available VRAM, batch=1."""
        result = calculate_safe_batch_size(
            available_vram_gb=4.0,
            model_weights_gb=4.5,  # exceeds available VRAM
            image_resolution=(512, 512),
        )
        assert result == 1

    def test_small_model_allows_batch_greater_than_one(self):
        """Plenty of VRAM and small images allow batch > 1."""
        result = calculate_safe_batch_size(
            available_vram_gb=24.0,
            model_weights_gb=1.0,
            image_resolution=(256, 256),
        )
        assert result > 1

    def test_larger_vram_larger_batch(self):
        """More VRAM means we can fit more images in a batch."""
        batch_small = calculate_safe_batch_size(4.0, 1.0, (512, 512))
        batch_large = calculate_safe_batch_size(16.0, 1.0, (512, 512))
        assert batch_large >= batch_small


class TestDeviceDetector:
    def test_force_cpu_returns_cpu_type(self):
        """force_cpu=True always returns CPU device."""
        from transformation_portal.core.device.detector import DeviceDetector

        mock_torch = MagicMock()
        mock_torch.device.return_value = MagicMock()
        mock_psutil = MagicMock()
        mock_psutil.virtual_memory.return_value.total = 16 * (1024 ** 3)

        with (
            patch("transformation_portal.core.device.detector._get_torch", return_value=mock_torch),
            patch.dict("sys.modules", {"psutil": mock_psutil}),
        ):
            info = DeviceDetector.get_optimal_device(force_cpu=True)

        assert info.type == DeviceType.CPU

    def test_detects_cuda_when_available(self):
        """When cuda.is_available() is True, CUDA device is returned."""
        from transformation_portal.core.device.detector import DeviceDetector

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_props = MagicMock()
        mock_props.total_memory = 8 * (1024 ** 3)
        mock_props.major = 8
        mock_props.minor = 0
        mock_props.name = "RTX 3080"
        mock_torch.cuda.get_device_properties.return_value = mock_props
        mock_torch.device.return_value = MagicMock()

        with patch("transformation_portal.core.device.detector._get_torch", return_value=mock_torch):
            info = DeviceDetector.get_optimal_device()

        assert info.type == DeviceType.CUDA

    def test_falls_back_to_mps_when_no_cuda(self):
        """When CUDA unavailable but MPS available, MPS device is returned."""
        from transformation_portal.core.device.detector import DeviceDetector

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.device.return_value = MagicMock()

        with patch("transformation_portal.core.device.detector._get_torch", return_value=mock_torch):
            info = DeviceDetector.get_optimal_device()

        assert info.type == DeviceType.MPS

    def test_falls_back_to_cpu_when_no_gpu(self):
        """When neither CUDA nor MPS available, CPU device is returned."""
        from transformation_portal.core.device.detector import DeviceDetector

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.device.return_value = MagicMock()
        mock_psutil = MagicMock()
        mock_psutil.virtual_memory.return_value.total = 16 * (1024 ** 3)

        with (
            patch("transformation_portal.core.device.detector._get_torch", return_value=mock_torch),
            patch.dict("sys.modules", {"psutil": mock_psutil}),
        ):
            info = DeviceDetector.get_optimal_device()

        assert info.type == DeviceType.CPU
