"""Unit tests for foundation.device_manager.

Covers device-type detection, capability dataclasses, backend priority ordering,
and optimization config generation — all without requiring real GPU hardware by
mocking the torch availability probes and platform calls.
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager(**kwargs):
    """Import and instantiate DeviceManager (deferred so mocks are applied first)."""
    from transformation_portal.foundation.device_manager import DeviceManager

    return DeviceManager(**kwargs)


def _cpu_caps():
    """Return a minimal CPU DeviceCapabilities for use in tests."""
    from transformation_portal.foundation.device_manager import DeviceCapabilities, DeviceType

    return DeviceCapabilities(
        device_type=DeviceType.CPU,
        device_name="Test CPU",
        total_memory_gb=16.0,
        available_memory_gb=13.6,
        supports_fp16=False,
        supports_bf16=False,
        supports_int8=True,
        neural_engine_available=False,
        performance_cores=8,
        efficiency_cores=4,
        gpu_cores=0,
        unified_memory=False,
        max_buffer_size_gb=6.8,
        recommended_batch_size=4,
        torch_version="2.0.0",
    )


# ---------------------------------------------------------------------------
# DeviceType enum
# ---------------------------------------------------------------------------


class TestDeviceTypeEnum:
    def test_mps_value(self):
        from transformation_portal.foundation.device_manager import DeviceType

        assert DeviceType.MPS.value == "mps"

    def test_cuda_value(self):
        from transformation_portal.foundation.device_manager import DeviceType

        assert DeviceType.CUDA.value == "cuda"

    def test_coreml_value(self):
        from transformation_portal.foundation.device_manager import DeviceType

        assert DeviceType.COREML.value == "coreml"

    def test_cpu_value(self):
        from transformation_portal.foundation.device_manager import DeviceType

        assert DeviceType.CPU.value == "cpu"

    def test_all_four_members_exist(self):
        from transformation_portal.foundation.device_manager import DeviceType

        assert len(DeviceType) == 4


# ---------------------------------------------------------------------------
# DeviceCapabilities dataclass
# ---------------------------------------------------------------------------


class TestDeviceCapabilitiesDataclass:
    def test_metal_version_defaults_to_none(self):
        caps = _cpu_caps()
        assert caps.metal_version is None

    def test_fields_round_trip(self):
        caps = _cpu_caps()
        assert caps.device_name == "Test CPU"
        assert caps.total_memory_gb == 16.0
        assert caps.supports_fp16 is False
        assert caps.supports_int8 is True
        assert caps.unified_memory is False


# ---------------------------------------------------------------------------
# DeviceInfo dataclass
# ---------------------------------------------------------------------------


class TestDeviceInfoDataclass:
    def test_contains_primary_device(self):
        from transformation_portal.foundation.device_manager import DeviceInfo, DeviceType

        caps = _cpu_caps()
        info = DeviceInfo(
            primary_device=torch.device("cpu"),
            capabilities=caps,
            backend_priority=[DeviceType.CPU],
            optimization_config={},
        )
        assert info.primary_device == torch.device("cpu")
        assert info.backend_priority == [DeviceType.CPU]


# ---------------------------------------------------------------------------
# DeviceManager initialisation
# ---------------------------------------------------------------------------


class TestDeviceManagerInit:
    def test_memory_fraction_clamped_low(self):
        mgr = _make_manager(memory_fraction=0.0)
        assert mgr.memory_fraction == pytest.approx(0.1)

    def test_memory_fraction_clamped_high(self):
        mgr = _make_manager(memory_fraction=1.0)
        assert mgr.memory_fraction == pytest.approx(0.95)

    def test_memory_fraction_preserved_in_range(self):
        mgr = _make_manager(memory_fraction=0.75)
        assert mgr.memory_fraction == pytest.approx(0.75)

    def test_prefer_ane_stored(self):
        mgr = _make_manager(prefer_ane=False)
        assert mgr.prefer_ane is False

    def test_device_info_is_none_before_detect(self):
        mgr = _make_manager()
        assert mgr.device_info is None


# ---------------------------------------------------------------------------
# Device detection: CPU fallback
# ---------------------------------------------------------------------------


class TestDeviceDetectionCPUFallback:
    """When neither MPS nor CUDA is available the manager must select CPU."""

    def test_detect_returns_cpu_device_type(self):
        from transformation_portal.foundation.device_manager import DeviceType

        with (
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
            patch(
                "transformation_portal.foundation.device_manager.DeviceManager._detect_capabilities",
                return_value=_cpu_caps(),
            ),
            patch("transformation_portal.foundation.device_manager.DeviceManager._log_device_info"),
        ):
            mgr = _make_manager()
            info = mgr.detect_devices()

        assert info.capabilities.device_type == DeviceType.CPU

    def test_detect_returns_cpu_torch_device(self):
        with (
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
            patch(
                "transformation_portal.foundation.device_manager.DeviceManager._detect_capabilities",
                return_value=_cpu_caps(),
            ),
            patch("transformation_portal.foundation.device_manager.DeviceManager._log_device_info"),
        ):
            mgr = _make_manager()
            info = mgr.detect_devices()

        assert info.primary_device == torch.device("cpu")

    def test_detect_caches_result(self):
        """detect_devices called twice returns same object without re-running."""
        with (
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
            patch(
                "transformation_portal.foundation.device_manager.DeviceManager._detect_capabilities",
                return_value=_cpu_caps(),
            ) as mock_detect,
            patch("transformation_portal.foundation.device_manager.DeviceManager._log_device_info"),
        ):
            mgr = _make_manager()
            first = mgr.detect_devices()
            second = mgr.detect_devices()

        assert first is second
        # _detect_capabilities should only be called once
        mock_detect.assert_called_once()


# ---------------------------------------------------------------------------
# Backend priority ordering
# ---------------------------------------------------------------------------


class TestBackendPriorityOrdering:
    def test_cpu_only_priority(self):
        from transformation_portal.foundation.device_manager import DeviceManager, DeviceType

        mgr = DeviceManager()
        caps = _cpu_caps()
        priority = mgr._determine_backend_priority(DeviceType.CPU, caps)
        assert priority == [DeviceType.CPU]

    def test_cuda_priority_ends_with_cpu(self):
        from transformation_portal.foundation.device_manager import DeviceCapabilities, DeviceManager, DeviceType

        mgr = DeviceManager()
        caps = DeviceCapabilities(
            device_type=DeviceType.CUDA,
            device_name="RTX 3090",
            total_memory_gb=24.0,
            available_memory_gb=20.4,
            supports_fp16=True,
            supports_bf16=True,
            supports_int8=True,
            neural_engine_available=False,
            performance_cores=0,
            efficiency_cores=0,
            gpu_cores=82,
            unified_memory=False,
            max_buffer_size_gb=16.0,
            recommended_batch_size=10,
            torch_version="2.0.0",
        )
        priority = mgr._determine_backend_priority(DeviceType.CUDA, caps)
        assert priority[0] == DeviceType.CUDA
        assert priority[-1] == DeviceType.CPU

    def test_mps_with_ane_preferred_puts_coreml_first(self):
        from transformation_portal.foundation.device_manager import DeviceCapabilities, DeviceManager, DeviceType

        mgr = DeviceManager(prefer_ane=True)
        caps = DeviceCapabilities(
            device_type=DeviceType.MPS,
            device_name="Apple M4 Max",
            total_memory_gb=128.0,
            available_memory_gb=108.8,
            supports_fp16=True,
            supports_bf16=True,
            supports_int8=True,
            neural_engine_available=True,
            performance_cores=12,
            efficiency_cores=4,
            gpu_cores=40,
            unified_memory=True,
            max_buffer_size_gb=97.9,
            recommended_batch_size=54,
            torch_version="2.0.0",
        )
        priority = mgr._determine_backend_priority(DeviceType.MPS, caps)
        assert priority[0] == DeviceType.COREML
        assert DeviceType.CPU in priority

    def test_mps_without_ane_preferred_puts_mps_first(self):
        from transformation_portal.foundation.device_manager import DeviceCapabilities, DeviceManager, DeviceType

        mgr = DeviceManager(prefer_ane=False)
        caps = DeviceCapabilities(
            device_type=DeviceType.MPS,
            device_name="Apple M1",
            total_memory_gb=16.0,
            available_memory_gb=13.6,
            supports_fp16=True,
            supports_bf16=False,
            supports_int8=True,
            neural_engine_available=True,
            performance_cores=8,
            efficiency_cores=4,
            gpu_cores=8,
            unified_memory=True,
            max_buffer_size_gb=12.0,
            recommended_batch_size=8,
            torch_version="2.0.0",
        )
        priority = mgr._determine_backend_priority(DeviceType.MPS, caps)
        assert priority[0] == DeviceType.MPS


# ---------------------------------------------------------------------------
# Optimization config
# ---------------------------------------------------------------------------


class TestOptimizationConfig:
    def test_cpu_config_has_required_keys(self):
        from transformation_portal.foundation.device_manager import DeviceManager

        mgr = DeviceManager()
        config = mgr._create_optimization_config(_cpu_caps())
        for key in ("device_type", "precision", "enable_amp", "max_batch_size", "memory_limit_gb"):
            assert key in config, f"Missing key: {key}"

    def test_cpu_config_disables_amp(self):
        from transformation_portal.foundation.device_manager import DeviceManager

        mgr = DeviceManager()
        config = mgr._create_optimization_config(_cpu_caps())
        # CPU caps has supports_fp16=False → amp should be off
        assert config["enable_amp"] is False

    def test_cpu_config_precision_is_fp32(self):
        from transformation_portal.foundation.device_manager import DeviceManager

        mgr = DeviceManager()
        config = mgr._create_optimization_config(_cpu_caps())
        assert config["precision"] == "fp32"

    def test_cpu_unified_memory_false_enables_pin_memory(self):
        from transformation_portal.foundation.device_manager import DeviceManager

        mgr = DeviceManager()
        config = mgr._create_optimization_config(_cpu_caps())
        # unified_memory=False → pin_memory should be True
        assert config["pin_memory"] is True

    def test_mps_config_includes_mps_specific_keys(self):
        from transformation_portal.foundation.device_manager import DeviceCapabilities, DeviceManager, DeviceType

        mgr = DeviceManager()
        mps_caps = DeviceCapabilities(
            device_type=DeviceType.MPS,
            device_name="Apple M4 Max",
            total_memory_gb=128.0,
            available_memory_gb=108.8,
            supports_fp16=True,
            supports_bf16=True,
            supports_int8=True,
            neural_engine_available=True,
            performance_cores=12,
            efficiency_cores=4,
            gpu_cores=40,
            unified_memory=True,
            max_buffer_size_gb=97.9,
            recommended_batch_size=54,
            torch_version="2.0.0",
        )
        config = mgr._create_optimization_config(mps_caps)
        assert "mps_allocator_strategy" in config
        assert "enable_neural_engine" in config

    def test_gradient_checkpointing_enabled_for_small_memory(self):
        from transformation_portal.foundation.device_manager import DeviceCapabilities, DeviceManager, DeviceType

        mgr = DeviceManager()
        small_caps = DeviceCapabilities(
            device_type=DeviceType.CPU,
            device_name="Low-mem CPU",
            total_memory_gb=8.0,
            available_memory_gb=6.8,
            supports_fp16=False,
            supports_bf16=False,
            supports_int8=True,
            neural_engine_available=False,
            performance_cores=4,
            efficiency_cores=0,
            gpu_cores=0,
            unified_memory=False,
            max_buffer_size_gb=3.4,
            recommended_batch_size=2,
            torch_version="2.0.0",
        )
        config = mgr._create_optimization_config(small_caps)
        assert config["enable_gradient_checkpointing"] is True

    def test_gradient_checkpointing_disabled_for_large_memory(self):
        from transformation_portal.foundation.device_manager import DeviceManager

        mgr = DeviceManager()
        caps = replace(_cpu_caps(), total_memory_gb=64.0)
        config = mgr._create_optimization_config(caps)
        assert config["enable_gradient_checkpointing"] is False


# ---------------------------------------------------------------------------
# Batch size calculation
# ---------------------------------------------------------------------------


class TestBatchSizeCalculation:
    def test_positive_memory_yields_positive_batch_size(self):
        from transformation_portal.foundation.device_manager import DeviceManager

        mgr = DeviceManager()
        batch = mgr._calculate_optimal_batch_size(available_memory_gb=64.0, gpu_cores=40)
        assert batch >= 1

    def test_zero_memory_yields_batch_size_one(self):
        from transformation_portal.foundation.device_manager import DeviceManager

        mgr = DeviceManager()
        batch = mgr._calculate_optimal_batch_size(available_memory_gb=0.5, gpu_cores=1)
        assert batch == 1

    def test_batch_size_capped_at_64(self):
        from transformation_portal.foundation.device_manager import DeviceManager

        mgr = DeviceManager()
        batch = mgr._calculate_optimal_batch_size(available_memory_gb=10_000.0, gpu_cores=10_000)
        assert batch <= 64


# ---------------------------------------------------------------------------
# Public convenience methods
# ---------------------------------------------------------------------------


class TestPublicConvenienceMethods:
    def test_get_device_triggers_detect_when_uninitialised(self):
        with (
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
            patch(
                "transformation_portal.foundation.device_manager.DeviceManager._detect_capabilities",
                return_value=_cpu_caps(),
            ),
            patch("transformation_portal.foundation.device_manager.DeviceManager._log_device_info"),
        ):
            mgr = _make_manager()
            assert mgr.device_info is None
            device = mgr.get_device()

        assert device == torch.device("cpu")

    def test_get_capabilities_returns_dataclass(self):
        with (
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
            patch(
                "transformation_portal.foundation.device_manager.DeviceManager._detect_capabilities",
                return_value=_cpu_caps(),
            ),
            patch("transformation_portal.foundation.device_manager.DeviceManager._log_device_info"),
        ):
            mgr = _make_manager()
            caps = mgr.get_capabilities()

        from transformation_portal.foundation.device_manager import DeviceCapabilities

        assert isinstance(caps, DeviceCapabilities)

    def test_get_optimization_config_returns_dict(self):
        with (
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
            patch(
                "transformation_portal.foundation.device_manager.DeviceManager._detect_capabilities",
                return_value=_cpu_caps(),
            ),
            patch("transformation_portal.foundation.device_manager.DeviceManager._log_device_info"),
        ):
            mgr = _make_manager()
            opt = mgr.get_optimization_config()

        assert isinstance(opt, dict)
        assert len(opt) > 0
