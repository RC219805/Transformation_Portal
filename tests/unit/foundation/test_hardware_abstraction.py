"""Unit tests for foundation.hardware_abstraction.

Covers CoreML availability gating, fallback-chain execution semantics and
device moves of nested containers — all on CPU with mocked availability probes.
"""

from __future__ import annotations

import builtins
import collections
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

pytestmark = [pytest.mark.unit, pytest.mark.ml]


def _cpu_only_registry(system: str = "Linux", coremltools_importable: bool = False):
    from transformation_portal.foundation.hardware_abstraction import BackendRegistry

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "coremltools":
            if coremltools_importable:
                return object()
            raise ImportError("No module named 'coremltools'")
        return real_import(name, *args, **kwargs)

    with (
        patch("torch.backends.mps.is_available", return_value=False),
        patch("torch.cuda.is_available", return_value=False),
        patch("platform.system", return_value=system),
        patch("builtins.__import__", side_effect=fake_import),
    ):
        return BackendRegistry()


def _cpu_only_hal():
    from transformation_portal.foundation.hardware_abstraction import HardwareAbstraction

    with (
        patch("torch.backends.mps.is_available", return_value=False),
        patch("torch.cuda.is_available", return_value=False),
        patch("platform.system", return_value="Linux"),
    ):
        return HardwareAbstraction()


class TestCoreMLAvailability:
    def test_coreml_not_reported_off_darwin(self):
        from transformation_portal.foundation.hardware_abstraction import BackendType

        registry = _cpu_only_registry(system="Linux", coremltools_importable=True)
        assert registry.get_available_backends() == [BackendType.CPU]

    def test_coreml_not_reported_on_darwin_without_coremltools(self):
        from transformation_portal.foundation.hardware_abstraction import BackendType

        registry = _cpu_only_registry(system="Darwin", coremltools_importable=False)
        assert BackendType.COREML not in registry.get_available_backends()

    def test_coreml_reported_on_darwin_with_coremltools(self):
        from transformation_portal.foundation.hardware_abstraction import BackendType

        registry = _cpu_only_registry(system="Darwin", coremltools_importable=True)
        assert BackendType.COREML in registry.get_available_backends()

    def test_cpu_only_linux_host_selects_cpu_primary(self):
        from transformation_portal.foundation.hardware_abstraction import BackendType

        hal = _cpu_only_hal()
        assert hal.primary_backend == BackendType.CPU
        assert hal.fallback_chain == [BackendType.CPU]


class TestExecuteWithFallback:
    def test_caller_error_propagates_unchanged_and_runs_once(self):
        hal = _cpu_only_hal()
        calls = []

        def bad(x):
            calls.append(x.device.type)
            raise ValueError("bad input")

        with pytest.raises(ValueError, match="bad input"):
            hal.execute_with_fallback(bad, torch.ones(2), operation_name="op")
        assert calls == ["cpu"]

    def test_backends_sharing_a_device_are_not_retried(self):
        from transformation_portal.foundation.hardware_abstraction import BackendType

        hal = _cpu_only_hal()
        hal.fallback_chain = [BackendType.COREML, BackendType.CPU]  # both map to cpu
        calls = []

        def device_failure(x):
            calls.append(x.device.type)
            raise RuntimeError("unsupported op on this device")

        with pytest.raises(RuntimeError, match="failed on all backends") as excinfo:
            hal.execute_with_fallback(device_failure, torch.ones(2), operation_name="op")
        assert calls == ["cpu"]
        assert isinstance(excinfo.value.__cause__, RuntimeError)

    def test_device_error_falls_back_to_next_device(self):
        from transformation_portal.foundation.hardware_abstraction import BackendType

        hal = _cpu_only_hal()
        hal.primary_backend = BackendType.CUDA
        hal.fallback_chain = [BackendType.CUDA, BackendType.CPU]
        moved_to = []

        def fake_move(obj, device):
            moved_to.append(device.type)
            if device.type == "cuda":
                raise RuntimeError("CUDA unavailable")
            return obj

        with patch.object(hal, "_move_to_device", side_effect=fake_move):
            result = hal.execute_with_fallback(lambda x: x + 1, torch.ones(2), operation_name="op")

        assert torch.equal(result, torch.full((2,), 2.0))
        assert moved_to[0] == "cuda" and "cpu" in moved_to

    def test_namedtuple_argument_is_rebuilt(self):
        hal = _cpu_only_hal()
        Pair = collections.namedtuple("Pair", "image mask")

        result = hal.execute_with_fallback(
            lambda p: p.image + p.mask, Pair(torch.ones(2), torch.ones(2)), operation_name="blend"
        )
        assert torch.equal(result, torch.full((2,), 2.0))

    def test_move_to_device_preserves_namedtuple_type(self):
        hal = _cpu_only_hal()
        Pair = collections.namedtuple("Pair", "image mask")

        moved = hal.to_device(Pair(torch.ones(1), torch.zeros(1)))
        assert type(moved) is Pair
        assert torch.equal(moved.mask, torch.zeros(1))

    def test_plain_tuple_and_list_still_supported(self):
        hal = _cpu_only_hal()
        moved = hal.to_device((torch.ones(1), [torch.zeros(1)]))
        assert type(moved) is tuple and type(moved[1]) is list
