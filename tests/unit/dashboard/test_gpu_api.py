"""Unit tests for the dashboard GPU monitoring API.

The test environment has no NVIDIA GPU / NVML, so the "unavailable" paths are
exercised directly and the NVML-backed paths are covered with an injected
fake ``pynvml`` module.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from transformation_portal.dashboard import gpu_api

pytestmark = pytest.mark.unit


class _FakeUtil:
    gpu = 42


class _FakeMem:
    used = 4 * 1024**3
    total = 8 * 1024**3


class _FakePynvml:
    """Minimal stand-in for the ``pynvml`` surface used by gpu_api."""

    NVML_TEMPERATURE_GPU = 0

    def nvmlDeviceGetHandleByIndex(self, index: int) -> str:
        return f"handle-{index}"

    def nvmlDeviceGetName(self, handle: str) -> bytes:
        return b"FakeGPU 9000"

    def nvmlDeviceGetTemperature(self, handle: str, sensor: int) -> int:
        return 71

    def nvmlDeviceGetUtilizationRates(self, handle: str) -> _FakeUtil:
        return _FakeUtil()

    def nvmlDeviceGetMemoryInfo(self, handle: str) -> _FakeMem:
        return _FakeMem()

    def nvmlDeviceGetPowerUsage(self, handle: str) -> int:
        return 150_000  # milliwatts

    def nvmlDeviceGetPowerManagementLimit(self, handle: str) -> int:
        return 300_000  # milliwatts


@pytest.fixture
def fake_nvml(monkeypatch: pytest.MonkeyPatch) -> _FakePynvml:
    """Inject a fake NVML backend reporting a single GPU."""
    fake = _FakePynvml()
    monkeypatch.setattr(gpu_api, "pynvml", fake, raising=False)
    monkeypatch.setattr(gpu_api, "NVML_AVAILABLE", True)
    monkeypatch.setattr(gpu_api, "GPU_COUNT", 1)
    return fake


class TestGetGpuStatsUnavailable:
    """Behaviour when NVML is not available (the CI default)."""

    def test_get_gpu_stats_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gpu_api, "NVML_AVAILABLE", False)

        assert gpu_api.get_gpu_stats(0) is None

    def test_get_all_gpu_stats_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gpu_api, "NVML_AVAILABLE", False)
        monkeypatch.setattr(gpu_api, "GPU_COUNT", 0)

        assert gpu_api.get_all_gpu_stats() == []


class TestGetGpuStatsAvailable:
    """Behaviour with an injected fake NVML backend."""

    def test_get_gpu_stats_populates_snapshot(self, fake_nvml: _FakePynvml) -> None:
        stats = gpu_api.get_gpu_stats(0)

        assert stats is not None
        assert stats.index == 0
        assert stats.name == "FakeGPU 9000"
        assert stats.temperature == 71
        assert stats.gpu_util == 42
        assert stats.memory_used == 4 * 1024**3
        assert stats.memory_total == 8 * 1024**3
        assert stats.memory_util == pytest.approx(50.0)
        assert stats.power_draw == pytest.approx(150.0)
        assert stats.power_limit == pytest.approx(300.0)

    def test_get_gpu_stats_handles_power_query_failure(self, fake_nvml: _FakePynvml, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(handle: str) -> int:
            raise RuntimeError("power query unsupported")

        monkeypatch.setattr(fake_nvml, "nvmlDeviceGetPowerUsage", _boom)

        stats = gpu_api.get_gpu_stats(0)

        assert stats is not None
        assert stats.power_draw == 0.0
        assert stats.power_limit == 0.0

    def test_get_gpu_stats_returns_none_on_handle_failure(
        self, fake_nvml: _FakePynvml, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(index: int) -> str:
            raise RuntimeError("device gone")

        monkeypatch.setattr(fake_nvml, "nvmlDeviceGetHandleByIndex", _boom)

        assert gpu_api.get_gpu_stats(0) is None

    def test_get_all_gpu_stats_collects_each_device(self, fake_nvml: _FakePynvml) -> None:
        stats = gpu_api.get_all_gpu_stats()

        assert len(stats) == 1
        assert stats[0].name == "FakeGPU 9000"


class TestGpuRouter:
    """Tests for the FastAPI router built by create_gpu_router."""

    def _client(self) -> TestClient:
        app = FastAPI()
        app.include_router(gpu_api.create_gpu_router())
        return TestClient(app)

    def test_status_reports_unavailable_without_nvml(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gpu_api, "NVML_AVAILABLE", False)

        response = self._client().get("/api/gpu/status")

        assert response.status_code == 200
        body = response.json()
        assert body["available"] is False
        assert "NVML" in body["message"]

    def test_status_reports_gpus_with_nvml(self, fake_nvml: _FakePynvml) -> None:
        response = self._client().get("/api/gpu/status")

        assert response.status_code == 200
        body = response.json()
        assert body["available"] is True
        assert body["gpu_count"] == 1
        assert len(body["gpus"]) == 1
        gpu = body["gpus"][0]
        assert gpu["name"] == "FakeGPU 9000"
        assert gpu["memory_total_gb"] == pytest.approx(8.0)
        assert gpu["memory_util_percent"] == pytest.approx(50.0)

    def test_dashboard_route_serves_html(self) -> None:
        response = self._client().get("/api/gpu/")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "GPU Monitor" in response.text


class TestDashboardHtml:
    """Tests for the static dashboard HTML helper."""

    def test_html_is_self_contained_document(self) -> None:
        html = gpu_api.get_gpu_dashboard_html()

        assert html.startswith("<!DOCTYPE html>")
        assert "/api/gpu/stream" in html
        assert html.strip().endswith("</html>")
