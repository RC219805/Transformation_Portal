"""Behavioral coverage for ``dashboard.gpu_api``.

In the core lane NVML/pynvml is unavailable, so these tests pin the
NVML-absent behavior (the production default on non-GPU CI hosts): the
stats helpers degrade to ``None``/empty, the status route reports
unavailability, the stream emits the error frame, and the HTML/guard paths
hold. GPU-present branches require real hardware and stay out of scope.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

from fastapi import FastAPI
from fastapi.testclient import TestClient

from transformation_portal.dashboard import gpu_api


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(gpu_api.create_gpu_router())
    return TestClient(app)


def test_get_gpu_stats_none_when_nvml_unavailable() -> None:
    # NVML is not initialized in core CI.
    assert gpu_api.NVML_AVAILABLE is False
    assert gpu_api.get_gpu_stats(0) is None


def test_get_all_gpu_stats_empty_without_gpus() -> None:
    assert gpu_api.get_all_gpu_stats() == []


def test_status_reports_unavailable(client: TestClient) -> None:
    body = client.get("/api/gpu/status").json()
    assert body["available"] is False
    assert "NVML" in body["message"]


def test_stream_emits_error_frame_without_nvml(client: TestClient) -> None:
    with client.websocket_connect("/api/gpu/stream") as ws:
        frame = ws.receive_json()
        assert frame == {"error": "NVML not available"}


def test_gpu_dashboard_serves_html(client: TestClient) -> None:
    resp = client.get("/api/gpu/")
    assert resp.status_code == 200
    assert "<html" in resp.text.lower()


def test_gpu_dashboard_html_has_stream_wiring() -> None:
    html = gpu_api.get_gpu_dashboard_html()
    assert "/api/gpu/stream" in html
    assert "websocket" in html.lower()


def test_create_router_requires_fastapi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_api, "FASTAPI_AVAILABLE", False)
    with pytest.raises(ImportError, match="FastAPI is required"):
        gpu_api.create_gpu_router()
