from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lux_depth_v2.observability.smoke_app import app
from lux_depth_v2.observability.json_logging import JsonFormatter, RequestIdFilter
from lux_depth_v2.observability.context import bind_request_id


@pytest.fixture(autouse=True)
def _env():
    # Save original values
    keys = ["LUX_METRICS_ENABLED", "LUX_HTTP_ACCESS_LOG"]
    original = {k: os.environ.get(k) for k in keys}
    os.environ["LUX_METRICS_ENABLED"] = "1"
    os.environ["LUX_HTTP_ACCESS_LOG"] = "0"  # keep test output clean
    try:
        yield
    finally:
        # Restore original values
        for k, v in original.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_request_id_propagation():
    c = TestClient(app)

    r1 = c.get("/health")
    assert r1.status_code == 200
    assert "x-request-id" in {k.lower(): v for k, v in r1.headers.items()}

    r2 = c.get("/health", headers={"X-Request-ID": "test-req-123"})
    assert r2.status_code == 200
    assert r2.headers.get("x-request-id") == "test-req-123"


def test_metrics_endpoint():
    c = TestClient(app)

    # hit any endpoint to ensure metrics have something to report
    c.get("/health")
    m = c.get("/metrics")
    assert m.status_code in (200, 404, 503)  # 404 when disabled, 503 when prometheus lib missing
    if m.status_code == 200:
        txt = m.text
        assert "lux_http_requests_total" in txt
        assert "lux_http_request_duration_seconds" in txt


def test_json_formatter_emits_request_id():
    fmt = JsonFormatter()
    flt = RequestIdFilter()

    import logging

    logger = logging.getLogger("lux_depth_v2.test_json")
    record = logger.makeRecord(
        name=logger.name,
        level=logging.INFO,
        fn=__file__,
        lno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )

    with bind_request_id("rid-abc"):
        assert flt.filter(record) is True
        out = fmt.format(record)

    data = json.loads(out)
    assert data["msg"] == "hello"
    assert data["request_id"] == "rid-abc"


def test_service_file_is_wired_additively():
    """
    Static integration check: ensure service.py was patched to call install_observability
    without importing service (avoids heavy model init in CI).
    """
    spec = importlib.util.find_spec("lux_depth_v2.service")
    if not spec or not spec.origin:
        pytest.skip("lux_depth_v2.service not importable in this environment")

    text = Path(spec.origin).read_text(encoding="utf-8", errors="ignore")
    assert "install_observability" in text
