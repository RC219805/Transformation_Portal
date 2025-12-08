from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Dict, Optional, Tuple

from .context import REQUEST_ID_HEADERS, bind_request_id, new_request_id
from .metrics import get_metrics

logger = logging.getLogger("lux_depth_v2.http")


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _headers_dict(raw_headers: list[tuple[bytes, bytes]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in raw_headers:
        try:
            out[k.decode("latin-1").lower()] = v.decode("latin-1")
        except Exception:
            continue
    return out


class ObservabilityMiddleware:
    """
    ASGI middleware:
    - Binds request_id to a contextvar for request correlation
    - Adds X-Request-ID to response
    - Emits Prometheus metrics (if enabled)
    - Emits structured access logs (if enabled)
    """

    def __init__(self, app: Any, service_name: str = "lux_depth_v2") -> None:
        self.app = app
        self.service_name = service_name
        self.access_log = _env_bool("LUX_HTTP_ACCESS_LOG", True)
        self.metrics = get_metrics()

    async def __call__(self, scope: Dict[str, Any], receive: Callable, send: Callable) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "GET")
        path = scope.get("path", "")

        hdrs = _headers_dict(scope.get("headers") or [])
        request_id = None
        for h in REQUEST_ID_HEADERS:
            if h in hdrs and hdrs[h].strip():
                request_id = hdrs[h].strip()
                break
        if request_id is None:
            request_id = new_request_id()

        # Capture status_code + inject response header
        status_code: int = 500
        route_template = None

        start = time.perf_counter()
        self.metrics.inflight_inc()

        async def send_wrapper(message: Dict[str, Any]) -> None:
            nonlocal status_code, route_template
            if message.get("type") == "http.response.start":
                status_code = int(message.get("status", 500))

                # Try to use route template when available (reduces cardinality)
                r = scope.get("route")
                route_template = getattr(r, "path", None) or path

                headers: list[Tuple[bytes, bytes]] = list(message.get("headers") or [])
                headers.append((b"x-request-id", request_id.encode("latin-1")))
                message["headers"] = headers
            await send(message)

        try:
            with bind_request_id(request_id):
                await self.app(scope, receive, send_wrapper)
        except Exception as e:
            self.metrics.observe_exception(where="http", exc_type=type(e).__name__)
            if self.access_log:
                logger.exception(
                    "http_request_failed",
                    extra={
                        "event": "http_request",
                        "service": self.service_name,
                        "method": method,
                        "path": path,
                        "route": route_template or path,
                        "status_code": 500,
                    },
                )
            raise
        finally:
            elapsed = time.perf_counter() - start
            self.metrics.inflight_dec()

            # Metrics (skip /metrics itself to avoid self-noise)
            if path != "/metrics":
                self.metrics.observe_http(
                    method=str(method),
                    route=str(route_template or path),
                    status_code=int(status_code),
                    duration_s=float(elapsed),
                )

            if self.access_log:
                client = scope.get("client") or ("", 0)
                ua = hdrs.get("user-agent")
                logger.info(
                    "http_request",
                    extra={
                        "event": "http_request",
                        "service": self.service_name,
                        "method": method,
                        "path": path,
                        "route": route_template or path,
                        "status_code": status_code,
                        "duration_ms": round(elapsed * 1000.0, 3),
                        "client_ip": client[0],
                        "user_agent": ua,
                    },
                )
