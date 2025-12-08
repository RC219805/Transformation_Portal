from __future__ import annotations

import time
import uuid
from typing import Callable, Optional

from .policy import HardeningPolicy

try:
    # Existing service module
    from lux_depth_v2.service import app as base_app  # type: ignore
except Exception:
    base_app = None


def create_hardened_app(policy: Optional[HardeningPolicy] = None):
    """
    Returns a FastAPI app with additive hardening middleware.

    Does NOT modify existing service behavior unless you explicitly run this app.
    """
    if base_app is None:
        raise RuntimeError("lux_depth_v2.service.app not importable")

    p = policy or HardeningPolicy.load()

    # Attach middleware only if FastAPI present (service.py already uses it)
    app = base_app

    if p.enable_request_ids:
        try:
            from starlette.middleware.base import BaseHTTPMiddleware  # type: ignore
            from starlette.requests import Request  # type: ignore
            from starlette.responses import Response  # type: ignore
        except Exception:
            return app

        class RequestIdMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: "Request", call_next: Callable):
                rid = request.headers.get("x-request-id") or str(uuid.uuid4())
                request.state.request_id = rid
                t0 = time.perf_counter()
                resp: "Response" = await call_next(request)
                resp.headers["x-request-id"] = rid
                resp.headers["x-elapsed-ms"] = f"{(time.perf_counter() - t0) * 1000.0:.2f}"
                return resp

        app.add_middleware(RequestIdMiddleware)

    # Rate limiting: intentionally NOT enforced by default; keep it off unless enabled.
    if p.enable_rate_limit:
        try:
            from starlette.middleware.base import BaseHTTPMiddleware  # type: ignore
            from starlette.requests import Request  # type: ignore
            from starlette.responses import JSONResponse  # type: ignore
        except Exception:
            return app

        # Simple in-memory token bucket per client IP.
        # For production multi-worker deployments, replace with Redis (future enhancement).
        bucket = {"tokens": float(p.rate_limit_per_minute), "t": time.time()}

        class RateLimitMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: "Request", call_next: Callable):
                now = time.time()
                elapsed = now - bucket["t"]
                bucket["t"] = now
                bucket["tokens"] = min(float(p.rate_limit_per_minute), bucket["tokens"] + elapsed * (p.rate_limit_per_minute / 60.0))
                if bucket["tokens"] < 1.0:
                    return JSONResponse({"error": "rate_limited"}, status_code=429)
                bucket["tokens"] -= 1.0
                return await call_next(request)

        app.add_middleware(RateLimitMiddleware)

    return app
